"""The Google Calendar implementation of `CalendarBackend`.

Imported lazily by `calendar_backend.get_backend()`, so the mock path never
needs the Google libraries installed or a credentials file on disk.

Auth is the standard installed-app flow: `credentials.json` identifies the app,
the first run opens a browser to get your consent, and the resulting
`token.json` is reused and refreshed silently afterwards. Both files are
gitignored.
"""

from datetime import date as DateType, datetime, time as TimeType, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from calendar_backend import CalendarBackend, CalendarEvent

# Narrow on purpose: read and write events, no access to calendar settings,
# sharing, or the list of calendars themselves.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")


def _authorize() -> Credentials:
    creds: Credentials | None = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"{CREDENTIALS_FILE} not found. Download the Desktop OAuth client "
                "from Google Cloud Console and save it in the project root."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        # port=0 picks a free loopback port, which the Desktop client type allows.
        creds = flow.run_local_server(port=0)

    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


class GoogleCalendar(CalendarBackend):
    """Real events on a real calendar. Never used by the eval suite."""

    name = "google"

    def __init__(self, calendar_id: str = "primary") -> None:
        self.calendar_id = calendar_id
        self._service = build("calendar", "v3", credentials=_authorize(), cache_discovery=False)
        self._zone: ZoneInfo | None = None

    @property
    def zone(self) -> ZoneInfo:
        """The calendar's own timezone, so 09:00 means what it means in the UI.

        Read from an events listing rather than `calendars().get`, which needs a
        wider scope than calendar.events. One cached call either way.
        """
        if self._zone is None:
            probe = self._service.events().list(calendarId=self.calendar_id, maxResults=1).execute()
            self._zone = ZoneInfo(probe["timeZone"])
        return self._zone

    def list_events(self, day: DateType) -> list[CalendarEvent]:
        opens = datetime.combine(day, TimeType.min, tzinfo=self.zone)
        closes = opens + timedelta(days=1)

        response = (
            self._service.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=opens.isoformat(),
                timeMax=closes.isoformat(),
                # Expands recurring series into individual occurrences, which is
                # what a day view needs; without it you get the rule, not the events.
                singleEvents=True,
                orderBy="startTime",
                maxResults=250,
            )
            .execute()
        )

        events = []
        for item in response.get("items", []):
            # All-day entries carry `date` instead of `dateTime`. They are usually
            # travel or birthdays, and treating them as busy would empty the day.
            if "dateTime" not in item.get("start", {}):
                continue
            begin = datetime.fromisoformat(item["start"]["dateTime"]).astimezone(self.zone)
            end = datetime.fromisoformat(item["end"]["dateTime"]).astimezone(self.zone)
            events.append(
                CalendarEvent(
                    id=item["id"],
                    title=item.get("summary", "(no title)"),
                    date=begin.date().isoformat(),
                    start_time=begin.strftime("%H:%M"),
                    duration_minutes=int((end - begin).total_seconds() // 60),
                )
            )
        return events

    def create_event(
        self, title: str, day: DateType, start: TimeType, duration_minutes: int
    ) -> CalendarEvent:
        begin = datetime.combine(day, start, tzinfo=self.zone)
        end = begin + timedelta(minutes=duration_minutes)

        created = (
            self._service.events()
            .insert(
                calendarId=self.calendar_id,
                body={
                    "summary": title,
                    "start": {"dateTime": begin.isoformat(), "timeZone": str(self.zone)},
                    "end": {"dateTime": end.isoformat(), "timeZone": str(self.zone)},
                },
            )
            .execute()
        )

        return CalendarEvent(
            id=created["id"],
            title=title,
            date=day.isoformat(),
            start_time=start.strftime("%H:%M"),
            duration_minutes=duration_minutes,
        )


if __name__ == "__main__":
    import sys

    cal = GoogleCalendar()
    day = DateType.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else DateType.today()
    print(f"calendar timezone: {cal.zone}")
    print(f"\nevents on {day}:")
    for event in cal.list_events(day) or []:
        print(f"  {event.start_time}  {event.duration_minutes:>4}m  {event.title}")
    print(f"\n60-minute openings: {cal.find_free_slots(day, 60)}")
