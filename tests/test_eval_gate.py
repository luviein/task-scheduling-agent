"""The regression gate's comparison logic.

The gate exists to be trusted unattended, so the thing that must not be wrong is
what it calls a regression. A gate that misses a drop is worse than no gate,
because it is the reason nobody looks at the numbers any more.
"""

import pytest

from eval_gate import TOLERANCE, compare, render


def report(cases: dict[str, dict[str, float]], config: dict | None = None) -> dict:
    """Build a report in the shape eval_score.py writes."""
    return {
        "model": "test-model",
        "config": config or {"model": "test-model", "prompt_sha": "abc123", "max_turns": 8},
        "recorded_at": "2026-09-11T00:00:00+00:00",
        "cases": [
            {
                "case": name,
                "metrics": {
                    metric: {"score": score, "passed": score >= 0.5, "reason": ""}
                    for metric, score in metrics.items()
                },
            }
            for name, metrics in cases.items()
        ],
    }


BASE = report({"book": {"Tool Selection": 1.0, "Task Execution": 1.0}})


def test_identical_reports_pass():
    result = compare(BASE, BASE)
    assert not result.failed
    assert result.regressions == []
    assert result.improvements == []


def test_a_dropped_score_is_a_regression():
    current = report({"book": {"Tool Selection": 0.5, "Task Execution": 1.0}})
    result = compare(BASE, current)

    assert result.failed
    assert len(result.regressions) == 1
    change = result.regressions[0]
    assert (change.case, change.metric) == ("book", "Tool Selection")
    assert change.delta == pytest.approx(-0.5)


def test_a_raised_score_is_an_improvement_not_a_failure():
    base = report({"book": {"Tool Selection": 0.5}})
    result = compare(base, report({"book": {"Tool Selection": 1.0}}))

    assert not result.failed
    assert len(result.improvements) == 1


def test_noise_under_the_tolerance_is_ignored():
    current = report({"book": {"Tool Selection": 1.0 - TOLERANCE / 2, "Task Execution": 1.0}})
    result = compare(BASE, current)

    assert not result.failed
    assert result.regressions == []


def test_a_drop_just_over_the_tolerance_is_caught():
    current = report({"book": {"Tool Selection": 1.0 - TOLERANCE * 2, "Task Execution": 1.0}})
    assert compare(BASE, current).failed


def test_a_case_that_vanished_fails_the_gate():
    """Deleting the case you cannot pass is the cheapest way to a perfect score."""
    base = report({"book": {"Tool Selection": 1.0}, "search": {"Tool Selection": 1.0}})
    result = compare(base, report({"book": {"Tool Selection": 1.0}}))

    assert result.failed
    assert result.missing_cases == ["search"]


def test_a_new_case_is_reported_but_does_not_fail():
    current = report(
        {"book": {"Tool Selection": 1.0, "Task Execution": 1.0}, "brand_new": {"Tool Selection": 0.2}}
    )
    result = compare(BASE, current)

    assert not result.failed
    assert result.new_cases == ["brand_new"]


def test_a_metric_that_stopped_being_scored_fails():
    """Otherwise --no-judge would quietly hide every judged regression."""
    current = report({"book": {"Tool Selection": 1.0}})
    result = compare(BASE, current)

    assert result.failed
    assert result.missing_metrics == ["book / Task Execution"]


def test_config_changes_are_surfaced():
    current = report(
        {"book": {"Tool Selection": 1.0, "Task Execution": 1.0}},
        config={"model": "test-model", "prompt_sha": "different", "max_turns": 8},
    )
    result = compare(BASE, current)

    assert result.config_changes == {"prompt_sha": ("abc123", "different")}


def test_a_config_change_alone_does_not_fail_the_gate():
    """Changing the prompt is normal. It is a caveat on the numbers, not a failure."""
    current = report(
        {"book": {"Tool Selection": 1.0, "Task Execution": 1.0}},
        config={"model": "other-model", "prompt_sha": "abc123", "max_turns": 8},
    )
    assert not compare(BASE, current).failed


def test_reports_without_config_compare_cleanly():
    """Trace files predating the fingerprint still have to be usable."""
    old = {"model": "m", "cases": BASE["cases"]}
    assert not compare(old, old).failed


def test_render_names_the_case_and_both_scores():
    output = render(compare(BASE, report({"book": {"Tool Selection": 0.5, "Task Execution": 1.0}})))

    assert "REGRESSIONS" in output
    assert "book / Tool Selection" in output
    assert "1.00 -> 0.50" in output
    assert output.strip().endswith("FAIL: scores went backwards")


def test_render_says_so_when_nothing_moved():
    assert render(compare(BASE, BASE)).strip().endswith("PASS: nothing went backwards")
