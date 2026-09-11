"""The regression gate's comparison logic.

The gate exists to be trusted unattended, so the thing that must not be wrong is
what it calls a regression. A gate that misses a drop is worse than no gate,
because it is the reason nobody looks at the numbers any more.
"""

import pytest

from eval_gate import TOKEN_TOLERANCE, TOLERANCE, compare, render


def report(
    cases: dict[str, dict[str, float]],
    config: dict | None = None,
    tokens: dict[str, int] | None = None,
) -> dict:
    """Build a report in the shape eval_score.py writes."""
    return {
        "model": "test-model",
        "config": config or {"model": "test-model", "prompt_sha": "abc123", "max_turns": 8},
        "recorded_at": "2026-09-11T00:00:00+00:00",
        "cases": [
            {
                "case": name,
                "usage": (
                    {"calls": 3, "input_tokens": (tokens or {}).get(name, 0), "output_tokens": 0}
                    if tokens and name in tokens
                    else {}
                ),
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
    assert output.strip().endswith("FAIL: something went backwards")


def test_render_says_so_when_nothing_moved():
    assert render(compare(BASE, BASE)).strip().endswith("PASS: nothing went backwards")


# --- spend ------------------------------------------------------------------
SCORES = {"book": {"Tool Selection": 1.0, "Task Execution": 1.0}}


def test_the_same_token_count_passes():
    base = report(SCORES, tokens={"book": 2000})
    assert not compare(base, report(SCORES, tokens={"book": 2000})).failed


def test_a_big_jump_in_tokens_fails_even_with_perfect_scores():
    """The whole point: a right answer that costs twice as much has got worse."""
    base = report(SCORES, tokens={"book": 2000})
    result = compare(base, report(SCORES, tokens={"book": 4000}))

    assert result.failed
    assert result.regressions == []  # scores are untouched
    assert len(result.costlier) == 1
    assert result.costlier[0].ratio == pytest.approx(2.0)


def test_token_drift_inside_the_tolerance_is_ignored():
    """The model is not deterministic; small movement is not a regression."""
    base = report(SCORES, tokens={"book": 2000})
    drifted = int(2000 * (1 + TOKEN_TOLERANCE * 0.8))
    assert not compare(base, report(SCORES, tokens={"book": drifted})).failed


def test_spending_much_less_is_reported_as_cheaper_and_passes():
    base = report(SCORES, tokens={"book": 4000})
    result = compare(base, report(SCORES, tokens={"book": 2000}))

    assert not result.failed
    assert len(result.cheaper) == 1


def test_cases_recorded_before_spend_was_tracked_are_skipped():
    """Older reports carry no usage. They must not read as zero tokens."""
    base = report(SCORES)  # no usage at all
    result = compare(base, report(SCORES, tokens={"book": 2000}))

    assert not result.failed
    assert (result.costlier, result.cheaper) == ([], [])


def test_render_explains_a_cost_regression():
    base = report(SCORES, tokens={"book": 2000})
    output = render(compare(base, report(SCORES, tokens={"book": 4000})))

    assert "MORE EXPENSIVE" in output
    assert "2000 -> 4000 tokens (+100%)" in output
    assert output.strip().endswith("FAIL: something went backwards")
