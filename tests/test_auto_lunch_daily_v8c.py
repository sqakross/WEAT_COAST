
from services.auto_lunch_daily_v8c import reconcile_daily_lunch


def part(worked, lunch=0, **kwargs):
    return {
        "status": "closed",
        "worked": worked,
        "lunch": lunch,
        "auto_lunch_enabled": True,
        "open": False,
        "unverified": False,
        "requires_review": False,
        **kwargs,
    }


def test_multiple_shifts_one_day():
    result = reconcile_daily_lunch(
        [part(4 * 3600), part(4 * 3600)],
        work_date="2026-11-01",
        today="2026-11-02",
    )
    assert result["status"] == "FINAL"
    assert result["worked_seconds"] == 8 * 3600
    assert result["proposed_lunch_seconds"] == 1800


def test_existing_lunch_no_double_count():
    result = reconcile_daily_lunch(
        [part(8 * 3600, 1800)],
        work_date="2026-11-01",
        today="2026-11-02",
    )
    assert result["lunch_seconds"] == 1800
    assert result["proposed_lunch_seconds"] == 0


def test_today_is_preliminary():
    result = reconcile_daily_lunch(
        [part(8 * 3600)],
        work_date="2026-11-02",
        today="2026-11-02",
    )
    assert result["status"] == "PRELIMINARY"
    assert not result["final_day"]


def test_open_shift_blocks_finalization():
    result = reconcile_daily_lunch(
        [part(8 * 3600, open=True)],
        work_date="2026-11-01",
        today="2026-11-02",
    )
    assert result["status"] == "REVIEW_REQUIRED"
    assert not result["payroll_eligible"]
    assert result["proposed_lunch_seconds"] == 0


def test_voided_shift_excluded():
    result = reconcile_daily_lunch(
        [part(8 * 3600), part(5 * 3600, status="voided")],
        work_date="2026-11-01",
        today="2026-11-02",
    )
    assert result["worked_seconds"] == 8 * 3600


def test_manual_mode_no_auto_lunch():
    result = reconcile_daily_lunch(
        [part(8 * 3600, auto_lunch_enabled=False)],
        work_date="2026-11-01",
        today="2026-11-02",
    )
    assert result["proposed_lunch_seconds"] == 0


def test_proposal_never_deducts_pay():
    result = reconcile_daily_lunch(
        [part(8 * 3600)],
        work_date="2026-11-01",
        today="2026-11-02",
    )
    assert result["worked_seconds"] == 8 * 3600
    assert result["lunch_seconds"] == 0
