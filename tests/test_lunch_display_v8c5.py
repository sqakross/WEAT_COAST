
import pytest

from services.lunch_display_v8c5 import calculate_daily_display


@pytest.mark.parametrize(
    "shift,recorded,required,auto,expected_lunch,expected_worked",
    [
        (8*3600+35*60+15, 0, 1800, True, 1800,
         8*3600+5*60+15),
        (11*3600, 0, 3600, True, 3600, 10*3600),
        (8*3600, 1200, 1800, True, 1800,
         7*3600+30*60),
        (11*3600, 1800, 3600, True, 3600,
         10*3600),
        (11*3600, 3600, 3600, True, 3600,
         10*3600),
        (8*3600, 0, 1800, False, 0, 8*3600),
        (8*3600, 1800, 3600, False, 1800,
         7*3600+30*60),
    ],
)
def test_unified_lunch(
    shift, recorded, required, auto,
    expected_lunch, expected_worked,
):
    result = calculate_daily_display(
        shift_seconds=shift,
        recorded_lunch_seconds=recorded,
        required_lunch_seconds=required,
        auto_lunch_enabled=auto,
    )

    assert result["lunch"] == expected_lunch
    assert result["worked"] == expected_worked
    assert result["payroll_approved"] is False


def test_other_unpaid_time():
    result = calculate_daily_display(
        shift_seconds=8*3600,
        recorded_lunch_seconds=0,
        required_lunch_seconds=1800,
        auto_lunch_enabled=True,
        other_unpaid_seconds=600,
    )
    assert result["worked"] == 7*3600+20*60


def test_invalid_time_rejected():
    with pytest.raises(ValueError):
        calculate_daily_display(
            shift_seconds=1800,
            recorded_lunch_seconds=0,
            required_lunch_seconds=3600,
            auto_lunch_enabled=True,
        )


def test_no_payroll_approval():
    result = calculate_daily_display(
        shift_seconds=36000,
        recorded_lunch_seconds=0,
        required_lunch_seconds=3600,
        auto_lunch_enabled=True,
    )
    assert result["calculation_only"] is True
    assert result["payroll_approved"] is False
