import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from services.employee_midnight_engine import next_midnight_utc

LA = ZoneInfo("America/Los_Angeles")


class MidnightBoundaryTests(unittest.TestCase):
    def check(self, start, expected):
        self.assertEqual(
            next_midnight_utc(start),
            expected.astimezone(timezone.utc)
        )

    def test_regular_day(self):
        self.check(
            datetime(2026, 10, 12, 8, tzinfo=LA),
            datetime(2026, 10, 13, 0, tzinfo=LA)
        )

    def test_spring_dst(self):
        self.check(
            datetime(2026, 3, 8, 1, tzinfo=LA),
            datetime(2026, 3, 9, 0, tzinfo=LA)
        )

    def test_fall_dst(self):
        self.check(
            datetime(2026, 11, 1, 1, tzinfo=LA),
            datetime(2026, 11, 2, 0, tzinfo=LA)
        )

    def test_utc_input(self):
        self.check(
            datetime(2026, 10, 12, 20, tzinfo=timezone.utc),
            datetime(2026, 10, 13, 0, tzinfo=LA)
        )


if __name__ == "__main__":
    unittest.main()
