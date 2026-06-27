import unittest

from ui.temp_files import safe_unlink_path


class TempFilesTests(unittest.TestCase):
    def test_safe_unlink_retries_permission_error(self) -> None:
        fake_path = FakePath([PermissionError("locked"), None])

        removed = safe_unlink_path(fake_path, attempts=2, delay_seconds=0, sleep=lambda _seconds: None)

        self.assertTrue(removed)
        self.assertEqual(2, fake_path.calls)

    def test_safe_unlink_returns_false_when_file_stays_locked(self) -> None:
        fake_path = FakePath([PermissionError("locked"), PermissionError("locked")])

        removed = safe_unlink_path(fake_path, attempts=2, delay_seconds=0, sleep=lambda _seconds: None)

        self.assertFalse(removed)
        self.assertEqual(2, fake_path.calls)

    def test_safe_unlink_treats_missing_file_as_removed(self) -> None:
        fake_path = FakePath([FileNotFoundError("missing")])

        removed = safe_unlink_path(fake_path, attempts=2, delay_seconds=0, sleep=lambda _seconds: None)

        self.assertTrue(removed)
        self.assertEqual(1, fake_path.calls)


class FakePath:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def unlink(self, missing_ok=False):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if outcome is None:
            return None
        raise outcome

    def __str__(self) -> str:
        return "fake.pdf"


if __name__ == "__main__":
    unittest.main()
