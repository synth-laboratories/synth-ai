"""Fresh cohort promotion does not use the retired PR workflow."""

from pathlib import Path
import unittest


class FreshRebuildPolicyTest(unittest.TestCase):
    def test_retired_promotion_workflow_is_absent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / ".github/workflows/promotion-boundary.yml").exists())


if __name__ == "__main__":
    unittest.main()
