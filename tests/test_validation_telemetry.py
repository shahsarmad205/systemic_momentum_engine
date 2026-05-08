import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class TestValidationTelemetry(unittest.TestCase):
    def test_telemetry_existence(self):
        """Task 6: Assert _MARKET_STATE_STATS exists and is a dict."""
        from model_selection import validation
        self.assertTrue(hasattr(validation, "_MARKET_STATE_STATS"))
        self.assertIsInstance(validation._MARKET_STATE_STATS, dict)
        self.assertIn("hits", validation._MARKET_STATE_STATS)
        self.assertIn("misses", validation._MARKET_STATE_STATS)
        self.assertIn("build_time_s", validation._MARKET_STATE_STATS)

    def test_telemetry_helper(self):
        """Task 4: Assert get_market_state_stats works and returns a copy."""
        from model_selection import validation
        stats = validation.get_market_state_stats()
        self.assertIsInstance(stats, dict)
        self.assertIsNot(stats, validation._MARKET_STATE_STATS, "Should be a copy")
        self.assertEqual(stats["hits"], validation._MARKET_STATE_STATS["hits"])

if __name__ == "__main__":
    unittest.main()
