# ============================================================================
# TEST_RISK_MANAGER.PY - Unit tests for risk management
# ============================================================================

import unittest

from risk.position_sizer import PositionSizer


class TestRiskManager(unittest.TestCase):
    """Test risk management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer(account_size=100000, risk_per_trade=0.02)
    
    def test_position_sizing(self):
        """Test position size calculation."""
        result = self.sizer.calculate_position_size(
            entry_price=100,
            stop_loss=98,
        )
        
        self.assertIn('position_size', result)
        self.assertGreater(result['position_size'], 0)
    
    def test_risk_amount(self):
        """Test risk amount calculation."""
        result = self.sizer.calculate_position_size(
            entry_price=100,
            stop_loss=98,
        )
        
        expected_risk = 100000 * 0.02
        self.assertEqual(result['risk_amount'], expected_risk)


if __name__ == '__main__':
    unittest.main()
