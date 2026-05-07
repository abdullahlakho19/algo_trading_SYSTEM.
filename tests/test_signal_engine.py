# ============================================================================
# TEST_SIGNAL_ENGINE.PY - Unit tests for signal generation
# ============================================================================

import unittest

import pandas as pd

from strategies.strategy_library.momentum_strategy import MomentumStrategy


class TestSignalEngine(unittest.TestCase):
    """Test signal generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample OHLCV data
        self.df = pd.DataFrame({
            'Open': [100 + i for i in range(50)],
            'High': [101 + i for i in range(50)],
            'Low': [99 + i for i in range(50)],
            'Close': [100.5 + i for i in range(50)],
            'Volume': [1000] * 50,
        })
        
        self.strategy = MomentumStrategy()
    
    def test_signal_generation(self):
        """Test signal generation."""
        signal = self.strategy.generate_signal(self.df)
        
        if signal:
            self.assertIn(signal.direction, ['BUY', 'SELL', 'HOLD'])
            self.assertTrue(0 <= signal.confidence <= 1)
    
    def test_stop_loss_below_entry_buy(self):
        """Test stop loss is below entry for BUY."""
        signal = self.strategy.generate_signal(self.df)
        
        if signal and signal.direction == 'BUY':
            self.assertLess(signal.stop_loss, signal.entry_price)


if __name__ == '__main__':
    unittest.main()
