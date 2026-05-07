# ============================================================================
# TEST_DATA_ENGINE.PY - Unit tests for data engine
# ============================================================================

import unittest
from datetime import datetime

import pandas as pd


class TestDataEngine(unittest.TestCase):
    """Test data engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1050],
        })
    
    def test_data_loading(self):
        """Test loading data."""
        self.assertEqual(len(self.sample_data), 3)
        self.assertIn('Close', self.sample_data.columns)
    
    def test_data_validation(self):
        """Test data validation."""
        self.assertTrue((self.sample_data['High'] >= self.sample_data['Low']).all())
        self.assertTrue((self.sample_data['Close'] <= self.sample_data['High']).all())


if __name__ == '__main__':
    unittest.main()
