#!/usr/bin/env python3
# ============================================================================
# validate_environment.py - Environment & Dependencies Validation
# Run this after installation to verify everything is working
# ============================================================================

import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

class EnvironmentValidator:
    """Validates trading agent environment and dependencies."""
    
    def __init__(self):
        self.results = {
            'core': [],
            'ml': [],
            'finance': [],
            'dashboard': [],
            'optional': [],
        }
        self.total = 0
        self.passed = 0
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        py_version = sys.version_info
        required = (3, 10)
        
        if py_version >= required:
            print(f"✅ Python {py_version.major}.{py_version.minor} (required: 3.10+)")
            self.passed += 1
            return True
        else:
            print(f"❌ Python {py_version.major}.{py_version.minor} (required: 3.10+)")
            return False
    
    def check_module(self, module_name: str, category: str = 'core', 
                    display_name: str = None) -> bool:
        """Check if a module can be imported."""
        self.total += 1
        display_name = display_name or module_name
        
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name}")
            self.results[category].append((display_name, True))
            self.passed += 1
            return True
        except ImportError as e:
            print(f"❌ {display_name} - {str(e)[:50]}")
            self.results[category].append((display_name, False))
            return False
        except Exception as e:
            print(f"⚠️  {display_name} - {str(e)[:50]}")
            self.results[category].append((display_name, False))
            return False
    
    def validate_all(self):
        """Run all validation checks."""
        print("\n" + "=" * 70)
        print("🔍 TRADING AGENT ENVIRONMENT VALIDATION")
        print("=" * 70 + "\n")
        
        # Python version
        print("1️⃣  PYTHON VERSION")
        print("-" * 70)
        self.check_python_version()
        print()
        
        # Core data libraries
        print("2️⃣  CORE DATA LIBRARIES")
        print("-" * 70)
        self.check_module('numpy', 'core', 'NumPy (numerical computing)')
        self.check_module('pandas', 'core', 'Pandas (data analysis)')
        self.check_module('scipy', 'core', 'SciPy (scientific computing)')
        print()
        
        # Machine Learning
        print("3️⃣  MACHINE LEARNING LIBRARIES")
        print("-" * 70)
        self.check_module('sklearn', 'ml', 'Scikit-Learn (ML models)')
        self.check_module('xgboost', 'ml', 'XGBoost (gradient boosting)')
        self.check_module('lightgbm', 'ml', 'LightGBM (fast gradient boosting)')
        print()
        
        # Deep Learning (optional)
        print("4️⃣  DEEP LEARNING (OPTIONAL)")
        print("-" * 70)
        self.check_module('torch', 'optional', 'PyTorch (neural networks)')
        print()
        
        # Financial APIs
        print("5️⃣  FINANCIAL APIs")
        print("-" * 70)
        self.check_module('alpaca_py', 'finance', 'Alpaca-PY (stocks/forex)')
        self.check_module('yfinance', 'finance', 'yFinance (stock data)')
        print()
        
        # Dashboard & Visualization
        print("6️⃣  DASHBOARD & VISUALIZATION")
        print("-" * 70)
        self.check_module('streamlit', 'dashboard', 'Streamlit (web dashboard)')
        self.check_module('plotly', 'dashboard', 'Plotly (interactive charts)')
        self.check_module('altair', 'dashboard', 'Altair (viz grammar)')
        print()
        
        # Utilities
        print("7️⃣  UTILITIES & INFRASTRUCTURE")
        print("-" * 70)
        self.check_module('aiohttp', 'core', 'aiohttp (async HTTP)')
        self.check_module('apscheduler', 'core', 'APScheduler (scheduling)')
        self.check_module('pydantic', 'core', 'Pydantic (validation)')
        self.check_module('dotenv', 'core', 'python-dotenv (env vars)')
        print()
        
        # Sentiment Analysis (optional)
        print("8️⃣  SENTIMENT & NLP (OPTIONAL)")
        print("-" * 70)
        self.check_module('vaderSentiment', 'optional', 'VADER (sentiment)')
        self.check_module('praw', 'optional', 'PRAW (Reddit API)')
        self.check_module('finnhub', 'optional', 'Finnhub (news API)')
        print()
        
        # Testing
        print("9️⃣  TESTING FRAMEWORK")
        print("-" * 70)
        self.check_module('pytest', 'core', 'Pytest (unit testing)')
        print()
        
        # Project structure
        print("🔟  PROJECT STRUCTURE")
        print("-" * 70)
        self.check_project_structure()
        print()
    
    def check_project_structure(self):
        """Check if project directories exist."""
        required_dirs = [
            'core', 'data_feeds', 'intelligence', 'microstructure',
            'macro', 'aiml', 'quant', 'strategies', 'risk',
            'execution', 'backtesting', 'reporting', 'dashboard',
            'data', 'models', 'tests'
        ]
        
        base_path = Path(__file__).parent
        
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/")
                self.passed += 1
            else:
                print(f"❌ {dir_name}/ (missing)")
            self.total += 1
    
    def print_summary(self):
        """Print validation summary."""
        print("=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        
        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        
        print(f"\n✅ Passed:  {self.passed}/{self.total}")
        print(f"❌ Failed:  {self.total - self.passed}/{self.total}")
        print(f"📈 Success Rate: {success_rate:.1f}%\n")
        
        if success_rate == 100:
            print("🎉 ALL CHECKS PASSED! Environment is ready to go.")
            return 0
        elif success_rate >= 80:
            print("⚠️  Most checks passed. Some optional packages are missing.")
            print("   Optional packages are not required for basic operation.")
            return 0
        else:
            print("❌ Several required packages are missing.")
            print("   Run: pip install -r requirements.txt")
            return 1
    
    def suggestions(self):
        """Print improvement suggestions."""
        print("\n" + "=" * 70)
        print("💡 SUGGESTIONS")
        print("=" * 70 + "\n")
        
        failed_optional = [name for name, passed in self.results['optional'] if not passed]
        
        if failed_optional:
            print("Optional packages not installed (can install later):")
            for pkg in failed_optional:
                print(f"  • {pkg}")
            print("\nTo install optional packages:")
            print("  pip install tensorflow transformers optuna\n")
        
        print("Next steps:")
        print("  1. Run setup wizard: python run.py")
        print("  2. Start trading agent: python main.py")
        print("  3. Launch dashboard: streamlit run dashboard/app.py")
        print("  4. Test backtest: python -m backtesting.backtest_engine")
        print()


def main():
    """Run environment validation."""
    validator = EnvironmentValidator()
    
    try:
        validator.validate_all()
        exit_code = validator.print_summary()
        validator.suggestions()
        return exit_code
    
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
