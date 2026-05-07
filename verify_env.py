#!/usr/bin/env python3
"""Diagnostic script to verify Python environment and core dependencies."""

import sys
from importlib import import_module

PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'scikit-learn': 'sklearn',
    'xgboost': 'xgboost',
    'tensorflow': 'tensorflow',
    'alpaca-py': 'alpaca',
    'yfinance': 'yfinance',
    'streamlit': 'streamlit',
    'quantlib': 'quantlib',
    'sqlalchemy': 'sqlalchemy',
    'python-dotenv': 'dotenv',
}

print("=" * 80)
print("PYTHON ENVIRONMENT DIAGNOSTIC")
print("=" * 80)
print(f"Python Version: {sys.version}\n")

successful = 0
failed = 0
failed_packages = []

for display_name, module_name in PACKAGES.items():
    try:
        mod = import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {display_name:20} v{version}")
        successful += 1
    except ImportError as e:
        print(f"❌ {display_name:20} NOT INSTALLED")
        failed += 1
        failed_packages.append(display_name)
    except Exception as e:
        print(f"⚠️  {display_name:20} ERROR: {str(e)[:40]}")
        failed += 1
        failed_packages.append(display_name)

print("\n" + "=" * 80)
print(f"SUMMARY: {successful} successful | {failed} failed")
if failed_packages:
    print(f"Failed packages: {', '.join(failed_packages)}")
print("=" * 80)
sys.exit(0 if failed == 0 else 1)
