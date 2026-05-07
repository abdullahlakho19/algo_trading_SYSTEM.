#!/usr/bin/env python3
"""Quick validation of trading system dependencies."""

packages = {
    'numpy': 'import numpy',
    'pandas': 'import pandas',
    'scipy': 'import scipy',
    'scikit-learn': 'import sklearn',
    'xgboost': 'import xgboost',
    'lightgbm': 'import lightgbm',
    'torch': 'import torch',
    'Alpaca': 'from alpaca.trading.client import TradingClient',
    'yFinance': 'import yfinance',
    'Streamlit': 'import streamlit',
    'Plotly': 'import plotly',
    'APScheduler': 'import apscheduler',
    'Pydantic': 'import pydantic',
    'requests': 'import requests',
    'aiohttp': 'import aiohttp',
}

print("=" * 70)
print("🔍 INSTITUTIONAL TRADING AGENT - DEPENDENCY VERIFICATION")
print("=" * 70 + "\n")

passed = 0
failed = 0
failed_packages = []

for name, import_stmt in packages.items():
    try:
        exec(import_stmt)
        print(f"  ✅ {name:20} LOADED")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name:20} FAILED: {str(e)[:50]}")
        failed += 1
        failed_packages.append(name)

print("\n" + "=" * 70)
print(f"✅ Passed:  {passed}/{len(packages)}")
print(f"❌ Failed:  {failed}/{len(packages)}")

if failed == 0:
    print("\n🎉 SUCCESS! ALL CRITICAL PACKAGES WORKING!")
    print("\n✨ Your trading system is ready for deployment:")
    print("   1. Configure .env with API keys")
    print("   2. Run: python main.py")
    print("   3. Monitor: streamlit run dashboard/app.py")
else:
    print(f"\n⚠️  Failed packages: {', '.join(failed_packages)}")
    print("   Install with: pip install [package_name]")

print("=" * 70)
