from trading_state import TradingStateManager

# Get all trades
trades = TradingStateManager.get_recent_trades(limit=100)

# Calculate total closed P&L
total_closed_pnl = sum(t.pnl for t in trades if t.status == 'CLOSED')
total_open_pnl = sum(t.pnl for t in trades if t.status == 'OPEN')

# Update portfolio
state = TradingStateManager.load_state()
state['portfolio']['closed_pnl'] = total_closed_pnl
state['portfolio']['open_pnl'] = total_open_pnl
state['portfolio']['current_equity'] = 100000.0 + total_closed_pnl + total_open_pnl
TradingStateManager.save_state(state)

portfolio = TradingStateManager.get_portfolio()
print('Portfolio Updated:')
print('  Initial Capital: ${:.2f}'.format(portfolio['initial_capital']))
print('  Current Equity: ${:.2f}'.format(portfolio['current_equity']))
print('  Closed P&L: ${:.2f}'.format(portfolio['closed_pnl']))
print('  Open P&L: ${:.2f}'.format(portfolio['open_pnl']))
print('  Available Cash: ${:.2f}'.format(portfolio['available_cash']))
print('  Total Trades: {}'.format(len(trades)))
