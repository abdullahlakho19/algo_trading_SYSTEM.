from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from pathlib import Path

# Create a new workbook
wb = Workbook()
ws = wb.active
ws.title = "Trading Report"

# Define styles
header_fill = PatternFill(start_color="1f6feb", end_color="1f6feb", fill_type="solid")
header_font = Font(bold=True, color="ffffff")
border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)

# Add headers
headers = ["Trade ID", "Symbol", "Direction", "Entry Price", "Exit Price", "Size", "P&L", "P&L %", "Status"]
ws.append(headers)

# Format headers
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.border = border

# Add sample trading data
trades = [
    ["TRADE_001", "AAPL", "LONG", 150.25, 152.50, 100, 225.00, 1.50, "CLOSED"],
    ["TRADE_002", "MSFT", "LONG", 380.00, 382.15, 50, 107.50, 0.56, "CLOSED"],
    ["TRADE_003", "GOOGL", "SHORT", 140.50, 138.75, 75, 131.25, 1.24, "CLOSED"],
    ["TRADE_004", "EUR/USD", "LONG", 1.0950, 1.0965, 50000, 750.00, 1.41, "CLOSED"],
    ["TRADE_005", "TSLA", "SHORT", 245.00, 242.50, 50, 125.00, 1.02, "CLOSED"],
]

for trade in trades:
    ws.append(trade)

# Format data cells
for row in ws.iter_rows(min_row=2, max_row=len(trades)+1, min_col=1, max_col=9):
    for cell in row:
        cell.border = border
        if cell.column in [4, 5, 7, 8]:  # Price and P&L columns
            cell.number_format = '0.00'
        if cell.column == 6:  # Size column
            cell.number_format = '0.0'
        cell.alignment = Alignment(horizontal="right")

# Auto-fit columns
column_widths = [12, 10, 10, 14, 14, 10, 12, 10, 12]
for i, width in enumerate(column_widths, 1):
    ws.column_dimensions[chr(64 + i)].width = width

# Add summary section
summary_row = len(trades) + 3
ws[f"A{summary_row}"] = "SUMMARY"
ws[f"A{summary_row}"].font = Font(bold=True, size=12)

summary_row += 1
ws[f"A{summary_row}"] = "Total Trades:"
ws[f"B{summary_row}"] = len(trades)

summary_row += 1
ws[f"A{summary_row}"] = "Total P&L:"
ws[f"B{summary_row}"] = sum(trade[7] for trade in trades)
ws[f"B{summary_row}"].number_format = '0.00'

summary_row += 1
ws[f"A{summary_row}"] = "Win Rate:"
win_count = sum(1 for trade in trades if trade[7] > 0)
ws[f"B{summary_row}"] = f"{(win_count / len(trades)) * 100:.1f}%"

# Create reports directory if needed
Path("reports").mkdir(exist_ok=True)

# Save the Excel file
filepath = Path("reports") / "trading_report.xlsx"
wb.save(filepath)
print(f"Excel file created successfully!")
print(f"Location: {filepath.absolute()}")
print(f"File size: {filepath.stat().st_size} bytes")
