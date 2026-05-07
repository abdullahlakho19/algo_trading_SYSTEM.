# ============================================================================
# EXCEL_EXPORTER.PY - Professional Excel Trade Report Generator
# Uses openpyxl for formatted, institutional-grade Excel output
# ============================================================================

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.styles import (
    Alignment,
    Border,
    Font,
    PatternFill,
    Side,
)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record for Excel export."""
    trade_id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    duration_minutes: float = 0.0
    strategy: str = ""
    reason: str = ""  # Why trade was taken
    exit_reason: str = ""  # How trade was exited
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED


class ExcelExporter:
    """
    Professional Excel trade report generator.
    
    Features:
    - Formatted worksheets with headers and styling
    - Trade-by-trade detailed records
    - Summary statistics sheet
    - Portfolio metrics
    - P&L breakdown by strategy
    - Professional styling (colors, fonts, borders)
    - Charts and pivot tables ready
    
    Designed for:
    - Post-trade analysis
    - Regulatory compliance reporting
    - Performance review meetings
    - Systematic record-keeping
    """
    
    # Color palette
    HEADER_COLOR = "1F4E78"  # Dark blue
    SUMMARY_COLOR = "D9E1F2"  # Light blue
    POSITIVE_COLOR = "C6EFCE"  # Light green
    NEGATIVE_COLOR = "FFC7CE"  # Light red
    NEUTRAL_COLOR = "FFF2CC"  # Light yellow
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize Excel exporter.
        
        Args:
            output_dir: Directory for report output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"✓ ExcelExporter initialized: output_dir={output_dir}")
    
    def _get_style_borders(self) -> Border:
        """Get standard borders for cells."""
        thin_border = Side(style='thin', color='000000')
        return Border(
            left=thin_border,
            right=thin_border,
            top=thin_border,
            bottom=thin_border
        )
    
    def _apply_header_style(self, cell) -> None:
        """Apply header styling to cell."""
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.fill = PatternFill(start_color=self.HEADER_COLOR, end_color=self.HEADER_COLOR, fill_type="solid")
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = self._get_style_borders()
    
    def _apply_data_style(self, cell, value_type: str = "text") -> None:
        """Apply data styling to cell."""
        cell.border = self._get_style_borders()
        cell.alignment = Alignment(horizontal='left', vertical='center' if value_type != "number" else 'right')
        
        if value_type == "number":
            cell.alignment = Alignment(horizontal='right', vertical='center')
        elif value_type == "currency":
            cell.alignment = Alignment(horizontal='right', vertical='center')
            cell.number_format = '$#,##0.00'
        elif value_type == "percent":
            cell.alignment = Alignment(horizontal='right', vertical='center')
            cell.number_format = '0.00%'
        elif value_type == "date":
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.number_format = 'yyyy-mm-dd hh:mm:ss'
    
    def _color_pnl_cell(self, cell, value: float) -> None:
        """Color P&L cell based on value."""
        if value > 0.01:
            cell.fill = PatternFill(start_color=self.POSITIVE_COLOR, end_color=self.POSITIVE_COLOR, fill_type="solid")
            cell.font = Font(color="006100", bold=True)
        elif value < -0.01:
            cell.fill = PatternFill(start_color=self.NEGATIVE_COLOR, end_color=self.NEGATIVE_COLOR, fill_type="solid")
            cell.font = Font(color="9C0006", bold=True)
        else:
            cell.fill = PatternFill(start_color=self.NEUTRAL_COLOR, end_color=self.NEUTRAL_COLOR, fill_type="solid")
    
    def _add_trades_sheet(self, wb: Workbook, trades: List[TradeRecord]) -> Worksheet:
        """
        Add trades detail sheet to workbook.
        
        Args:
            wb: Workbook
            trades: List of trade records
            
        Returns:
            Worksheet
        """
        ws = wb.create_sheet("Trades", 0)
        
        # Define columns
        headers = [
            "Trade ID", "Symbol", "Direction", "Entry Time", "Entry Price",
            "Exit Time", "Exit Price", "Size", "P&L ($)", "P&L (%)",
            "Duration (min)", "Strategy", "Reason", "Exit Reason", "Status"
        ]
        
        # Add headers
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            self._apply_header_style(cell)
        
        # Add trade data
        for row_idx, trade in enumerate(trades, 2):
            ws.cell(row=row_idx, column=1, value=trade.trade_id)
            ws.cell(row=row_idx, column=2, value=trade.symbol)
            ws.cell(row=row_idx, column=3, value=trade.direction)
            
            entry_time_cell = ws.cell(row=row_idx, column=4, value=trade.entry_time)
            self._apply_data_style(entry_time_cell, "date")
            
            entry_price_cell = ws.cell(row=row_idx, column=5, value=trade.entry_price)
            self._apply_data_style(entry_price_cell, "currency")
            
            if trade.exit_time:
                exit_time_cell = ws.cell(row=row_idx, column=6, value=trade.exit_time)
                self._apply_data_style(exit_time_cell, "date")
            
            exit_price_cell = ws.cell(row=row_idx, column=7, value=trade.exit_price)
            self._apply_data_style(exit_price_cell, "currency")
            
            size_cell = ws.cell(row=row_idx, column=8, value=trade.size)
            self._apply_data_style(size_cell, "number")
            
            pnl_cell = ws.cell(row=row_idx, column=9, value=trade.pnl)
            self._apply_data_style(pnl_cell, "currency")
            self._color_pnl_cell(pnl_cell, trade.pnl)
            
            pnl_pct_cell = ws.cell(row=row_idx, column=10, value=trade.pnl_percent)
            self._apply_data_style(pnl_pct_cell, "percent")
            self._color_pnl_cell(pnl_pct_cell, trade.pnl_percent)
            
            duration_cell = ws.cell(row=row_idx, column=11, value=trade.duration_minutes)
            self._apply_data_style(duration_cell, "number")
            
            ws.cell(row=row_idx, column=12, value=trade.strategy)
            ws.cell(row=row_idx, column=13, value=trade.reason)
            ws.cell(row=row_idx, column=14, value=trade.exit_reason)
            ws.cell(row=row_idx, column=15, value=trade.status)
            
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                self._apply_data_style(cell, "text")
        
        # Auto-adjust column widths
        widths = [12, 10, 12, 20, 14, 20, 14, 10, 12, 12, 15, 15, 20, 20, 12]
        for idx, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(idx)].width = width
        
        # Freeze header row
        ws.freeze_panes = "A2"
        
        logger.debug(f"✓ Added trades sheet with {len(trades)} records")
        
        return ws
    
    def _add_summary_sheet(self, wb: Workbook, trades: List[TradeRecord]) -> Worksheet:
        """
        Add summary statistics sheet.
        
        Args:
            wb: Workbook
            trades: List of trades
            
        Returns:
            Worksheet
        """
        ws = wb.create_sheet("Summary", 1)
        
        # Calculate statistics
        closed_trades = [t for t in trades if t.status == "CLOSED"]
        open_trades = [t for t in trades if t.status == "OPEN"]
        
        if closed_trades:
            wins = [t for t in closed_trades if t.pnl > 0]
            losses = [t for t in closed_trades if t.pnl <= 0]
            
            win_rate = len(wins) / len(closed_trades) if closed_trades else 0
            total_pnl = sum(t.pnl for t in closed_trades)
            avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
            profit_factor = sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 0
        else:
            win_rate = total_pnl = avg_win = avg_loss = profit_factor = 0
        
        open_pnl = sum(t.pnl for t in open_trades)
        
        # Summary data
        summary_data = [
            ("Total Trades", len(closed_trades) + len(open_trades)),
            ("Closed Trades", len(closed_trades)),
            ("Open Trades", len(open_trades)),
            ("", ""),
            ("Win Rate", win_rate),
            ("Total P&L", total_pnl),
            ("Open P&L", open_pnl),
            ("Average Win", avg_win),
            ("Average Loss", avg_loss),
            ("Profit Factor", profit_factor),
            ("", ""),
            ("Winning Trades", len(wins) if closed_trades else 0),
            ("Losing Trades", len(losses) if closed_trades else 0),
            ("Breakeven Trades", len([t for t in closed_trades if t.pnl == 0]) if closed_trades else 0),
        ]
        
        # Add summary content
        for row_idx, (label, value) in enumerate(summary_data, 2):
            label_cell = ws.cell(row=row_idx, column=1, value=label)
            label_cell.font = Font(bold=True, size=11)
            label_cell.border = self._get_style_borders()
            
            value_cell = ws.cell(row=row_idx, column=2, value=value)
            value_cell.border = self._get_style_borders()
            
            if isinstance(value, float):
                if "%w" in label or "Rate" in label:
                    self._apply_data_style(value_cell, "percent")
                elif "$" in label or "P&L" in label:
                    self._apply_data_style(value_cell, "currency")
                    if value > 0:
                        value_cell.fill = PatternFill(start_color=self.POSITIVE_COLOR, end_color=self.POSITIVE_COLOR, fill_type="solid")
                    elif value < 0:
                        value_cell.fill = PatternFill(start_color=self.NEGATIVE_COLOR, end_color=self.NEGATIVE_COLOR, fill_type="solid")
                else:
                    self._apply_data_style(value_cell, "number")
            else:
                self._apply_data_style(value_cell, "text")
        
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 20
        
        logger.debug("✓ Added summary sheet")
        
        return ws
    
    def _add_strategy_breakdown_sheet(self, wb: Workbook, trades: List[TradeRecord]) -> Worksheet:
        """
        Add strategy performance breakdown sheet.
        
        Args:
            wb: Workbook
            trades: List of trades
            
        Returns:
            Worksheet
        """
        ws = wb.create_sheet("Strategy Breakdown", 2)
        
        # Group by strategy
        strategies = {}
        for trade in trades:
            if trade.strategy not in strategies:
                strategies[trade.strategy] = []
            strategies[trade.strategy].append(trade)
        
        # Headers
        headers = ["Strategy", "Trades", "Wins", "Losses", "Win Rate", "Total P&L", "Avg P&L", "Best", "Worst"]
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            self._apply_header_style(cell)
        
        # Strategy data
        row_idx = 2
        for strategy, strategy_trades in strategies.items():
            closed = [t for t in strategy_trades if t.status == "CLOSED"]
            if not closed:
                continue
            
            wins = [t for t in closed if t.pnl > 0]
            losses = [t for t in closed if t.pnl <= 0]
            win_rate = len(wins) / len(closed) if closed else 0
            total_pnl = sum(t.pnl for t in closed)
            avg_pnl = total_pnl / len(closed) if closed else 0
            best_pnl = max(t.pnl for t in closed) if closed else 0
            worst_pnl = min(t.pnl for t in closed) if closed else 0
            
            ws.cell(row=row_idx, column=1, value=strategy or "UNSPECIFIED")
            ws.cell(row=row_idx, column=2, value=len(closed))
            ws.cell(row=row_idx, column=3, value=len(wins))
            ws.cell(row=row_idx, column=4, value=len(losses))
            
            wr_cell = ws.cell(row=row_idx, column=5, value=win_rate)
            self._apply_data_style(wr_cell, "percent")
            
            pnl_cell = ws.cell(row=row_idx, column=6, value=total_pnl)
            self._apply_data_style(pnl_cell, "currency")
            self._color_pnl_cell(pnl_cell, total_pnl)
            
            avg_cell = ws.cell(row=row_idx, column=7, value=avg_pnl)
            self._apply_data_style(avg_cell, "currency")
            
            best_cell = ws.cell(row=row_idx, column=8, value=best_pnl)
            self._apply_data_style(best_cell, "currency")
            
            worst_cell = ws.cell(row=row_idx, column=9, value=worst_pnl)
            self._apply_data_style(worst_cell, "currency")
            
            row_idx += 1
        
        for col_idx in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = 15
        
        logger.debug("✓ Added strategy breakdown sheet")
        
        return ws
    
    async def export_trades_to_excel(
        self,
        trades: List[TradeRecord],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export trades to Excel file.
        
        Args:
            trades: List of trade records
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to generated file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_report_{timestamp}.xlsx"
        
        filepath = self.output_dir / filename
        
        try:
            # Create workbook
            wb = Workbook()
            wb.remove(wb.active)  # Remove default sheet
            
            # Add sheets
            self._add_trades_sheet(wb, trades)
            self._add_summary_sheet(wb, trades)
            self._add_strategy_breakdown_sheet(wb, trades)
            
            # Save
            wb.save(filepath)
            
            logger.info(f"✓ Excel report exported: {filepath}")
            
            return filepath
        
        except Exception as e:
            logger.error(f"✗ Excel export failed: {e}")
            raise
    
    async def append_trade_to_export(
        self,
        filepath: Path,
        trade: TradeRecord,
    ) -> bool:
        """
        Append trade to existing Excel file.
        
        Args:
            filepath: Path to Excel file
            trade: Trade record to append
            
        Returns:
            True if successful
        """
        try:
            from openpyxl import load_workbook
            
            wb = load_workbook(filepath)
            ws = wb["Trades"]
            
            # Get next row
            next_row = ws.max_row + 1
            
            # Append trade
            ws.cell(row=next_row, column=1, value=trade.trade_id)
            ws.cell(row=next_row, column=2, value=trade.symbol)
            ws.cell(row=next_row, column=3, value=trade.direction)
            ws.cell(row=next_row, column=4, value=trade.entry_time)
            ws.cell(row=next_row, column=5, value=trade.entry_price)
            ws.cell(row=next_row, column=6, value=trade.exit_time)
            ws.cell(row=next_row, column=7, value=trade.exit_price)
            ws.cell(row=next_row, column=8, value=trade.size)
            ws.cell(row=next_row, column=9, value=trade.pnl)
            ws.cell(row=next_row, column=10, value=trade.pnl_percent)
            ws.cell(row=next_row, column=11, value=trade.duration_minutes)
            ws.cell(row=next_row, column=12, value=trade.strategy)
            ws.cell(row=next_row, column=13, value=trade.reason)
            ws.cell(row=next_row, column=14, value=trade.exit_reason)
            ws.cell(row=next_row, column=15, value=trade.status)
            
            wb.save(filepath)
            
            logger.debug(f"✓ Trade appended to {filepath}")
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Append trade failed: {e}")
            return False
