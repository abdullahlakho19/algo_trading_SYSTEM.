# ============================================================================
# EVENT_CALENDAR.PY - Economic events & scheduled releases (FOMC, NFP, CPI)
# ============================================================================
# Source: Hardcoded 2026 calendar for testing (yFinance does not provide economic calendar)

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional
import logging

import pytz



logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """Event impact level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EconomicEvent:
    """Economic event record."""
    name: str
    country: str
    release_date: datetime
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    impact: EventImpact = EventImpact.MEDIUM


class EventCalendar:
    """Economic event calendar."""
    
    # Major events
    MAJOR_EVENTS = {
        'FOMC': {'impact': EventImpact.HIGH, 'frequency': 'bi-weekly'},
        'NFP': {'impact': EventImpact.HIGH, 'frequency': 'monthly'},
        'CPI': {'impact': EventImpact.HIGH, 'frequency': 'monthly'},
        'PPI': {'impact': EventImpact.MEDIUM, 'frequency': 'monthly'},
        'ISM_MANUFACTURING': {'impact': EventImpact.MEDIUM, 'frequency': 'monthly'},
        'ISM_SERVICES': {'impact': EventImpact.MEDIUM, 'frequency': 'monthly'},
        'DURABLE_GOODS': {'impact': EventImpact.MEDIUM, 'frequency': 'monthly'},
        'JOBLESS_CLAIMS': {'impact': EventImpact.MEDIUM, 'frequency': 'weekly'},
        'RETAIL_SALES': {'impact': EventImpact.MEDIUM, 'frequency': 'monthly'},
        'GDPR': {'impact': EventImpact.HIGH, 'frequency': 'quarterly'},
    }
    
    def __init__(self):
        """Initialize calendar."""
        self.events: List[EconomicEvent] = []
        self._generate_2026_calendar()
    
    def _generate_2026_calendar(self) -> None:
        """Generate 2026 economic calendar."""
        # FOMC meetings (8 per year, roughly every 6 weeks)
        fomc_dates = [
            datetime(2026, 1, 28, 19, 0),
            datetime(2026, 3, 18, 19, 0),
            datetime(2026, 5, 6, 19, 0),
            datetime(2026, 6, 17, 19, 0),
            datetime(2026, 7, 29, 19, 0),
            datetime(2026, 9, 23, 19, 0),
            datetime(2026, 11, 4, 19, 0),
            datetime(2026, 12, 16, 19, 0),
        ]
        
        for date in fomc_dates:
            self.events.append(EconomicEvent(
                name='FOMC Decision',
                country='USA',
                release_date=pytz.UTC.localize(date),
                impact=EventImpact.HIGH,
            ))
        
        # NFP (First Friday of each month, 13:30 UTC = 8:30 EST)
        for month in range(1, 13):
            first_day = datetime(2026, month, 1)
            # Find first Friday
            days_to_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_friday)
            
            date = first_friday.replace(hour=13, minute=30)
            self.events.append(EconomicEvent(
                name='Non-Farm Payroll (NFP)',
                country='USA',
                release_date=pytz.UTC.localize(date),
                impact=EventImpact.HIGH,
            ))
    
    def get_events_for_date(self, date: datetime) -> List[EconomicEvent]:
        """Get events for a specific date."""
        if date.tzinfo is None:
            date = pytz.UTC.localize(date)
        
        date_only = date.date()
        return [e for e in self.events if e.release_date.date() == date_only]
    
    def get_upcoming_events(self, days: int = 7, impact_min: EventImpact = EventImpact.MEDIUM) -> List[EconomicEvent]:
        """Get upcoming events within N days."""
        now = datetime.now(pytz.UTC)
        future = now + timedelta(days=days)
        
        high_priority = [
            e for e in self.events
            if now <= e.release_date <= future
            and e.impact.value >= impact_min.value
        ]
        
        return sorted(high_priority, key=lambda x: x.release_date)
    
    def is_high_impact_event_soon(self, hours: int = 24) -> bool:
        """
        Check if high-impact event in next N hours.
        
        Source: Hardcoded calendar (yFinance fallback)
        
        Returns:
            True if high-impact event found within N hours
        """
        # Check hardcoded calendar for high-impact events
        high_impact = self.get_upcoming_events(days=hours/24, impact_min=EventImpact.HIGH)
        return len(high_impact) > 0
    
    def get_events_this_week(self) -> List[EconomicEvent]:
        """Get events for this week."""
        now = datetime.now(pytz.UTC)
        days_to_sunday = (6 - now.weekday()) % 7
        week_end = now + timedelta(days=days_to_sunday)
        
        return [e for e in self.events if now <= e.release_date <= week_end]


def main():
    """Test calendar."""
    cal = EventCalendar()
    
    upcoming = cal.get_upcoming_events(days=30)
    print(f"Upcoming HIGH-impact events (30 days): {len(upcoming)}")
    
    for event in upcoming[:5]:
        print(f"  • {event.name} ({event.country}) - {event.release_date}")


if __name__ == "__main__":
    main()
