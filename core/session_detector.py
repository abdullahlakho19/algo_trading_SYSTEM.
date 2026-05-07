# ============================================================================
# SESSION_DETECTOR.PY - Detects trading sessions (London, NY, Asia, Overlap)
# ============================================================================

from datetime import datetime, time, timedelta
from enum import Enum
from typing import Tuple

import pytz


class TradingSession(Enum):
    """Trading session types."""
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_ASIA_LONDON = "overlap_asia_london"
    OFFLINE = "offline"


class SessionDetector:
    """Detects current trading session from market hours."""
    
    # Times in UTC
    SESSIONS = {
        TradingSession.ASIA: {
            'start': time(0, 0),      # 00:00 UTC = 08:00 Tokyo
            'end': time(8, 0),        # 08:00 UTC = 16:00 Tokyo
            'timezone': 'Asia/Tokyo',
        },
        TradingSession.LONDON: {
            'start': time(7, 0),      # 07:00 UTC = 08:00 London
            'end': time(15, 0),       # 15:00 UTC = 16:00 London (winter) / 15:00 UTC (summer)
            'timezone': 'Europe/London',
        },
        TradingSession.NEW_YORK: {
            'start': time(13, 0),     # 13:00 UTC = 09:00 EST
            'end': time(21, 0),       # 21:00 UTC = 17:00 EST
            'timezone': 'America/New_York',
        },
        TradingSession.OVERLAP_LONDON_NY: {
            'start': time(13, 0),     # 13:00 UTC (09:00 EST, 13:00 London)
            'end': time(15, 0),       # 15:00 UTC (10:00 EST, 15:00 London close)
        },
        TradingSession.OVERLAP_ASIA_LONDON: {
            'start': time(7, 0),      # 07:00 UTC (15:00 Tokyo, 07:00 London)
            'end': time(8, 0),        # 08:00 UTC (16:00 Tokyo, 08:00 London)
        }
    }
    
    @classmethod
    def get_current_session(cls, dt: datetime = None) -> TradingSession:
        """
        Get current trading session.
        
        Args:
            dt: datetime to check (default: now)
        
        Returns:
            Current TradingSession
        """
        if dt is None:
            dt = datetime.now(pytz.UTC)
        elif dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        else:
            dt = dt.astimezone(pytz.UTC)
        
        current_time = dt.time()
        weekday = dt.weekday()
        
        # Market closed on weekends
        if weekday >= 5:
            return TradingSession.OFFLINE
        
        # Check overlaps first (they have priority)
        if cls._is_in_session(current_time, TradingSession.OVERLAP_LONDON_NY):
            return TradingSession.OVERLAP_LONDON_NY
        
        if cls._is_in_session(current_time, TradingSession.OVERLAP_ASIA_LONDON):
            return TradingSession.OVERLAP_ASIA_LONDON
        
        # Individual sessions
        for session in [TradingSession.ASIA, TradingSession.LONDON, TradingSession.NEW_YORK]:
            if cls._is_in_session(current_time, session):
                return session
        
        return TradingSession.OFFLINE
    
    @classmethod
    def _is_in_session(cls, current_time: time, session: TradingSession) -> bool:
        """Check if time falls within session hours."""
        session_info = cls.SESSIONS[session]
        start = session_info['start']
        end = session_info['end']
        
        if start < end:
            return start <= current_time < end
        else:
            return current_time >= start or current_time < end
    
    @classmethod
    def get_session_info(cls, session: TradingSession) -> dict:
        """Get session information."""
        info = cls.SESSIONS.get(session)
        if info:
            return {
                'name': session.value,
                'start_utc': info['start'],
                'end_utc': info['end'],
                'timezone': info.get('timezone'),
            }
        return {}
    
    @classmethod
    def get_all_active_sessions(cls, dt: datetime = None) -> list:
        """Get all currently active sessions."""
        active = []
        
        if dt is None:
            dt = datetime.now(pytz.UTC)
        elif dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        
        current_time = dt.time()
        
        for session in TradingSession:
            if session != TradingSession.OFFLINE:
                if cls._is_in_session(current_time, session):
                    active.append(session)
        
        return active
    
    @classmethod
    def get_session_start_end(cls, session: TradingSession, dt: datetime = None) -> Tuple[datetime, datetime]:
        """Get session start and end times."""
        if dt is None:
            dt = datetime.now(pytz.UTC)
        elif dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        
        session_info = cls.SESSIONS.get(session)
        if not session_info:
            return None, None
        
        start_time = session_info['start']
        end_time = session_info['end']
        
        # Build datetime objects
        start = dt.replace(hour=start_time.hour, minute=start_time.minute, second=0, microsecond=0)
        end = dt.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)
        
        # Adjust if end is next day (e.g., overnight sessions)
        if end <= start:
            end += timedelta(days=1)
        
        return start, end


def main():
    """Test session detection."""
    detector = SessionDetector()
    
    current = detector.get_current_session()
    print(f"Current session: {current.value}")
    
    active = detector.get_all_active_sessions()
    print(f"Active sessions: {[s.value for s in active]}")


if __name__ == "__main__":
    main()
