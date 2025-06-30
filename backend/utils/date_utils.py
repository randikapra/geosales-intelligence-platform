"""
Date and time utility functions for SFA system.
Handles date formatting, timezone operations, and business logic calculations.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Union
import pytz
from dateutil import parser
from dateutil.relativedelta import relativedelta
import calendar

# Sri Lankan timezone
LK_TZ = pytz.timezone('Asia/Colombo')
UTC_TZ = pytz.UTC

class DateUtils:
    """Comprehensive date utility class for SFA operations."""
    
    @staticmethod
    def get_lk_now() -> datetime:
        """Get current time in Sri Lankan timezone."""
        return datetime.now(LK_TZ)
    
    @staticmethod
    def get_utc_now() -> datetime:
        """Get current UTC time."""
        return datetime.now(UTC_TZ)
    
    @staticmethod
    def to_lk_timezone(dt: datetime) -> datetime:
        """Convert datetime to Sri Lankan timezone."""
        if dt.tzinfo is None:
            dt = UTC_TZ.localize(dt)
        return dt.astimezone(LK_TZ)
    
    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        """Convert datetime to UTC."""
        if dt.tzinfo is None:
            dt = LK_TZ.localize(dt)
        return dt.astimezone(UTC_TZ)
    
    @staticmethod
    def parse_flexible_date(date_str: str) -> Optional[datetime]:
        """Parse various date string formats commonly found in datasets."""
        if not date_str or date_str.strip() == '':
            return None
        
        try:
            # Handle common SFA date formats
            formats = [
                '%Y-%m-%d %H:%M:%S.%f',  # GPS data format
                '%d/%m/%Y %H:%M:%S',     # Order date format
                '%Y-%m-%d %H:%M:%S',     # Standard format
                '%Y-%m-%d',              # Date only
                '%d/%m/%Y',              # DMY format
                '%m/%d/%Y',              # MDY format
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            # Use dateutil as fallback
            return parser.parse(date_str.strip())
            
        except Exception:
            return None
    
    @staticmethod
    def format_for_display(dt: datetime, format_type: str = 'datetime') -> str:
        """Format datetime for display in different contexts."""
        if not dt:
            return ''
        
        # Ensure we're working with LK timezone
        if dt.tzinfo is None:
            dt = LK_TZ.localize(dt)
        else:
            dt = dt.astimezone(LK_TZ)
        
        formats = {
            'datetime': '%Y-%m-%d %H:%M:%S',
            'date': '%Y-%m-%d',
            'time': '%H:%M:%S',
            'display': '%d %b %Y, %I:%M %p',
            'short': '%d/%m/%Y',
            'month_year': '%b %Y',
            'day_month': '%d %b'
        }
        
        return dt.strftime(formats.get(format_type, formats['datetime']))
    
    @staticmethod
    def get_business_hours_range(date: datetime = None) -> Tuple[datetime, datetime]:
        """Get business hours range for a given date (9 AM - 6 PM LK time)."""
        if date is None:
            date = DateUtils.get_lk_now().date()
        elif isinstance(date, datetime):
            date = date.date()
        
        start_time = LK_TZ.localize(datetime.combine(date, datetime.min.time().replace(hour=9)))
        end_time = LK_TZ.localize(datetime.combine(date, datetime.min.time().replace(hour=18)))
        
        return start_time, end_time
    
    @staticmethod
    def is_business_hours(dt: datetime) -> bool:
        """Check if datetime falls within business hours."""
        start_time, end_time = DateUtils.get_business_hours_range(dt.date())
        dt_lk = DateUtils.to_lk_timezone(dt)
        return start_time <= dt_lk <= end_time
    
    @staticmethod
    def get_date_range(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> List[datetime]:
        """Generate list of dates between start and end date."""
        if isinstance(start_date, str):
            start_date = DateUtils.parse_flexible_date(start_date)
        if isinstance(end_date, str):
            end_date = DateUtils.parse_flexible_date(end_date)
        
        if not start_date or not end_date:
            return []
        
        dates = []
        current = start_date.date()
        end = end_date.date()
        
        while current <= end:
            dates.append(LK_TZ.localize(datetime.combine(current, datetime.min.time())))
            current += timedelta(days=1)
        
        return dates
    
    @staticmethod
    def get_month_boundaries(year: int, month: int) -> Tuple[datetime, datetime]:
        """Get first and last day of a month."""
        first_day = LK_TZ.localize(datetime(year, month, 1))
        last_day_num = calendar.monthrange(year, month)[1]
        last_day = LK_TZ.localize(datetime(year, month, last_day_num, 23, 59, 59))
        
        return first_day, last_day
    
    @staticmethod
    def get_quarter_boundaries(year: int, quarter: int) -> Tuple[datetime, datetime]:
        """Get first and last day of a quarter."""
        if quarter not in [1, 2, 3, 4]:
            raise ValueError("Quarter must be 1, 2, 3, or 4")
        
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        
        start_date = LK_TZ.localize(datetime(year, start_month, 1))
        last_day_num = calendar.monthrange(year, end_month)[1]
        end_date = LK_TZ.localize(datetime(year, end_month, last_day_num, 23, 59, 59))
        
        return start_date, end_date
    
    @staticmethod
    def get_working_days_count(start_date: datetime, end_date: datetime, 
                              exclude_weekends: bool = True) -> int:
        """Count working days between two dates."""
        if start_date > end_date:
            return 0
        
        current = start_date.date()
        end = end_date.date()
        working_days = 0
        
        while current <= end:
            if exclude_weekends and current.weekday() >= 5:  # Saturday = 5, Sunday = 6
                pass
            else:
                working_days += 1
            current += timedelta(days=1)
        
        return working_days
    
    @staticmethod
    def calculate_age_in_days(start_date: datetime, end_date: datetime = None) -> int:
        """Calculate age in days between two dates."""
        if end_date is None:
            end_date = DateUtils.get_lk_now()
        
        return (end_date.date() - start_date.date()).days
    
    @staticmethod
    def get_relative_time_description(dt: datetime) -> str:
        """Get human-readable relative time description."""
        now = DateUtils.get_lk_now()
        dt_lk = DateUtils.to_lk_timezone(dt)
        
        diff = now - dt_lk
        
        if diff.days > 0:
            if diff.days == 1:
                return "1 day ago"
            elif diff.days < 7:
                return f"{diff.days} days ago"
            elif diff.days < 30:
                weeks = diff.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            elif diff.days < 365:
                months = diff.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
            else:
                years = diff.days // 365
                return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.seconds > 0:
            if diff.seconds < 60:
                return "Just now"
            elif diff.seconds < 3600:
                minutes = diff.seconds // 60
                return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                hours = diff.seconds // 3600
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            return "Just now"
    
    @staticmethod
    def get_tour_date_from_code(tour_code: str) -> Optional[datetime]:
        """Extract date from tour code format (e.g., TU202U265230016)."""
        try:
            # Assuming tour code contains date info
            # This is a placeholder - adjust based on actual tour code format
            if len(tour_code) >= 10:
                date_part = tour_code[2:8]  # Extract date portion
                if date_part.isdigit():
                    year = 2000 + int(date_part[:2])
                    month = int(date_part[2:4])
                    day = int(date_part[4:6])
                    return LK_TZ.localize(datetime(year, month, day))
        except Exception:
            pass
        return None
    
    @staticmethod
    def group_dates_by_period(dates: List[datetime], period: str = 'month') -> dict:
        """Group dates by specified period (day, week, month, quarter, year)."""
        grouped = {}
        
        for date in dates:
            if period == 'day':
                key = date.strftime('%Y-%m-%d')
            elif period == 'week':
                key = f"{date.year}-W{date.isocalendar()[1]:02d}"
            elif period == 'month':
                key = date.strftime('%Y-%m')
            elif period == 'quarter':
                quarter = (date.month - 1) // 3 + 1
                key = f"{date.year}-Q{quarter}"
            elif period == 'year':
                key = str(date.year)
            else:
                key = date.strftime('%Y-%m-%d')
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(date)
        
        return grouped
    
    @staticmethod
    def get_fiscal_year(date: datetime = None, fiscal_start_month: int = 4) -> int:
        """Get fiscal year for a given date (default: April start)."""
        if date is None:
            date = DateUtils.get_lk_now()
        
        if date.month >= fiscal_start_month:
            return date.year
        else:
            return date.year - 1
    
    @staticmethod
    def time_since_last_gps_update(last_update: datetime) -> dict:
        """Calculate time since last GPS update with detailed breakdown."""
        now = DateUtils.get_lk_now()
        last_update_lk = DateUtils.to_lk_timezone(last_update)
        
        diff = now - last_update_lk
        
        return {
            'total_seconds': int(diff.total_seconds()),
            'total_minutes': int(diff.total_seconds() / 60),
            'total_hours': int(diff.total_seconds() / 3600),
            'days': diff.days,
            'hours': diff.seconds // 3600,
            'minutes': (diff.seconds % 3600) // 60,
            'seconds': diff.seconds % 60,
            'is_recent': diff.total_seconds() < 300,  # Within 5 minutes
            'status': 'active' if diff.total_seconds() < 300 else 'inactive'
        }


# Convenience functions for common operations
def now_lk() -> datetime:
    """Quick access to current LK time."""
    return DateUtils.get_lk_now()

def parse_date(date_str: str) -> Optional[datetime]:
    """Quick date parsing."""
    return DateUtils.parse_flexible_date(date_str)

def format_date(dt: datetime, fmt: str = 'datetime') -> str:
    """Quick date formatting."""
    return DateUtils.format_for_display(dt, fmt)

def business_hours_today() -> Tuple[datetime, datetime]:
    """Get today's business hours."""
    return DateUtils.get_business_hours_range()

def is_working_time(dt: datetime = None) -> bool:
    """Check if current or given time is working hours."""
    if dt is None:
        dt = DateUtils.get_lk_now()
    return DateUtils.is_business_hours(dt)