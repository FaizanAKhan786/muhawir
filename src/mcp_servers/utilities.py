from datetime import date, datetime
import logging
from zoneinfo import ZoneInfo
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    class FastMCP:  # type: ignore
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(fn):
                return fn
            return decorator

        def run(self, transport="stdio"):
            pass


from src.agents.tools import calculate as safe_calculate

logger = logging.getLogger(__name__)

mcp = FastMCP("MuhawirUtilities")


def _gregorian_to_jd(year: int, month: int, day: int) -> float:
    """Compute Julian Day number from Gregorian date."""
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + (a // 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    return jd


def _jd_to_gregorian(jd: float) -> date:
    """Compute Gregorian date from Julian Day number."""
    z = int(jd + 0.5)
    f = (jd + 0.5) - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4)
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return date(int(year), int(month), int(day))


def _hijri_to_jd(year: int, month: int, day: int) -> float:
    """Compute Julian Day number from Tabular Islamic Calendar."""
    return (
        day
        + int(29.5001 * (month - 1) + 0.99)
        + (year - 1) * 354
        + int((3 + (11 * year)) / 30)
        + 1948438.5
        - 1
    )


def _jd_to_hijri(jd: float) -> tuple[int, int, int]:
    """Compute Tabular Islamic Calendar date (year, month, day) from Julian Day."""
    jd = int(jd) + 0.5
    days = jd - 1948439.5
    cycles = int(days / 10631)
    rem_days = days % 10631

    cycle_year = int((rem_days + 1) * 30 / 10631)
    year = cycles * 30 + cycle_year + 1

    rem_in_year = rem_days - (int((cycle_year * 10631 + 14) / 30))
    month = int((rem_in_year + 28.5) / 29.5)
    if month == 0:
        month = 1
    if month > 12:
        month = 12

    # Approximate day
    month_start_day = int(29.5001 * (month - 1) + 0.99)
    day = int(rem_in_year - month_start_day) + 1
    if day < 1:
        day = 1
    if day > 30:
        day = 30

    return year, month, day


@mcp.tool()
def get_current_datetime(timezone_name: str = "Asia/Qatar") -> str:
    """Return current ISO datetime string in the specified timezone (default Asia/Qatar)."""
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        from datetime import timezone as dt_timezone, timedelta
        if "qatar" in timezone_name.lower():
            tz = dt_timezone(timedelta(hours=3))
        else:
            tz = dt_timezone.utc
    return datetime.now(tz).isoformat()


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely without code execution vulnerabilities."""
    return safe_calculate(expression)


@mcp.tool()
def convert_hijri_gregorian(date_str: str, direction: str = "hijri_to_gregorian") -> str:
    """Convert between Hijri (Islamic) and Gregorian calendar dates.

    Format expected: YYYY-MM-DD.
    Direction: 'hijri_to_gregorian' or 'gregorian_to_hijri'.
    """
    try:
        parts = [int(p) for p in date_str.strip().split("-")]
        if len(parts) != 3:
            return "Error: Date must be in YYYY-MM-DD format."
        year, month, day = parts[0], parts[1], parts[2]

        if direction == "hijri_to_gregorian":
            jd = _hijri_to_jd(year, month, day)
            g_date = _jd_to_gregorian(jd)
            return g_date.strftime("%Y-%m-%d")
        elif direction == "gregorian_to_hijri":
            jd = _gregorian_to_jd(year, month, day)
            h_year, h_month, h_day = _jd_to_hijri(jd)
            return f"{h_year:04d}-{h_month:02d}-{h_day:02d}"
        else:
            return "Error: direction must be either 'hijri_to_gregorian' or 'gregorian_to_hijri'."
    except Exception as e:
        logger.error("Date conversion error for %s: %s", date_str, e)
        return f"Error: Conversion failed ({str(e)})"


if __name__ == "__main__":
    mcp.run(transport="stdio")
