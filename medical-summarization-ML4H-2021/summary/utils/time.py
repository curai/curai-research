"""Time Related Utilities"""
from datetime import datetime
from pytz import timezone

def get_est_time():
    tz = timezone('America/New_York')
    time = datetime.now(tz).strftime("%m.%d.%Y-%I:%M%p")
    return time
