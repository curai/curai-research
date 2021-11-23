from datetime import datetime

import pytz
from pytz import timezone


def get_datetime_string(date_format="%m_%d_%Y %H_%M_%S") -> str:
    '''
    We'll use PST (California time) for our project timestamps
    '''
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)
    return pstDateTime
