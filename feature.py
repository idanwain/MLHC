from typing import NamedTuple
from datetime import datetime


class Feature(NamedTuple):
    """feature"""
    time: datetime
    value: int
    uom: str
