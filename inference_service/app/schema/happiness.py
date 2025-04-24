"""Schema for World Happiness"""

from pydantic import BaseModel


class Happiness(BaseModel):
    """
    Happiness Schema

    'gdp': Country's GDP,
    'support': Social Support,
    'healthy': How healthy are you,
    'freedom': Perception of freedom,
    'corruption': Perception of corruption

    """
    gdp: float | int
    support: float | int
    healthy: float | int
    freedom: float | int
    corruption: float | int
