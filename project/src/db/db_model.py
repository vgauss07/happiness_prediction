from sqlalchemy import REAL, INTEGER
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from config import db_settings


class Base(DeclarativeBase):
    """Base class for SQLALchemy models"""
    pass  # noqa: WPS420, WPS604, WPS115


class WorldHappiness(Base):
    __tablename__ = db_settings.world_happiness_table_name

    Country_name: Mapped[int] = mapped_column(INTEGER(), primary_key=True)
    year: Mapped[int] = mapped_column(INTEGER())
    Life_Ladder: Mapped[float] = mapped_column(REAL())
    Log_GDP_per_capita: Mapped[float] = mapped_column(REAL())
    Social_support: Mapped[float] = mapped_column(REAL())
    Healthy_life_expectancy_at_birth: Mapped[float] = mapped_column(REAL())
    Freedom_to_make_life_choices: Mapped[float] = mapped_column(REAL())
    Generosity: Mapped[float] = mapped_column(REAL())
    Perceptions_of_corruption: Mapped[float] = mapped_column(REAL())
    Positive_affect: Mapped[float] = mapped_column(REAL())
    Negative_affect: Mapped[float] = mapped_column(REAL())
