from sqlalchemy import Column, Integer, String, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TravelInfo(Base):
    __tablename__ = 'travel_info'

    id = Column(Integer, primary_key=True, index=True)
    sido = Column(String, nullable=False)
    visit_area_nm = Column(String, nullable=False)
    gungu = Column(String, nullable=False)
    visit_area_type_cd = Column(Float, nullable=False)
    visit_area_id = Column(BigInteger, nullable=False)
    residence_time_min_mean = Column(Float, nullable=False)
    rcmdtn_intention_mean = Column(Float, nullable=False)
    revisit_yn_mean = Column(Float, nullable=False)
    travel_companions_num_mean = Column(Float, nullable=False)
    revisit_intention_mean = Column(Float, nullable=False)
