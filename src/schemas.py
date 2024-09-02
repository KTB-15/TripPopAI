from pydantic import BaseModel

class TravelInfoBase(BaseModel):
    sido: str
    visit_area_nm: str
    gungu: str
    visit_area_type_cd: float
    visit_area_id: int
    residence_time_min_mean: float
    rcmdtn_intention_mean: float
    revisit_yn_mean: float
    travel_companions_num_mean: float
    revisit_intention_mean: float

class TravelInfoCreate(TravelInfoBase):
    pass

class TravelInfo(TravelInfoBase):
    id: int

    class Config:
        orm_mode = True
