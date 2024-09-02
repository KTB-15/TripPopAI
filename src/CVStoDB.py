import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import TravelInfo

# CSV 파일 읽기
file_path = '/Users/takhaseon/Documents/KTB/project/TripPopAI/data/Testset_travel_info.csv'
df = pd.read_csv(file_path)


# 데이터베이스 세션 생성
db = SessionLocal()

# 데이터베이스에 데이터 삽입
def insert_data_from_csv(db: Session, df: pd.DataFrame):
    # 데이터베이스의 모든 데이터 삭제
    db.query(TravelInfo).delete()
    db.commit()

    for index, row in df.iterrows():
        travel_info = TravelInfo(
            sido=row['SIDO'],
            visit_area_nm=row['VISIT_AREA_NM'],
            gungu=row['GUNGU'],
            visit_area_type_cd=row['VISIT_AREA_TYPE_CD'],
            visit_area_id=row['VISIT_AREA_ID'],
            residence_time_min_mean=row['RESIDENCE_TIME_MIN_mean'],
            rcmdtn_intention_mean=row['RCMDTN_INTENTION_mean'],
            revisit_yn_mean=row['REVISIT_YN_mean'],
            travel_companions_num_mean=row['TRAVEL_COMPANIONS_NUM_mean'],
            revisit_intention_mean=row['REVISIT_INTENTION_mean']
        )
        db.add(travel_info)
    db.commit()

# 데이터 삽입 실행
insert_data_from_csv(db, df)

# 세션 종료
db.close()
