from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal
from models import TravelInfo

app = FastAPI()

# 모델 불러오기
model_path = "/Users/takhaseon/Documents/KTB/project/TripPopAI/data/travel_recommend_model.pkl"
with open(model_path, 'rb') as f:
    model = joblib.load(f)

# 입력 데이터 구조 정의
class TravelData(BaseModel):
    GENDER: str
    AGE_GRP: int
    TRAVEL_STYL_1: Union[int, None] = None
    TRAVEL_STYL_2: Union[int, None] = None
    TRAVEL_STYL_4: Union[int, None] = None
    TRAVEL_STYL_5: Union[int, None] = None
    TRAVEL_STYL_6: Union[int, None] = None
    TRAVEL_STYL_7: Union[int, None] = None
    TRAVEL_STYL_8: Union[int, None] = None
    SIDO: str
    GUNGU: str


@app.post("/predict/")
async def predict_travel(data: TravelData):
    # 기본 값이 ''로 초기화된 데이터프레임 생성
    default_values = {
        'TRAVEL_ID': '',
        'TRAVEL_MISSION_PRIORITY': '',
        'GENDER': '',
        'AGE_GRP': '',
        'INCOME': '',
        'TRAVEL_STYL_1': '',
        'TRAVEL_STYL_2': '',
        'TRAVEL_STYL_3': '',
        'TRAVEL_STYL_4': 4,
        'TRAVEL_STYL_5': '',
        'TRAVEL_STYL_6': '',
        'TRAVEL_STYL_7': '',
        'TRAVEL_STYL_8': '',
        'TRAVEL_MOTIVE_1': '',
        'TRAVEL_NUM': '',
        'TRAVEL_COMPANIONS_NUM': '',
        'sido_gungu_list': ''
    }

    # 입력 받은 데이터로 기본 값을 업데이트
    input_data = data.dict()
    for key, value in input_data.items():
        if key=='GENDER':
            default_values[key] = '남' if input_data[key] == 'MALE' else '여'
        elif key=='SIDO' or key=='GUNGU':
            continue

    default_values['sido_gungu_list'] = f"[{data.SIDO}+{data.GUNGU}]"

    # 최종 데이터프레임 생성
    user_data = pd.DataFrame([default_values])

    db: Session = SessionLocal()
    info_query = db.query(TravelInfo).filter(data.SIDO == TravelInfo.sido, data.GUNGU == TravelInfo.gungu).all()
    db.close()

    # 데이터프레임으로 변환
    info = pd.DataFrame([{
        'SIDO': record.sido,
        'GUNGU': record.gungu,
        'VISIT_AREA_NM': record.visit_area_nm,
        'VISIT_AREA_TYPE_CD': record.visit_area_type_cd,
        'VISIT_AREA_ID': record.visit_area_id,
        'RESIDENCE_TIME_MIN_mean': record.residence_time_min_mean,
        'RCMDTN_INTENTION_mean': record.rcmdtn_intention_mean,
        'REVISIT_YN_mean': record.revisit_yn_mean,
        'TRAVEL_COMPANIONS_NUM_mean': record.travel_companions_num_mean,
        'REVISIT_INTENTION_mean': record.revisit_intention_mean
    } for record in info_query])

    final_df = pd.DataFrame(columns=['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'VISIT_AREA_TYPE_CD', 'VISIT_AREA_ID',
                                     'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP', 'INCOME',
                                     'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                                     'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                                     'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
                                     'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean', 'REVISIT_YN_mean',
                                     'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean'])  # 빈 데이터프레임에 내용 추가


    # 데이터 전처리 및 예측 과정
    temp = user_data['sido_gungu_list'][0].replace("[", "").replace("]", "").replace("'", "").replace(", ", ",")
    if temp == '세종시+':
        sido = '세종시'
        gungu = ''
    else:
        sido, gungu = temp.split('+')

    info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu)].copy()

    info_df.drop(['SIDO'], inplace=True, axis=1)
    info_df.reset_index(inplace=True, drop=True)
    user_data = user_data.drop(['sido_gungu_list'], axis=1)


    user_df = pd.DataFrame([user_data.iloc[0].to_list()] * len(info_df), columns = ['SIDO', 'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP', 'INCOME',
    'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
    'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'])
    df = pd.concat([user_df, info_df], axis=1)
    df = df[['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'VISIT_AREA_TYPE_CD', 'VISIT_AREA_ID',
             'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP', 'INCOME',
             'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
             'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
             'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
             'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean', 'REVISIT_YN_mean',
             'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']]  # 변수정렬
    df['VISIT_AREA_TYPE_CD'] = df['VISIT_AREA_TYPE_CD'].astype('string')
    final_df = pd.concat([final_df, df], axis=0)

    # 모델 예측
    y_pred = model.predict(final_df)
    y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    pred_df = pd.concat([final_df, y_pred], axis=1)
    pred_df.sort_values(by=['y_pred'], ascending=False, inplace=True)

    # 상위 10개 관광지 추출
    top_10_places = pred_df.iloc[0:10]['VISIT_AREA_NM'].tolist()
    # top_10_places = [int(place) for place in top_10_places if not pd.isna(place)]  # 정수로 변환하며 NaN 값 제거
    # 결과 반환
    return {"recommended_places": top_10_places}