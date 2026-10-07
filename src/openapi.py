import requests
import pandas as pd
import xml.etree.ElementTree as ET
import time

# 설정
SERVICE_KEY = "비밀이므로 생략하여 제공함"

# getAtcStp3ClList1.2 엔드포인트
# 반드시 공공데이터포털의 상세 API 명세에서 확인한 URL 사용
BASE_URL = "https://apis.data.go.kr/B551182/msupUserInfoService1.2/getMeftDivSickList1.2"

params = {
    "ServiceKey": SERVICE_KEY,
    "numOfRows": 10000, #1만행 지정해도 작동됨. 콜당 대부분 3천에서 5천행 사이다. 
    "pageNo": 1,
    # 진료년월
    "diagYm": "202305",
    # 보험자 구분 (0: 전체, 4: 건강보험, 5: 의료급여, 7: 보훈)
    "insupTp": "0",
    # 01: 조제기준, 02: 처방기준
    "cpmdPrscTp": "01",

    "meftDivNo" : "232",

    # ATC 3단계 (예: A10B)
    # "atcStep3Cd": "A10C",

    # # 지역
    # "sidoCd": "11",
    # "sgguCd": "110001",
    # # 요양기관종별
    # "clCd": "01",
}

response = requests.get(BASE_URL, params=params, timeout=60)

print("HTTP status:", response.status_code)
print(response.url)
print(response.text)