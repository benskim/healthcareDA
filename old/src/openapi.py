import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

# -------------------------------------------------------------
# 1. API 및 인증 정보 설정
# -------------------------------------------------------------
ENCODING_SERVICE_KEY = "AgI588lXohE0%2F1pICR7K1niL2Lpzoa4LAIuKGw2dOK9BaIyh0yBnarrbmeJA84JjnuFKWrmxBgXDEwd8LNmvCw%3D%3D"
BASE_URL = "https://apis.data.go.kr/B551182/msupUserInfoService1.2"

OUTPUT_DIR = "api_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------------
# 2. XML 응답 수집 및 파싱 함수
# -------------------------------------------------------------
def fetch_xml_api(endpoint_path, query_params):
    """
    요청 변수(query_params)를 반영하여 API를 호출하고 XML을 파싱합니다.
    """
    # 1. 기본 파라미터 구성 (serviceKey는 이중 인코딩 방지를 위해 직접 결합)
    query_string = urllib.parse.urlencode(query_params)
    url = f"{BASE_URL}{endpoint_path}?serviceKey={ENCODING_SERVICE_KEY}&{query_string}"

    req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.getcode() != 200:
                return None, 0

            xml_data = response.read().decode("utf-8")
            root = ET.fromstring(xml_data)

            # 결과 코드 확인
            result_code = root.find(".//resultCode")
            if result_code is not None and result_code.text != "00":
                result_msg = root.find(".//resultMsg")
                msg = result_msg.text if result_msg is not None else "Unknown Error"
                print(f"  └─ API 응답 에러 [{result_code.text}]: {msg}")
                return None, 0

            # <item> 항목 추출
            items = root.findall(".//item")
            page_list = []
            for item in items:
                item_dict = {child.tag: child.text for child in item}
                page_list.append(item_dict)

            # 전체 건수(totalCount) 추출
            total_count_elem = root.find(".//totalCount")
            total_count = (
                int(total_count_elem.text)
                if total_count_elem is not None and total_count_elem.text
                else 0
            )

            return page_list, total_count

    except Exception as e:
        print(f"  └─ 요청 중 에러 발생: {e}")
        return None, 0


# -------------------------------------------------------------
# 3. 데이터 수집 파이프라인 함수
# -------------------------------------------------------------
def collect_data(endpoint_path, custom_params, save_filename):
    print(f"\n[작업 시작] {endpoint_path} 수집 진행 중...")

    all_data = []
    page_no = 1
    num_of_rows = 1000

    while True:
        # 요청 변수 조합 (기본 페이지 변수 + 사용자 지정 변수)
        params = {
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            **custom_params,  # diagYm, insupTp, cpmdPrscTp 등 반영
        }

        print(f"  └─ Page {page_no} 수집 중... (조건: {custom_params})")
        items, total_count = fetch_xml_api(endpoint_path, params)

        if not items:
            break

        all_data.extend(items)

        # 전체 건수 도달 시 종료
        if len(all_data) >= total_count or len(items) < num_of_rows:
            break

        page_no += 1

    if all_data:
        df = pd.DataFrame(all_data)
        output_path = os.path.join(OUTPUT_DIR, save_filename)
        df.to_parquet(output_path, index=False)
        print(f"  └─ [성공] 총 {len(df):,}건 저장 완료 -> '{output_path}'")
    else:
        print("  └─ [경고] 수집된 데이터가 없습니다. (요청 변수/조회 조건을 확인하세요)")


# -------------------------------------------------------------
# 4. 실행부 (요청 변수 적용 예시)
# -------------------------------------------------------------
if __name__ == "__main__":

    # 1) 4단계ATC별상병별사용량목록조회 (/getAtcStp4SickList1.2)
    atc_params = {
        "diagYm": "202312",  # 진료년월 (YYYYMM)
        "insupTp": "0",  # 0:전체, 4:건강보험, 5:의료급여, 7:보훈
        "cpmdPrscTp": "01",  # 01:조제기준, 02:처방기준
        # "atcStep4Cd": "A10BA"  # 특정 ATC 코드 지정 시 주석 해제
    }
    collect_data(
        "/getAtcStp4SickList1.2", atc_params, "atc_stp4_sick_list.parquet"
    )

    # 2) 성분별상병별사용량목록조회 (/getCmpnSickList1.2)
    cmpn_params = {
        "diagYm": "202312",  # 진료년월 (YYYYMM)
        "insupTp": "0",  # 보험자구분
        "cpmdPrscTp": "01",  # 조제처방구분
        # "gnlNmCd": "100101ATB" # 특정 성분코드 지정 시 주석 해제
    }
    collect_data("/getCmpnSickList1.2", cmpn_params, "cmpn_sick_list.parquet")