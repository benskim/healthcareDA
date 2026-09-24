import os
import duckdb


## encoding check
from pathlib import Path

file_path = Path("parquet_data/03_getMeftDivSickList_2020.csv")

raw = file_path.read_bytes()

for encoding in ["utf-8", "cp949", "euc-kr"]:
    try:
        raw.decode(encoding)
        print(f"가능한 인코딩: {encoding}")
    except UnicodeDecodeError:
        print(f"실패: {encoding}")

# 실패: utf-8
# 가능한 인코딩: cp949
# 가능한 인코딩: euc-kr


##
con = duckdb.connect()

def read_csv_sample(con, file_path, limit=4):
    return con.execute(
        f"""
        SELECT *
        FROM read_csv(
            '{file_path}',
            encoding='CP949',
            header=true,
            auto_detect=true
        )
        LIMIT {limit}
        """
    ).fetchdf()


df_sample = read_csv_sample(con, file_path)
print(df_sample)


# # 탐색할 Parquet 파일 경로 (실제 경로로 수정하여 사용)
# target_files = [
#     "parquet_data/getAtcStp3SickList_2020.parquet",
#     "parquet_data/getMeftDivSickList_2020.parquet",
# ]

# for file_path in target_files:
#     print("=" * 80)
#     print(f"📂 파일: {file_path}")
#     print("=" * 80)

#     if not os.path.exists(file_path):
#         print(f"❌ 파일 없음: {file_path}\n")
#         continue

#     # 전체 데이터 중 샘플 4개 행만 Pandas DataFrame으로 불러오기
#     df_sample = con.execute(f"SELECT * FROM '{file_path}' LIMIT 4").fetchdf()

#     # 컬럼 및 데이터 4개 바로 출력
#     print(df_sample.to_string(index=False))
#     print("\n")