import duckdb

con = duckdb.connect()

# 1. parquet_data 폴더 안의 12개 파일 전체 행 수 조회
total_rows = con.execute(
    "SELECT COUNT(*) FROM 'parquet_data/*.parquet'"
).fetchone()[0]
print(f"총 분석 데이터 행 수: {total_rows:,}개")

# 2. 데이터 구조(컬럼 정보) 확인
schema = con.execute("DESCRIBE SELECT * FROM 'parquet_data/*.parquet'").df()
print("\n--- 데이터 컬럼 구조 ---")
print(schema[["column_name", "column_type"]])

# 3. 예시: 원하는 조건으로 집계 분석
# (아래 column_1, column_2를 실제 파일의 컬럼명으로 수정하여 사용하세요)
query = """
    SELECT 
        COUNT(*) AS row_count
    FROM 'parquet_data/*.parquet'
"""

result = con.execute(query).df()
print("\n--- 집계 결과 ---")
print(result)