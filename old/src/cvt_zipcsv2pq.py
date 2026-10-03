from pathlib import Path
import zipfile
import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 설정
# ============================================================

RAW_DIR = Path("raw_data")
OUTPUT_DIR = Path("parquet")

CHUNK_SIZE = 200_000

# 지원할 인코딩
ENCODINGS = [
    "cp949",
    "euc-kr",
    "utf-8-sig",
    "utf-8",
]

# CSV 구분자 후보
DELIMITERS = [
    "^",
    ",",
    "\t",
    "|",
]


# ============================================================
# 인코딩 자동 감지
# ============================================================

def detect_encoding(raw_bytes):
    """
    몇 KB를 읽어서 가장 가능성 높은 인코딩을 찾는다.
    """

    sample = raw_bytes[:100_000]

    for encoding in ENCODINGS:

        try:
            sample.decode(encoding)
            return encoding

        except UnicodeDecodeError:
            continue

    # 최후의 fallback
    return "cp949"


# ============================================================
# 구분자 자동 감지
# ============================================================

def detect_delimiter(text):
    """
    첫 번째 데이터 라인을 기준으로 구분자를 추정한다.
    """

    lines = [
        line for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return "^"

    line = lines[0]

    counts = {
        delimiter: line.count(delimiter)
        for delimiter in DELIMITERS
    }

    delimiter = max(
        counts,
        key=counts.get
    )

    # 구분자가 발견되지 않은 경우
    if counts[delimiter] == 0:
        return ","

    return delimiter


# ============================================================
# CSV 파일 하나 → Parquet
# ============================================================

def convert_csv(
    zip_file,
    csv_name,
    output_path
):

    print()
    print("=" * 80)
    print(f"CSV : {csv_name}")
    print(f"OUT : {output_path}")
    print("=" * 80)

    # --------------------------------------------------------
    # CSV 앞부분을 읽어서 encoding / delimiter 확인
    # --------------------------------------------------------

    with zip_file.open(csv_name) as f:

        sample_bytes = f.read(100_000)

    encoding = detect_encoding(sample_bytes)

    sample_text = sample_bytes.decode(
        encoding,
        errors="replace"
    )

    delimiter = detect_delimiter(
        sample_text
    )

    print(f"encoding : {encoding}")
    print(f"delimiter: {repr(delimiter)}")

    # --------------------------------------------------------
    # 컬럼 수 확인
    # --------------------------------------------------------

    first_line = next(
        line
        for line in sample_text.splitlines()
        if line.strip()
    )

    column_count = len(
        first_line.split(delimiter)
    )

    print(f"columns  : {column_count}")

    # --------------------------------------------------------
    # Parquet 변환
    # --------------------------------------------------------

    writer = None
    total_rows = 0

    with zip_file.open(csv_name) as f:

        reader = pd.read_csv(
            f,

            # 자동 감지된 구분자
            sep=delimiter,

            # 자동 감지된 인코딩
            encoding=encoding,

            # 공공데이터 CSV의 경우
            # 헤더가 없는 경우가 많으므로 일단 없음
            header=None,

            # 원본 값 보존
            dtype=str,

            # 대용량 처리
            chunksize=CHUNK_SIZE,

            low_memory=False,

        )

        for chunk in reader:

            # ------------------------------------------------
            # 컬럼명 생성
            # ------------------------------------------------

            if chunk.shape[1] != column_count:

                raise ValueError(
                    f"컬럼 수 불일치: "
                    f"expected={column_count}, "
                    f"actual={chunk.shape[1]}"
                )

            chunk.columns = [
                f"col_{i}"
                for i in range(
                    1,
                    column_count + 1
                )
            ]

            # ------------------------------------------------
            # 문자열 양쪽 공백 제거
            # ------------------------------------------------

            for col in chunk.columns:

                chunk[col] = (
                    chunk[col]
                    .str.strip()
                )

            # ------------------------------------------------
            # Arrow Table
            # ------------------------------------------------

            table = pa.Table.from_pandas(
                chunk,
                preserve_index=False
            )

            # ------------------------------------------------
            # 첫 chunk에서 schema 생성
            # ------------------------------------------------

            if writer is None:

                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd"
                )

            writer.write_table(table)

            total_rows += len(chunk)

            print(
                f"rows: {total_rows:,}",
                end="\r"
            )

    if writer is not None:
        writer.close()

    print()
    print(
        f"✓ 완료: {total_rows:,} rows"
    )


# ============================================================
# ZIP 하나 처리
# ============================================================

def process_zip(zip_path):

    print()
    print("#" * 80)
    print(f"ZIP: {zip_path}")
    print("#" * 80)

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
            and not name.endswith("/")
        ]

        if not csv_files:

            print("CSV 파일 없음")
            return

        print(
            f"CSV {len(csv_files)}개 발견"
        )

        for csv_name in csv_files:

            filename = Path(
                csv_name
            ).name

            # ZIP별 별도 폴더
            zip_output_dir = (
                OUTPUT_DIR
                / zip_path.stem
            )

            zip_output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                zip_output_dir
                / f"{Path(filename).stem}.parquet"
            )

            convert_csv(
                z,
                csv_name,
                output_path
            )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_files = list(
        RAW_DIR.glob("*.zip")
    )

    if not zip_files:

        print(
            f"ZIP 파일이 없습니다: {RAW_DIR}"
        )

        return

    print(
        f"ZIP 파일 {len(zip_files)}개 발견"
    )

    for zip_path in zip_files:

        try:

            process_zip(
                zip_path
            )

        except Exception as e:

            print()
            print(
                f"❌ 오류: {zip_path}"
            )
            print(
                f"   {type(e).__name__}: {e}"
            )

    print()
    print("=" * 80)
    print("전체 작업 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()