from pathlib import Path
import duckdb

# DuckDB의 해당 CSV 파싱 경로와 이 파일의 문자 인코딩 조합이 문제라고 보는 것이 타당합니다.

# 이미 확인한 사실이:
# 필드 수 = 8
# delimiter = ^
# encoding = CP949
# 해당 행의 raw bytes 정상
# Python CP949 decode 정상

# 이므로 DuckDB CSV parser를 계속 디버깅할 실익이 별로 없습니다.

# 가장 좋은 해결책
# Python에서 CP949 CSV를 읽고 → Parquet으로 변환 → DuckDB는 Parquet만 읽게 하세요.

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

file_list = ['2020','2021','2022']
for filename in file_list :
    INPUT = Path(f"raw_data/06_getAtcStp3SickList_{filename}.csv")
    OUTPUT = Path(f"parquet/06_getAtcStp3SickList_{filename}.parquet")

    COLUMNS = [
        "diagYm",
        "meftDivNo",
        "meftDivNoNm",
        "st3SickSym",
        "st3SickSymNm",
        "insupTpCd",
        "msupUseAmt",
        "totUseQty",
    ]

    BATCH_SIZE = 100_000

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    writer = None

    buffers = [[] for _ in COLUMNS]
    total_rows = 0

    with open(INPUT, "r", encoding="cp949", newline="") as f:
        reader = csv.reader(f, delimiter="^")

        for line_no, row in enumerate(reader, start=1):

            if len(row) != 8:
                raise ValueError(
                    f"Invalid column count at line {line_no}: "
                    f"expected 8, got {len(row)}\n{row}"
                )

            for i, value in enumerate(row):
                buffers[i].append(value)

            total_rows += 1

            if len(buffers[0]) >= BATCH_SIZE:

                table = pa.table(
                    {
                        "atcStep3Cd": pa.array(
                            buffers[2], type=pa.string()
                        ),
                        "atcStep3CdNm": pa.array(
                            buffers[7], type=pa.string()
                        ),
                        "diagYm": pa.array(
                            buffers[0], type=pa.string()
                        ),
                        "insupTpCd": pa.array(
                            buffers[3], type=pa.string()
                        ),
                        "msupUseAmt": pa.array(
                            [int(x) for x in buffers[4]],
                            type=pa.int64(),
                        ),
                        "st3SickSym": pa.array(
                            buffers[6], type=pa.string()
                        ),
                        "st3SickSymNm": pa.array(
                            buffers[1], type=pa.string()
                        ),
                        "totUseQty": pa.array(
                            [int(x) for x in buffers[5]],
                            type=pa.int64(),
                        ),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        OUTPUT,
                        table.schema,
                        compression="zstd",
                    )

                writer.write_table(table)

                buffers = [[] for _ in COLUMNS]

                print(f"{total_rows:,} rows processed")


    # 마지막 batch
    if buffers[0]:

        table = pa.table(
            {
                "atcStep3Cd": pa.array(
                    buffers[2], type=pa.string()
                ),
                "atcStep3CdNm": pa.array(
                    buffers[7], type=pa.string()
                ),
                "diagYm": pa.array(
                    buffers[0], type=pa.string()
                ),
                "insupTpCd": pa.array(
                    buffers[3], type=pa.string()
                ),
                "msupUseAmt": pa.array(
                    [int(x) for x in buffers[4]],
                    type=pa.int64(),
                ),
                "st3SickSym": pa.array(
                    buffers[6], type=pa.string()
                ),
                "st3SickSymNm": pa.array(
                    buffers[1], type=pa.string()
                ),
                "totUseQty": pa.array(
                    [int(x) for x in buffers[5]],
                    type=pa.int64(),
                ),
            }
        )

        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT,
                table.schema,
                compression="zstd",
            )

        writer.write_table(table)


    if writer is not None:
        writer.close()

    print()
    print(f"Done: {OUTPUT}")
    print(f"Rows: {total_rows:,}")