import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 0. 환경
# ============================================================

DB_PATH = "healthcare.duckdb"
OUT_DIR = Path("presentation_output")
OUT_DIR.mkdir(exist_ok=True)

con = duckdb.connect(DB_PATH, read_only=True)

# 한글 폰트
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 공통 함수
# ============================================================

def save_csv(df, name):
    path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"[parquet] {path}")
    return path


def save_png(name):
    path = OUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PNG] {path}")
    return path


# ============================================================
# 1. 데이터 존재 여부 확인
# ============================================================

tables = con.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_name IN (
        'candidate_mefi',
        'candidate_atc4',
        'candidate_cmpn',
        'candidate_atc4_ids',
        'candidate_cmpn_ids'
    )
    ORDER BY table_name
""").df()

print("\n=== 발표용 candidate 테이블 ===")
print(tables)


# ============================================================
# 2. Slide 2
# MEFI 시장규모 Top 10
# ============================================================

mefi = con.execute("""
SELECT
    meftDivNo,
    meftDivNoNm,
    amt_2020,
    amt_2021,
    amt_2022,
    growth_amt,
    cumulative_growth_rate,
    cagr_2y,
    consistent_growth,
    largest_rank,
    rapid_rank,
    consistent_rank,
    market_position
FROM candidate_mefi
ORDER BY amt_2022 DESC
""").df()

save_csv(mefi, "slide02_mefi_all")


mefi_top10 = mefi.head(10).copy()

save_csv(mefi_top10, "slide02_mefi_top10")


# ============================================================
# 3. Slide 3
# 시장 규모 × CAGR
# ============================================================

save_csv(
    mefi[
        [
            "meftDivNo",
            "meftDivNoNm",
            "amt_2022",
            "growth_amt",
            "cagr_2y",
            "consistent_growth",
            "market_position"
        ]
    ],
    "slide03_mefi_scatter"
)


# ============================================================
# 4. Slide 4
# ATC4 Top 10
# ============================================================

atc4 = con.execute("""
SELECT
    atc4cd,
    atc4cdNm,
    amt_2020,
    amt_2021,
    amt_2022,
    growth_amt,
    cumulative_growth_rate,
    cagr_2y,
    consistent_growth,
    size_rank,
    growth_rank,
    persistence_rank,
    atc4_position
FROM candidate_atc4
ORDER BY amt_2022 DESC
""").df()

save_csv(atc4, "slide04_atc4_all")

atc4_top10 = atc4.head(10).copy()

save_csv(atc4_top10, "slide04_atc4_top10")


# ============================================================
# 5. Slide 5
# ATC4 × 질환
#
# 후보 ATC4 전체를 대상으로 계산한 뒤
# 발표에서는 상위 5개 ATC4를 사용
# ============================================================

disease = con.execute("""
WITH candidate AS (
    SELECT DISTINCT
        atc4cd
    FROM candidate_atc4
),

base AS (
    SELECT
        a.atcStep4Cd AS atc4cd,
        a.atcStep4CdNm AS atc4cdNm,
        a.st3SickSym AS disease_cd,
        a.st3SickSymNm AS disease_nm,
        SUM(
            CASE
                WHEN a.diagYm = '202212'
                THEN a.msupUseAmt
                ELSE 0
            END
        ) AS amt_2022
    FROM atc4_sick AS a
    INNER JOIN candidate AS c
        ON a.atcStep4Cd = c.atc4cd
    WHERE a.diagYm BETWEEN '202001' AND '202212'
      AND a.insupTpCd IN ('4','5','7')
    GROUP BY
        a.atcStep4Cd,
        a.atcStep4CdNm,
        a.st3SickSym,
        a.st3SickSymNm
),

ranked AS (
    SELECT
        *,
        SUM(amt_2022) OVER (
            PARTITION BY atc4cd
        ) AS atc4_total,
        RANK() OVER (
            PARTITION BY atc4cd
            ORDER BY amt_2022 DESC
        ) AS disease_rank
    FROM base
)

SELECT
    atc4cd,
    atc4cdNm,
    disease_cd,
    disease_nm,
    amt_2022,
    atc4_total,
    amt_2022 / NULLIF(atc4_total, 0) AS disease_share
FROM ranked
WHERE disease_rank <= 5
ORDER BY atc4_total DESC, atc4cd, disease_rank
""").df()

save_csv(disease, "slide05_atc4_disease")


# ============================================================
# 6. Slide 6-A
# 지역 Top 10
# ============================================================

region = con.execute("""
WITH candidate AS (
    SELECT DISTINCT
        atc4cd
    FROM candidate_atc4
),

region_market AS (
    SELECT
        a.atcStep4Cd AS atc4cd,
        a.atcStep4CdNm AS atc4cdNm,
        a.regionStep2CdNm AS region_nm,
        SUM(
            CASE
                WHEN a.diagYm = '202212'
                THEN a.msupUseAmt
                ELSE 0
            END
        ) AS amt_2022
    FROM atc4_region_inst AS a
    INNER JOIN candidate AS c
        ON a.atcStep4Cd = c.atc4cd
    WHERE a.diagYm BETWEEN '202001' AND '202212'
      AND a.insupTpCd IN ('4','5','7')
    GROUP BY
        a.atcStep4Cd,
        a.atcStep4CdNm,
        a.regionStep2CdNm
)

SELECT
    region_nm,
    SUM(amt_2022) AS amt_2022
FROM region_market
GROUP BY region_nm
ORDER BY amt_2022 DESC
LIMIT 10
""").df()

save_csv(region, "slide06_region_top10")


# ============================================================
# 7. Slide 6-B
# 의료기관 유형
# ============================================================

institution = con.execute("""
WITH candidate AS (
    SELECT DISTINCT
        atc4cd
    FROM candidate_atc4
)

SELECT
    medInstType,
    SUM(
        CASE
            WHEN diagYm = '202012'
            THEN msupUseAmt
            ELSE 0
        END
    ) AS amt_2020,
    SUM(
        CASE
            WHEN diagYm = '202212'
            THEN msupUseAmt
            ELSE 0
        END
    ) AS amt_2022
FROM atc4_region_inst AS a
INNER JOIN candidate AS c
    ON a.atcStep4Cd = c.atc4cd
WHERE a.diagYm BETWEEN '202001' AND '202212'
  AND a.insupTpCd IN ('4','5','7')
GROUP BY medInstType
ORDER BY amt_2022 DESC
""").df()

save_csv(institution, "slide06_institution")


# ============================================================
# 8. Slide 7
# Component Top 10
# ============================================================

cmpn = con.execute("""
SELECT
    gnlNmCd,
    gnlNmCdNm,
    amt_2020,
    amt_2021,
    amt_2022,
    growth_amt,
    cumulative_growth_rate,
    cagr_2y,
    consistent_growth,
    size_rank,
    growth_rank,
    persistence_rank,
    cmpn_position
FROM candidate_cmpn
ORDER BY amt_2022 DESC
""").df()

save_csv(cmpn, "slide07_cmpn_all")

cmpn_top10 = cmpn.head(10).copy()

save_csv(cmpn_top10, "slide07_cmpn_top10")


# ============================================================
# 9. Slide 8
# Component → Product → Company
# ============================================================

product_company = con.execute("""
WITH candidate_component AS (
    SELECT 
        gnlNmCd
    FROM candidate_cmpn
    GROUP BY 1
),

mapping AS (
    SELECT DISTINCT
        m.gnlNmCd,
        m.ProdCd,
        m.ProdNm,
        m.companyNm,
        m.meftDivNo,
        m.AtCcd,
        m.AtCcdNm,
        m.atc4cd,
        m.atc4cdNm
    FROM prod_atc_map_named AS m
    INNER JOIN candidate_component AS c
        ON m.gnlNmCd = c.gnlNmCd
    WHERE m.gnlNmCd IS NOT NULL
)

SELECT
    *
FROM mapping
ORDER BY
    gnlNmCd,
    companyNm,
    ProdNm
""").df()

save_csv(
    product_company,
    "slide08_component_product_company"
)


# ============================================================
# 10. 발표용 핵심 숫자
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "MEFI 후보 수",
        "ATC4 후보 수",
        "Component 후보 수",
        "MEFI 전체 행 수",
        "ATC4 전체 행 수",
        "Component 전체 행 수"
    ],
    "value": [
        len(mefi),
        len(atc4),
        len(cmpn),
        con.execute("SELECT COUNT(*) FROM mefi_sick").fetchone()[0],
        con.execute("SELECT COUNT(*) FROM atc4_sick").fetchone()[0],
        con.execute("SELECT COUNT(*) FROM cmpn_sick").fetchone()[0]
    ]
})

save_csv(summary, "presentation_summary")

print("\n=== 완료 ===")
print(summary)

con.close()