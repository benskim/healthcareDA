#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1 (약효분류별 월별 금액·수량) 무결성 점검

사용법:
    python d1_integrity_check.py meft_2014-2025.csv
    python d1_integrity_check.py meft_2014-2025.csv --outdir d1_check_out
    python d1_integrity_check.py meft_2014-2025.csv --jump-thr 0.7 --min-qty 1000 --min-amt 100

필요: Python 3.8+, pandas, numpy
단위: 수량 = 천개, 금액 = 백만원 (고정).  단가(원/개) = 금액*1000/수량
출력: 콘솔 요약 + outdir 아래 CSV/TXT 파일 (한글 엑셀용 utf-8-sig)
종료 코드: FAIL 항목이 있으면 1, 없으면 0

가정: 첫 줄은 헤더(이름은 무시, 열 위치로 읽음), 열 순서는 코드·분류명·진료년월·연도·월·수량·금액 (7열).
v2 변경: 필수 열이 비면 크래시 대신 bad_rows.csv로 분리하고 계속 진행 / 수량만 NaN인 행 집계 /
         연도·월 재색인으로 YoY 왜곡 방지 / --min-qty는 한쪽 달만 만족해도 포함 /
         수량 0 & 금액>0 구간(qty_zero_amt_pos.csv) 별도 점검 / 파일·파싱 오류 메시지 정리
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

MON = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
COLS = ["code", "name", "ym_str", "year", "month", "qty", "amt"]  # 위치 기준 (헤더가 깨져도 동작)

RESULTS = []  # (status, check, detail)


def rec(status, check, detail=""):
    RESULTS.append((status, check, detail))
    print(f"[{status:<4}] {check}" + (f"  ->  {detail}" if detail else ""))


def load(path):
    # utf-8을 먼저 엄격하게 시도하고, 실패하면 cp949 (한글 Windows 기본)
    last = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as e:
            last = e
    raise last


def parse_ym(s):
    """'Jan-14' -> Timestamp(2014-01-01). 로케일 영향이 없도록 직접 변환."""
    try:
        m, y = str(s).strip().split("-")
        return pd.Timestamp(year=2000 + int(y), month=MON[m.title()[:3]], day=1)
    except Exception:
        return pd.NaT


def main():
    ap = argparse.ArgumentParser(description="D1 무결성 점검")
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="d1_check_out")
    ap.add_argument("--jump-thr", type=float, default=0.7, help="단가 월간 |Δln| 임계값 (기본 0.7)")
    ap.add_argument("--min-qty", type=float, default=0,
                    help="단가 급변 점검 대상: 수량(천개) 하한, 이전·당월 중 한쪽만 만족해도 포함 (기본 0 = 사용 안 함, 금액 기준만 권장)")
    ap.add_argument("--min-amt", type=float, default=100, help="단가 급변 점검 대상: 금액(백만원) 하한")
    a = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda f: os.path.join(a.outdir, f)

    # ---------------------------------------------------------------- 1. 로드
    print("=" * 70, "\n1. 로드·구조\n" + "=" * 70)
    RESULTS.clear()
    try:
        raw, enc = load(a.csv)
    except FileNotFoundError:
        ap.error(f"파일을 찾을 수 없습니다: {a.csv}")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        ap.error(f"CSV를 읽을 수 없습니다: {e}")
    rec("INFO", "인코딩", enc)
    rec("INFO", "행·열", f"{raw.shape[0]:,}행 x {raw.shape[1]}열")
    if raw.shape[1] != 7:
        rec("FAIL", "열 수 7개", f"{raw.shape[1]}개")
        return finish(a)
    rec("INFO", "헤더", " | ".join(map(str, raw.columns)))
    df = raw.copy()
    df.columns = COLS
    for c in ("code", "year", "month", "qty", "amt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ym"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1), errors="coerce")
    bad = df[["code", "year", "month"]].isna().any(axis=1) | df["ym"].isna()
    if bad.any():
        raw[bad.values].to_csv(out("bad_rows.csv"), index=False, encoding="utf-8-sig")
        rec("FAIL", "코드·연도·월 비수치/빈 값 또는 유효하지 않은 연월",
            f"{int(bad.sum())}행 -> bad_rows.csv, 이후 점검에서 제외")
        df = df[~bad.values].copy()
        if df.empty:
            return finish(a)
    else:
        rec("PASS", "코드·연도·월에 비수치/빈 값 없음")
    df[["code", "year", "month"]] = df[["code", "year", "month"]].astype(int)
    ym_chk = df["ym_str"].map(parse_ym)
    n_bad = int((ym_chk != df["ym"]).sum())
    rec("PASS" if n_bad == 0 else "FAIL", "진료년월 문자열 == 연도·월", f"불일치 {n_bad}행")

    # ---------------------------------------------------------------- 2. 키
    print("\n" + "=" * 70, "\n2. 키 무결성\n" + "=" * 70)
    dup = df.duplicated(["code", "ym"], keep=False)
    rec("PASS" if not dup.any() else "FAIL", "(코드, 연월) 중복 없음", f"중복 {int(dup.sum())}행")
    if dup.any():
        df[dup].sort_values(["code", "ym"]).to_csv(out("dup_keys.csv"), index=False, encoding="utf-8-sig")
    cn = df.groupby("code")["name"].nunique()
    nc = df.groupby("name")["code"].nunique()
    rec("PASS" if (cn <= 1).all() and (nc <= 1).all() else "WARN", "코드 <-> 분류명 1:1",
        f"코드당 다중명 {int((cn > 1).sum())}, 이름당 다중코드 {int((nc > 1).sum())}")
    rec("INFO", "코드 자릿수", str(pd.Series(df.code.astype(int).astype(str).str.len().unique()).tolist()))

    # ---------------------------------------------------------------- 3. 격자
    print("\n" + "=" * 70, "\n3. 격자(코드 x 월) 완전성\n" + "=" * 70)
    months = pd.date_range(df.ym.min(), df.ym.max(), freq="MS")
    codes = sorted(df.code.unique())
    grid_n = len(codes) * len(months)
    miss = pd.MultiIndex.from_product([codes, months], names=["code", "ym"]).difference(
        df.set_index(["code", "ym"]).index).to_frame(index=False)
    rec("INFO", "기간", f"{months[0]:%Y-%m} ~ {months[-1]:%Y-%m} ({len(months)}개월)")
    rec("INFO", "코드 수", f"{len(codes)}개")
    rec("INFO", "격자 크기 / 실제 행 / 누락", f"{grid_n:,} / {len(df):,} / {len(miss):,}")

    names = df.groupby("code")["name"].first()
    first = df.groupby("code")["ym"].min()
    last = df.groupby("code")["ym"].max()
    if len(miss):
        miss["name"] = miss.code.map(names)
        miss["kind"] = np.where(miss.ym < miss.code.map(first), "lead",
                                np.where(miss.ym > miss.code.map(last), "trail", "gap"))
        miss.to_csv(out("missing_keys.csv"), index=False, encoding="utf-8-sig")
        sm = (miss.groupby(["code", "name"])
              .agg(n_missing=("ym", "size"), miss_from=("ym", "min"), miss_to=("ym", "max"),
                   n_lead=("kind", lambda s: int((s == "lead").sum())),
                   n_trail=("kind", lambda s: int((s == "trail").sum())),
                   n_gap=("kind", lambda s: int((s == "gap").sum()))).reset_index())
        sm["first_present"] = sm.code.map(first)
        sm["last_present"] = sm.code.map(last)
        sm.to_csv(out("missing_by_code.csv"), index=False, encoding="utf-8-sig")
        n_gap = int((miss.kind == "gap").sum())
        rec("WARN", "누락 행 존재", f"{len(miss):,}행 / {miss.code.nunique()}개 코드 "
            f"(앞부분 {int((miss.kind == 'lead').sum())}, 뒷부분 {int((miss.kind == 'trail').sum())}, 중간 {n_gap})")
        rec("PASS" if n_gap == 0 else "WARN", "중간 결측(gap) 없음", f"{n_gap}행")
        print(sm.to_string(index=False))
    else:
        rec("PASS", "격자 완전")

    # ---------------------------------------------------------------- 4. NaN / 0
    print("\n" + "=" * 70, "\n4. 결측(NaN)과 0\n" + "=" * 70)
    na = df[df.qty.isna() | df.amt.isna()]
    both = int((na.qty.isna() & na.amt.isna()).sum())
    rec("INFO", "NaN 행", f"{len(na):,}행 (수량·금액 동시 {both:,}, 한쪽만 {len(na) - both:,})")
    rec("PASS" if len(na) == both else "WARN", "NaN은 수량·금액 동시 발생")
    v = df.dropna(subset=["qty", "amt"])
    z = pd.DataFrame({
        "valid_rows": v.groupby("code").size(),
        "zero_amt": v[v.amt == 0].groupby("code").size(),
        "zero_qty": v[v.qty == 0].groupby("code").size(),
        "qty0_amt_pos": v[(v.qty == 0) & (v.amt > 0)].groupby("code").size(),
        "amt0_qty_pos": v[(v.amt == 0) & (v.qty > 0)].groupby("code").size(),
    }).fillna(0).astype(int)
    z["nan_rows"] = df[df.amt.isna() | df.qty.isna()].groupby("code").size().reindex(z.index).fillna(0).astype(int)
    z["name"] = z.index.map(names)
    z.to_csv(out("zero_nan_by_code.csv"), encoding="utf-8-sig")
    na_codes = sorted(na.code.unique())
    rec("INFO", "NaN이 있는 코드", f"{len(na_codes)}개: {[int(c) for c in na_codes]}")
    allzero = z[(z.zero_amt == z.valid_rows) & (z.valid_rows > 0)]
    rec("INFO", "유효값이 전부 0인 코드", f"{len(allzero)}개: {[int(c) for c in allzero.index]}")
    n_pos = int((df.groupby("code").amt.max().fillna(0) > 0).sum())
    rec("INFO", "금액 > 0이 한 번이라도 있는 코드(= '유효 분류' 수)", f"{n_pos}개")
    rec("INFO", "0값 행", f"금액 0: {int((v.amt == 0).sum()):,} / 수량 0: {int((v.qty == 0).sum()):,} "
        f"(수량 0 & 금액>0: {int(((v.qty == 0) & (v.amt > 0)).sum()):,}, 금액 0 & 수량>0: {int(((v.amt == 0) & (v.qty > 0)).sum()):,}) -> 단위 절삭 가능성")

    # ---------------------------------------------------------------- 5. 값 범위
    print("\n" + "=" * 70, "\n5. 값 범위\n" + "=" * 70)
    rec("PASS" if not ((v.qty < 0) | (v.amt < 0)).any() else "FAIL", "음수 없음")
    rec("PASS" if ((v.qty % 1 == 0) & (v.amt % 1 == 0)).all() else "WARN", "수량·금액이 정수")
    print(v[["qty", "amt"]].describe().round(1).to_string())
    pos = v[v.qty > 0].assign(unit=lambda d: d.amt * 1000 / d.qty)
    print("\n단가(원/개) 분포:\n" + pos.unit.describe().round(1).to_string())

    # ---------------------------------------------------------------- 6. 연도·월 총액
    print("\n" + "=" * 70, "\n6. 총액 추세\n" + "=" * 70)
    yr = df.groupby("year").agg(amt=("amt", "sum"), qty=("qty", "sum"), n_codes_with_rows=("code", "nunique"))
    yr = yr.reindex(range(int(yr.index.min()), int(yr.index.max()) + 1))  # 빠진 연도는 NaN으로 남겨 YoY 왜곡 방지
    miss_years = [int(y) for y in yr.index[yr.amt.isna()]]
    rec("PASS" if not miss_years else "WARN", "연도 누락 없음", f"누락 연도 {miss_years}" if miss_years else "")
    yr["amt_yoy_%"] = (yr.amt.pct_change(fill_method=None) * 100).round(1)
    yr["qty_yoy_%"] = (yr.qty.pct_change(fill_method=None) * 100).round(1)
    yr.to_csv(out("yearly_total.csv"), encoding="utf-8-sig")
    print(yr.to_string())
    mt = df.groupby("ym").amt.sum().reindex(months)  # 전체 월 격자로 재색인 (빠진 달은 NaN)
    yoy = (mt.pct_change(12, fill_method=None) * 100).dropna()
    pd.DataFrame({"amt": mt, "yoy_%": mt.pct_change(12, fill_method=None) * 100}).to_csv(
        out("monthly_total.csv"), encoding="utf-8-sig")
    print("\n월 총액 YoY 상위 3 / 하위 3 (%):")
    print(pd.concat([yoy.nlargest(3), yoy.nsmallest(3)]).round(1).to_string())
    cnt = df[df.amt.notna()].groupby("ym").code.nunique()
    rec("INFO", "월별 유효 코드 수 (최소~최대)", f"{int(cnt.min())} ~ {int(cnt.max())}")
    share = df[df.year == df.year.max()].groupby("code").amt.sum()
    part = set(na_codes) | (set(miss.code.unique()) if len(miss) else set())
    rec("INFO", f"결측·부분기간 코드의 {int(df.year.max())}년 금액 비중",
        f"{share.reindex(sorted(part)).fillna(0).sum() / share.sum() * 100:.3f}% ({len(part)}개 코드)")

    # ---------------------------------------------------------------- 7. 단가 급변
    print("\n" + "=" * 70, "\n7. 단가 급변 (수량 break 후보)\n" + "=" * 70)
    # 단가 계산은 수량>0 행만 가능하므로 수량 0 행은 별도로 점검한다 (수량이 0으로 절삭된 구간의 정의 변경 후보)
    n_excl = int(((v.qty == 0) & (v.amt > 0)).sum())
    qz = v[(v.qty == 0) & (v.amt >= a.min_amt)].copy()
    qz["name"] = qz.code.map(names)
    qz[["code", "name", "ym", "qty", "amt"]].to_csv(out("qty_zero_amt_pos.csv"), index=False, encoding="utf-8-sig")
    rec("INFO", "단가 계산에서 제외되는 행(수량 0 & 금액>0)", f"{n_excl:,}행 (코드별은 zero_nan_by_code.csv의 qty0_amt_pos)")
    rec("WARN" if len(qz) else "PASS", f"수량 0인데 금액>={a.min_amt:g}인 행 (수량 절삭·정의 변경 후보)",
        f"{len(qz)}행 / {qz.code.nunique()}개 코드 -> qty_zero_amt_pos.csv")

    p = pos.sort_values(["code", "ym"]).copy()
    p["prev_ym"] = p.groupby("code").ym.shift()
    p["prev_qty"] = p.groupby("code").qty.shift()
    p["prev_amt"] = p.groupby("code").amt.shift()
    p["prev_unit"] = p.groupby("code").unit.shift()
    consec = (p.ym.dt.year * 12 + p.ym.dt.month) - (p.prev_ym.dt.year * 12 + p.prev_ym.dt.month) == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        p["dlog_unit"] = np.log(p.unit / p.prev_unit)
    p = p[consec & np.isfinite(p.dlog_unit)]
    p["name"] = p.code.map(names)
    big = p[p.dlog_unit.abs() > a.jump_thr]
    # 금액은 양 달 모두 하한 이상, 수량 하한은(지정 시) 한쪽 달만 만족해도 포함 (수량 정의 변경은 한쪽 수량이 작을 수 있음)
    mid = big[(big.amt >= a.min_amt) & (big.prev_amt >= a.min_amt)]
    if a.min_qty > 0:
        mid = mid[(mid.qty >= a.min_qty) | (mid.prev_qty >= a.min_qty)]
    cols = ["code", "name", "prev_ym", "ym", "prev_qty", "qty", "prev_amt", "amt", "prev_unit", "unit", "dlog_unit"]
    big[cols].to_csv(out("unit_jumps_all.csv"), index=False, encoding="utf-8-sig")
    mid[cols].sort_values("dlog_unit", key=abs, ascending=False).to_csv(
        out("unit_jumps_midsize.csv"), index=False, encoding="utf-8-sig")
    rec("INFO", f"|Δln 단가| > {a.jump_thr} 전체", f"{len(big)}건 / {big.code.nunique()}개 코드 (소규모 절삭 노이즈 포함)")
    q_txt = f", 수량>={a.min_qty:g} 한쪽 이상" if a.min_qty > 0 else ""
    rec("WARN" if len(mid) else "PASS", f"위 중 중간 규모(금액>={a.min_amt:g} 양 달 모두{q_txt})",
        f"{len(mid)}건 / {mid.code.nunique()}개 코드 -> unit_jumps_midsize.csv (사람이 검토할 목록이며 합격 기준 아님)")
    if len(mid):
        show = mid[cols].sort_values("dlog_unit", key=abs, ascending=False).head(30).copy()
        for c in ("prev_ym", "ym"):
            show[c] = show[c].dt.strftime("%Y-%m")
        print(show.round(2).to_string(index=False))

    return finish(a)


def finish(a):
    print("\n" + "=" * 70, "\n요약\n" + "=" * 70)
    s = pd.Series([r[0] for r in RESULTS]).value_counts()
    print({k: int(s.get(k, 0)) for k in ("PASS", "WARN", "FAIL", "INFO")})
    with open(os.path.join(a.outdir, "report.txt"), "w", encoding="utf-8-sig") as f:
        for st, ck, dt in RESULTS:
            f.write(f"[{st}] {ck}" + (f" -> {dt}" if dt else "") + "\n")
    print(f"결과 파일: {os.path.abspath(a.outdir)}")
    return 1 if (s.get("FAIL", 0) > 0) else 0


if __name__ == "__main__":
    sys.exit(main())