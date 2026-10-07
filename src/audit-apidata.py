#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_api_data.py — 상병 데이터(data_*.pq)의 완전성·정합성 점검과 분류별 사용 등급 판정

보유 분류 데이터(meft_*.csv)와 API 수집분을 (진료년월 × 약효분류) 단위로 대조한다.
  1) 커버리지 = 상병 합계 ÷ (보유 값 × 단위 환산계수). 환산계수는 비율 중앙값에 가장 가까운 10의 거듭제곱(--scale-amt/--scale-qty로 지정 가능)
  2) 분류별 커버리지의 수준·안정성(월별 변동계수)·월 누락·잘림 의심으로 사용 등급을 매긴다
       A    : 커버리지 ≥ 98%, 변동계수 ≤ 2%, 월 누락 없음 → 상병 귀속·상병 시장 총량에 사용
       B    : 커버리지 ≥ 90%, 변동계수 ≤ 5%            → 상병 구성(비중·추세)에 사용, 미귀속 잔여 항을 함께 표기
       C    : 그 외(저조·불안정·월 누락·잘림·시기별 변화) → 상병 귀속 보류, 원인 규명 후 재수집 대상
            (시기별 변화: 초기 25% 구간 대비 최근 25% 구간 커버리지 차이가 3%p 초과)
       미수집: API 행이 전혀 없음                       → 로그로 원인 구분(실패/응답 없음/미시도)
  3) 귀속 신뢰비: 커버리지가 기간 중 바뀌면 상병 합계의 성장률이 총계 성장률에서 벗어난다.
       오차 ≈ (최근 연도 커버리지 ÷ 첫 연도 커버리지 − 1), 귀속 신뢰비 = |오차| ÷ max(|분류 성장률|, 5%).
       0.2 미만이면 커버리지 변화가 분류 성장의 20% 미만이므로 상병 귀속을 신뢰한다 (초기 기준).
  4) 상병 귀속 사용 범위(운영 라벨):
       UNAVAILABLE : 해당 분류의 상병 데이터가 전혀 없음 → 상병 귀속 제외 (분류 총량 분석에는 계속 사용), 제외 분류의 시장 비중을 표시
       USABLE      : 미귀속률 ≤ --max-unattributed(기본 4%, 운영 임계값) 이고 시기별로 안정(귀속 신뢰비 < 0.2, 월 누락·잘림 없음)
       LIMITED     : 수집은 되지만 위 조건 미달 → 귀속 분석은 주의 표기 또는 제외
       상병 금액을 100%에 맞추려고 배율을 곱하는 보정은 하지 않는다. 관측된 상병 금액 + 미귀속 잔여항으로 그대로 둔다.
  5) 연도별 가중 커버리지(잔여 비중의 시기별 변화), 보험구분코드 분포, 상병 코드 형식 확인
  4) 수량은 같은 방식으로 요약만 한다 (수량은 분석 판정에 사용하지 않음)

입력  : data_*.pq (컬럼: 진료년월, 약효분류번호, 상병코드, 의약품사용금액, 총사용량, [보험구분코드]), meft_*.csv, [log.csv]
출력  : <out>/audit_report.md, <out>/audit_class_tiers.csv, <out>/audit_by_class_month.csv

실행  : python audit_api_data.py --api-dir api_data --csv meft_2014-2025.csv --out audit [--log api_data/log.csv]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CAP = 10_000
TIER_A = (0.98, 0.02)
TIER_B = (0.90, 0.05)


def load_api(api_dir: Path):
    files = sorted(api_dir.glob("data_*.pq"))
    if not files:
        raise SystemExit(f"{api_dir} 에 data_*.pq 가 없습니다.")
    parts, insup, dx_len, req = [], Counter(), Counter(), Counter()
    for f in files:
        d = pd.read_parquet(f)
        d["진료년월"] = d["진료년월"].astype(str).str.replace(r"\D", "", regex=True).str[:6]
        d["code"] = pd.to_numeric(d["약효분류번호"], errors="coerce")
        if "보험구분코드" in d.columns:
            for k, v in d.groupby("보험구분코드")["의약품사용금액"].sum().items():
                insup[str(k)] += float(v)
        if "요청_조제처방" in d.columns:
            for k, v in (d["요청_보험자"].astype(str) + "/" + d["요청_조제처방"].astype(str)).value_counts().items():
                req[k] += int(v)
        for k, v in d["상병코드"].astype(str).str.len().value_counts().items():
            dx_len[int(k)] += int(v)
        g = d.groupby(["진료년월", "code"]).agg(api_amt=("의약품사용금액", "sum"), api_qty=("총사용량", "sum"),
                                              rows=("상병코드", "size"), n_dx=("상병코드", "nunique"),
                                              qty_nan=("총사용량", lambda x: x.isna().mean())).reset_index()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["code"] = out["code"].astype("Int64")
    return out, insup, dx_len, req


def load_base(csv: Path) -> pd.DataFrame:
    b = pd.read_csv(csv, encoding="utf-8-sig")
    b = b.rename(columns={"약효분류코드": "code", "진료년월": "ym", "수량_천개": "base_qty", "사용금액_백만원": "base_amt"})
    b["진료년월"] = b["ym"].astype(str).str.replace(r"\D", "", regex=True).str[:6]
    b["code"] = pd.to_numeric(b["code"], errors="coerce").astype("Int64")
    return b[["진료년월", "code", "base_amt", "base_qty"]].dropna(subset=["code"])


def pick_scale(ratio: pd.Series) -> float:
    r = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    return float(10 ** round(np.log10(r.median()))) if len(r) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-dir", default="api_data")
    ap.add_argument("--csv", default="meft_2014-2025.csv")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--log", default=None, help="수집기 log.csv (미수집 원인 구분용)")
    ap.add_argument("--min-base-amt", type=float, default=100.0, help="커버리지 계산에 쓸 최소 보유 금액(백만원), 절삭 영향 배제")
    ap.add_argument("--max-unattributed", type=float, default=0.04, help="USABLE 판정의 미귀속률 상한 (운영 임계값)")
    ap.add_argument("--scale-amt", type=float, default=None)
    ap.add_argument("--scale-qty", type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    api, insup, dx_len, req = load_api(Path(a.api_dir))
    base = load_base(Path(a.csv))
    months = sorted(api["진료년월"].unique())
    base = base[base["진료년월"].isin(months)]
    m = base.merge(api, on=["진료년월", "code"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"].copy()

    ref = both[(both["base_amt"] >= a.min_base_amt) & (both["api_amt"] > 0)]
    scale_amt = a.scale_amt or pick_scale(ref["api_amt"] / ref["base_amt"])
    refq = both[(both["base_qty"] >= 100) & (both["api_qty"] > 0)]
    scale_qty = a.scale_qty or pick_scale(refq["api_qty"] / refq["base_qty"])
    m["cov_amt"] = m["api_amt"] / (m["base_amt"].where(m["base_amt"] > 0) * scale_amt)
    m["cov_qty"] = m["api_qty"] / (m["base_qty"].where(m["base_qty"] > 0) * scale_qty)
    m["year"] = m["진료년월"].str[:4]
    m.to_csv(out / "audit_by_class_month.csv", index=False, encoding="utf-8-sig")

    # ---- 전체 커버리지
    bm = m[m["base_amt"].fillna(0) > 0]
    tot_base = bm["base_amt"].sum() * scale_amt
    cov_all = bm["api_amt"].fillna(0).sum() / tot_base
    cq = m[m["base_qty"].fillna(0) > 0]
    cov_qty_all = cq["api_qty"].fillna(0).sum() / (cq["base_qty"].sum() * scale_qty)

    # ---- 분류별 등급
    log = None
    if a.log and Path(a.log).exists():
        log = pd.read_csv(a.log, encoding="utf-8-sig")
        log["분류"] = pd.to_numeric(log["분류"], errors="coerce").astype("Int64")
    rows, yr_rows = [], []
    for code, g in bm.groupby("code"):
        g = g.sort_values("진료년월")
        got = g[g["api_amt"].notna()]
        base_sum = g["base_amt"].sum() * scale_amt
        cov_w = got["api_amt"].sum() / (got["base_amt"].sum() * scale_amt) if len(got) else np.nan
        cov_incl_missing = g["api_amt"].fillna(0).sum() / base_sum
        mon = got.loc[got["base_amt"] >= a.min_base_amt, "cov_amt"]
        cv = float(mon.std() / mon.mean()) if len(mon) >= 3 and mon.mean() > 0 else np.nan
        k = max(3, len(mon) // 4)
        drift = float(mon.iloc[-k:].mean() - mon.iloc[:k].mean()) if len(mon) >= 8 else np.nan
        by = g.groupby("year").agg(b=("base_amt", "sum"), n=("base_amt", "count"))
        by["mm"] = by["b"] / by["n"]
        growth = float(by["mm"].iloc[-1] / by["mm"].iloc[0] - 1) if len(by) >= 2 and by["mm"].iloc[0] > 0 else np.nan
        cov_y = {}
        for yy, gg in g.groupby("year"):
            gm = gg[gg["api_amt"].notna()]
            cov_y[yy] = gm["api_amt"].sum() / (gm["base_amt"].sum() * scale_amt) if len(gm) and gm["base_amt"].sum() > 0 else np.nan
        for yy, v in cov_y.items():
            yr_rows.append({"code": code, "year": yy, "unattributed": (1 - v) if pd.notna(v) else np.nan})
        cy = [v for v in cov_y.values() if pd.notna(v)]
        err = float(cy[-1] / cy[0] - 1) if len(cy) >= 2 and cy[0] > 0 else np.nan
        ratio_attr = abs(err) / max(abs(growth), 0.05) if pd.notna(err) and pd.notna(growth) else np.nan
        miss_m = int(len(g) - len(got))
        cap_m = int((got["rows"] % CAP == 0).sum())
        if len(got) == 0:
            tier = "미수집"
        elif miss_m > 0 or cap_m > 0 or (pd.notna(drift) and abs(drift) > 0.03):
            tier = "C"
        elif cov_w >= TIER_A[0] and (pd.isna(cv) or cv <= TIER_A[1]):
            tier = "A"
        elif cov_w >= TIER_B[0] and (pd.isna(cv) or cv <= TIER_B[1]):
            tier = "B"
        else:
            tier = "C"
        cause = ""
        if tier == "C" and miss_m == 0 and cap_m == 0:
            if pd.notna(cv) and cv > TIER_B[1]:
                cause = "월별 불안정(수집 손실 의심)"
            elif pd.notna(drift) and abs(drift) > 0.03:
                cause = "시기별 변화"
            elif pd.notna(cov_w) and cov_w < TIER_B[0]:
                cause = "커버리지 저조"
        if tier in ("미수집", "C") and miss_m > 0:
            if log is None:
                cause = "로그 없음"
            else:
                miss_ym = set(g.loc[g["api_amt"].isna(), "진료년월"])
                lg = log[(log["분류"] == code)]
                lg = lg[lg["진료년월"].astype(str).isin(miss_ym)]
                if lg.empty:
                    cause = "미시도(로그에 없음)"
                else:
                    cause = ",".join(f"{k}:{v}" for k, v in lg["상태"].str.split(":").str[0].value_counts().items())
        if tier == "미수집":
            status = "UNAVAILABLE"
        elif (tier in ("A", "B") and pd.notna(cov_w) and (1 - cov_w) <= a.max_unattributed + 1e-9
              and (pd.isna(ratio_attr) or ratio_attr < 0.2)):
            status = "USABLE"
        else:
            status = "LIMITED"
        rows.append({"code": code, "tier": tier, "status": status, "unattributed_rate": (1 - cov_w) if pd.notna(cov_w) else np.nan, "months_base": len(g), "months_missing": miss_m,
                     "base_amt_won": base_sum, "share_of_base": base_sum / tot_base,
                     "coverage_weighted": cov_w, "coverage_incl_missing": cov_incl_missing,
                     "coverage_cv": cv, "coverage_drift": drift, "base_growth": growth, "attrib_error": err,
                     "attrib_ratio": ratio_attr, "truncation_months": cap_m, "cause": cause})
    ct = pd.DataFrame(rows).sort_values("base_amt_won", ascending=False)
    ct["uncovered_share_of_base"] = ct["share_of_base"] * (1 - ct["coverage_incl_missing"])
    ct.to_csv(out / "audit_class_tiers.csv", index=False, encoding="utf-8-sig")
    if yr_rows:
        pd.DataFrame(yr_rows).pivot(index="code", columns="year", values="unattributed").to_csv(
            out / "audit_class_year_unattributed.csv", encoding="utf-8-sig")

    # ---- 연도별 가중 커버리지
    yr = bm.groupby("year").apply(lambda x: x["api_amt"].fillna(0).sum() / (x["base_amt"].sum() * scale_amt))
    yq = cq.assign(year=cq["진료년월"].str[:4]).groupby("year").apply(
        lambda x: x["api_qty"].fillna(0).sum() / (x["base_qty"].sum() * scale_qty))

    st_share = ct.groupby("status")["share_of_base"].sum()
    st_n = ct["status"].value_counts()
    tier_share = ct.groupby("tier")["share_of_base"].sum()
    tier_n = ct["tier"].value_counts()
    unavail = ct[ct["status"] == "UNAVAILABLE"]
    L = ["# 상병 데이터 점검 결과\n",
         "## 상병 귀속 분석 사용 범위\n",
         f"- 미귀속률 상한(운영 임계값): {a.max_unattributed:.0%}",
         "", "| 상태 | 분류 수 | 보유 금액 비중 |", "|---|---:|---:|"] + [
         f"| {s_} | {int(st_n.get(s_, 0))} | {st_share.get(s_, 0):.1%} |" for s_ in ["USABLE", "LIMITED", "UNAVAILABLE"]] + [
         "", f"- 상병 귀속 분석 대상(USABLE): {int(st_n.get('USABLE', 0))}개, 주의 표기(LIMITED): {int(st_n.get('LIMITED', 0))}개, 제외(UNAVAILABLE): {len(unavail)}개",
         f"- **제외 분류의 시장 비중: {unavail['share_of_base'].sum():.2%}** — 이 시장은 질환별 귀속을 확인할 수 없음 (분류 총량 분석에는 사용)",
         "- 제외 분류: " + (", ".join(str(c) for c in unavail["code"].tolist()) if len(unavail) else "없음"), "",
         "# 상세 점검\n",
         f"- 대상: {len(months)}개월 ({months[0]} ~ {months[-1]}), 보유 분류 {ct['code'].nunique()}개",
         f"- 단위 환산계수(추정): 금액 {scale_amt:g}, 수량 {scale_qty:g}",
         f"- **금액 커버리지(보유 금액 가중, 미수집 포함): {cov_all:.1%}**, 수량 {cov_qty_all:.1%}",
         "", "## 분류별 사용 등급", "", "| 등급 | 분류 수 | 보유 금액 비중 |", "|---|---:|---:|"]
    for t in ["A", "B", "C", "미수집"]:
        L.append(f"| {t} | {int(tier_n.get(t, 0))} | {tier_share.get(t, 0):.1%} |")
    ok_ab = ct[ct["tier"].isin(["A", "B"])]
    trust = ok_ab.loc[ok_ab["attrib_ratio"].fillna(0) < 0.2, "share_of_base"].sum()
    L += ["", f"- 사용 가능(A+B) 금액 비중: **{tier_share.get('A', 0) + tier_share.get('B', 0):.1%}**",
          f"- 그 중 귀속 신뢰비 < 0.2 (커버리지 변화가 분류 성장의 20% 미만): **{trust:.1%}**",
          "", "## 연도별 가중 금액 커버리지 (시기별로 변하면 상병 귀속이 편향됨)", "", "| 연도 | 금액 | 수량 |", "|---|---:|---:|"]
    for y in yr.index:
        L.append(f"| {y} | {yr[y]:.1%} | {yq.get(y, np.nan):.1%} |")
    spread = float(yr.max() - yr.min())
    L.append(f"\n- 연도별 금액 커버리지 최대-최소 {spread:.1%}p " + ("(안정)" if spread <= 0.03 else "(**3%p 초과: 미귀속 잔여 비중이 시기별로 변함**)"))
    L += ["", "## 미수집·C등급 분류 (미커버 금액 비중 상위 15)", "",
          "| 분류 | 등급 | 보유 금액 비중 | 커버리지 | 월 누락 | 잘림 | 원인 |", "|---|---|---:|---:|---:|---:|---|"]
    for _, r in ct[ct["tier"].isin(["미수집", "C"])].sort_values("uncovered_share_of_base", ascending=False).head(15).iterrows():
        cvg = "-" if pd.isna(r["coverage_weighted"]) else f"{r['coverage_weighted']:.1%}"
        L.append(f"| {r['code']} | {r['tier']} | {r['share_of_base']:.2%} | {cvg} | {r['months_missing']} | {r['truncation_months']} | {r['cause']} |")
    tot_insup = sum(insup.values())
    L += ["", "## 보험구분코드 분포 (금액 기준)", ""]
    if tot_insup > 0:
        L += ["| 코드 | 비중 |", "|---|---:|"] + [f"| {k} | {v / tot_insup:.1%} |" for k, v in sorted(insup.items())]
        L.append("\n- 보험자 구분 파라미터 4·5·7은 호출이 되지 않으므로 보험자별 분리는 불가하다. 응답 행의 보험구분코드에 '5'(의료급여)·'7'(보훈)이 없으면 전체(0) 응답이 건강보험 위주일 수 있다.")
    else:
        L.append("- 보험구분코드 컬럼 없음")
    L += ["", "## 수집 조건 (보험자/조제처방)", ""]
    if req:
        L += ["| 조건 | 행 수 |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in sorted(req.items())]
        if len(req) > 1:
            L.append("\n- **조건이 섞여 있습니다.** 보유 분류 데이터와 일치하는 기준은 0/02(전체/처방)입니다. 다른 조건의 행은 재수집하세요.")
        elif "02" not in next(iter(req)):
            L.append("\n- 보유 분류 데이터와 일치하는 기준은 처방(02)입니다. 현재 조건은 다릅니다. 재수집하세요.")
    else:
        L.append("- 수집 조건 기록이 없습니다(이전 수집분). 처방(02)·보험자 0(전체)로 받았는지 확인하세요. 조제(01)이면 커버리지가 약 50%입니다.")
    L += ["", "## 상병코드 길이 분포", "", "| 길이 | 행 수 |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in sorted(dx_len.items())]
    L.append("\n- 3자리(KCD 3단)와 4자리(접두 한 글자)가 섞여 있으면 상병 코드 정규화 규칙이 필요함")
    L += ["", "## 판독 가이드",
          "- **월 누락·잘림**이 있는 분류는 먼저 재수집한다 (수집기는 log.csv로 이어받기·재시도). 재수집 후 다시 점검한다.",
          "- 재수집 후에도 커버리지가 100% 미만이고 분류별로 일정하면 자료 정의상의 차이다 (상병 미기재 청구, 비식별 처리 등). 호출 조건은 처방(02)·보험자 0(전체)로 확정되었다 (조제 01은 약 50%, 보험자 4·5·7은 응답 없음).",
          "- 커버리지가 월별로 들쭉날쭉하면 수집 손실, 시기별로 변하면 집계 방식 변화를 의심한다.",
          "- 수량 커버리지는 판정에 쓰지 않는다 (수량 단위·정의 변경 이력이 있음)."]
    (out / "audit_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:16]))


if __name__ == "__main__":
    main()