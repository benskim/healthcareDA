#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_api_data.py — 상병 데이터(data_*.pq)의 완전성·정합성 점검 (데이터 무결성 점검 전용)

보유 분류 데이터(meft_*.csv, 기준값)와 API 수집분을 (진료년월 × 약효분류) 단위로 대조해 다음을 보고한다.
  1) 전체·연도별 금액/수량 커버리지 (커버리지 = 상병 합계 ÷ (보유 값 × 단위 환산계수))
  2) 상병 데이터가 전혀 없는 분류, 월 누락이 있는 분류, 잘림 의심(수신 행 수가 10,000의 배수)
  3) 보험구분코드 분포, 수집 조건 기록(요청_보험자/요청_조제처방), 상병코드 형식(길이, 첫 글자)

이 스크립트는 분류의 "사용 가능 여부"를 판정하지 않는다. 판정은 classify_usability.py가 단독으로 담당한다
(두 스크립트가 서로 다른 모집단·기준으로 판정하면 결과가 어긋나기 때문이다).
집계·해당없음 코드(110, 140, 230, 250, 310, 398, 999)는 분석 단위가 아니므로 양쪽 모두에서 제외한다.

입력  : data_*.pq (컬럼: 진료년월, 약효분류번호, 상병코드, 의약품사용금액, 총사용량, [보험구분코드, 요청_*]), meft_*.csv, [log.csv]
출력  : <out>/audit_report.md, <out>/audit_by_class.csv, <out>/audit_by_class_month.csv

실행  : python audit_api_data.py --api-dir api_data --csv meft_2014-2025.csv --out audit [--log api_data/log.csv]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CAP = 10_000
EXCLUDE_CODES = {110, 140, 230, 250, 310, 398, 999}   # 상위 집계·해당없음 (분석 단위 아님)
SCALES = [1.0, 1e3, 1e6]                               # 허용하는 단위 환산계수


def load_api(api_dir: Path):
    files = sorted(api_dir.glob("data_*.pq"))
    if not files:
        raise SystemExit(f"{api_dir} 에 data_*.pq 가 없습니다.")
    parts, insup, dx_len, dx_first, req = [], Counter(), Counter(), Counter(), Counter()
    for f in files:
        d = pd.read_parquet(f)
        d["진료년월"] = d["진료년월"].astype(str).str.replace(r"\D", "", regex=True).str[:6]
        d["code"] = pd.to_numeric(d["약효분류번호"], errors="coerce")
        d = d[~d["code"].isin(EXCLUDE_CODES)]
        if "보험구분코드" in d.columns:
            for k, v in d.groupby("보험구분코드")["의약품사용금액"].sum().items():
                insup[str(k)] += float(v)
        if "요청_조제처방" in d.columns:
            for k, v in (d["요청_보험자"].astype(str) + "/" + d["요청_조제처방"].astype(str)).value_counts().items():
                req[k] += int(v)
        dx = d["상병코드"].astype(str)
        for k, v in dx.str.len().value_counts().items():
            dx_len[int(k)] += int(v)
        for k, v in dx.str[:1].value_counts().items():
            dx_first[str(k)] += int(v)
        g = d.groupby(["진료년월", "code"]).agg(api_amt=("의약품사용금액", "sum"), api_qty=("총사용량", "sum"),
                                              rows=("상병코드", "size"), n_dx=("상병코드", "nunique")).reset_index()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["code"] = out["code"].astype("Int64")
    return out, insup, dx_len, dx_first, req


def load_base(csv: Path) -> pd.DataFrame:
    b = pd.read_csv(csv, encoding="utf-8-sig")
    b = b.rename(columns={"약효분류코드": "code", "진료년월": "ym", "수량_천개": "base_qty", "사용금액_백만원": "base_amt"})
    b["진료년월"] = b["ym"].astype(str).str.replace(r"\D", "", regex=True).str[:6]
    b["code"] = pd.to_numeric(b["code"], errors="coerce")
    b = b[~b["code"].isin(EXCLUDE_CODES)].dropna(subset=["code"])
    b["code"] = b["code"].astype("Int64")
    return b[["진료년월", "code", "base_amt", "base_qty"]]


def pick_scale(ratio: pd.Series) -> float:
    """비율 중앙값에 로그 거리상 가장 가까운 허용 환산계수(1, 1e3, 1e6). 커버리지가 낮아도 자릿수가 틀어지지 않게 후보를 제한한다."""
    r = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    if r.empty:
        return 1.0
    lm = np.log10(r.median())
    return min(SCALES, key=lambda s: abs(np.log10(s) - lm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-dir", default="sick")
    ap.add_argument("--csv", default="meft_2014-2025.csv")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--log", default=None, help="수집기 log.csv (월 누락 원인 구분용)")
    ap.add_argument("--min-base-amt", type=float, default=100.0, help="평가할 최소 월 보유 금액(백만원), 절삭 영향 배제")
    ap.add_argument("--scale-amt", type=float, default=None)
    ap.add_argument("--scale-qty", type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    api, insup, dx_len, dx_first, req = load_api(Path(a.api_dir))
    base = load_base(Path(a.csv))
    months = sorted(api["진료년월"].unique())
    base = base[base["진료년월"].isin(months)]
    m = base.merge(api, on=["진료년월", "code"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    ref = both[(both["base_amt"] >= a.min_base_amt) & (both["api_amt"] > 0)]
    scale_amt = a.scale_amt or pick_scale(ref["api_amt"] / ref["base_amt"])
    refq = both[(both["base_qty"] >= 100) & (both["api_qty"] > 0)]
    scale_qty = a.scale_qty or pick_scale(refq["api_qty"] / refq["base_qty"])
    m["eval"] = m["base_amt"].fillna(0) >= a.min_base_amt
    m["cov_amt"] = m["api_amt"] / (m["base_amt"].where(m["base_amt"] > 0) * scale_amt)
    m["cov_qty"] = m["api_qty"] / (m["base_qty"].where(m["base_qty"] > 0) * scale_qty)
    m["year"] = m["진료년월"].str[:4]
    m.to_csv(out / "audit_by_class_month.csv", index=False, encoding="utf-8-sig")

    ev = m[m["eval"]]
    tot_base = ev["base_amt"].sum() * scale_amt
    cov_all = ev["api_amt"].fillna(0).sum() / tot_base
    cq = m[m["base_qty"].fillna(0) > 0]
    cov_qty_all = cq["api_qty"].fillna(0).sum() / (cq["base_qty"].sum() * scale_qty)
    yr = ev.groupby("year").apply(lambda x: x["api_amt"].fillna(0).sum() / (x["base_amt"].sum() * scale_amt))
    yq = cq.assign(year=cq["진료년월"].str[:4]).groupby("year").apply(
        lambda x: x["api_qty"].fillna(0).sum() / (x["base_qty"].sum() * scale_qty))

    log = None
    if a.log and Path(a.log).exists():
        log = pd.read_csv(a.log, encoding="utf-8-sig")
        log["분류"] = pd.to_numeric(log["분류"], errors="coerce").astype("Int64")

    rows = []
    for code, g in ev.groupby("code"):
        got = g[g["api_amt"].notna()]
        miss = g[g["api_amt"].isna()]
        base_sum = g["base_amt"].sum() * scale_amt
        cause = ""
        if len(miss):
            if log is None:
                cause = "로그 없음"
            else:
                lg = log[(log["분류"] == code) & log["진료년월"].astype(str).isin(set(miss["진료년월"]))]
                cause = "미시도(로그에 없음)" if lg.empty else ",".join(
                    f"{k}:{v}" for k, v in lg["상태"].astype(str).str.split(":").str[0].value_counts().items())
        rows.append({"code": code, "eval_months": len(g), "months_with_api": len(got), "months_missing": len(miss),
                     "no_api_data": len(got) == 0, "share_of_base": base_sum / tot_base,
                     "coverage_matched": got["api_amt"].sum() / (got["base_amt"].sum() * scale_amt) if len(got) else np.nan,
                     "coverage_incl_missing": g["api_amt"].fillna(0).sum() / base_sum,
                     "truncation_months": int((got["rows"] % CAP == 0).sum()), "cause": cause})
    ct = pd.DataFrame(rows).sort_values("share_of_base", ascending=False)
    ct.to_csv(out / "audit_by_class.csv", index=False, encoding="utf-8-sig")

    none = ct[ct["no_api_data"]]
    partial = ct[(~ct["no_api_data"]) & (ct["months_missing"] > 0)]
    trunc = ct[ct["truncation_months"] > 0]
    L = ["# 상병 데이터 점검 결과 (무결성 점검)\n",
         f"- 대상: {len(months)}개월 ({months[0]} ~ {months[-1]}), 집계·해당없음 코드 제외, 평가 분류 {ct['code'].nunique()}개 (월 보유 금액 ≥ {a.min_base_amt:g}백만원인 달이 있는 분류)",
         f"- 단위 환산계수(추정): 금액 {scale_amt:g}, 수량 {scale_qty:g}  (후보: {', '.join(f'{s:g}' for s in SCALES)})",
         f"- **금액 커버리지(평가 월 가중, 미수집 포함): {cov_all:.1%}**, 수량 {cov_qty_all:.1%}",
         f"- 상병 데이터가 전혀 없는 분류: **{len(none)}개, 보유 금액 비중 {none['share_of_base'].sum():.2%}**",
         f"- 일부 월만 누락된 분류: {len(partial)}개 (비중 {partial['share_of_base'].sum():.2%})",
         f"- 잘림 의심(수신 행 수가 {CAP:,}의 배수인 월이 있는 분류): {len(trunc)}개",
         "- **분류별 사용 가능 여부 판정은 `classify_usability.py`가 담당한다.**",
         "", "## 연도별 가중 금액 커버리지", "", "| 연도 | 금액 | 수량 |", "|---|---:|---:|"]
    for y in yr.index:
        L.append(f"| {y} | {yr[y]:.1%} | {yq.get(y, np.nan):.1%} |")
    L.append(f"\n- 최대-최소 {yr.max() - yr.min():.1%}p")
    L += ["", "## 상병 데이터가 전혀 없는 분류", ""]
    L += (["| 분류 | 보유 금액 비중 | 월 누락 | 원인 |", "|---|---:|---:|---|"] +
          [f"| {r['code']} | {r['share_of_base']:.2%} | {r['months_missing']} | {r['cause']} |" for _, r in none.iterrows()]) if len(none) else ["- 없음"]
    if len(partial):
        L += ["", "## 일부 월 누락 분류 (금액 비중 상위 15)", "", "| 분류 | 비중 | 누락 월 / 평가 월 | 원인 |", "|---|---:|---:|---|"]
        L += [f"| {r['code']} | {r['share_of_base']:.2%} | {r['months_missing']}/{r['eval_months']} | {r['cause']} |"
              for _, r in partial.head(15).iterrows()]
    tot_insup = sum(insup.values())
    L += ["", "## 보험구분코드 분포 (금액 기준)", ""]
    if tot_insup > 0:
        L += ["| 코드 | 비중 |", "|---|---:|"] + [f"| {k} | {v / tot_insup:.1%} |" for k, v in sorted(insup.items())]
        L.append("\n- 보험자 구분 0(전체) 응답에 건강보험(4)·의료급여(5)·보훈(7) 행이 함께 포함되는지 위 분포로 확인한다.")
    else:
        L.append("- 보험구분코드 컬럼 없음")
    L += ["", "## 수집 조건 (보험자/조제처방)", ""]
    if req:
        L += ["| 조건 | 행 수 |", "|---|---:|"] + [f"| {k} | {v:,} |" for k, v in sorted(req.items())]
        if len(req) > 1 or "02" not in next(iter(req)):
            L.append("\n- **조건이 섞여 있거나 처방(02)이 아닙니다.** 보유 분류 데이터와 일치하는 기준은 0/02입니다. 다른 조건의 수집분은 재수집하세요.")
    else:
        L.append("- 수집 조건 기록이 없습니다(이전 수집분). 처방(02)·보험자 0(전체)로 받았는지 확인하세요.")
    tot_len, tot_first = sum(dx_len.values()), sum(dx_first.values())
    L += ["", "## 상병코드 형식", "", "| 길이 | 행 수 | 비중 |", "|---|---:|---:|"]
    L += [f"| {k} | {v:,} | {v / tot_len:.1%} |" for k, v in sorted(dx_len.items())]
    L += ["", "첫 글자 분포 (상위 8):", "", "| 첫 글자 | 비중 |", "|---|---:|"]
    L += [f"| {k} | {v / tot_first:.1%} |" for k, v in dx_first.most_common(8)]
    L.append("\n- 길이가 모두 4이고 첫 글자가 한 값에 몰려 있으면 접두 한 글자를 떼고 KCD 3단(예: `AE10` → `E10`)으로 정규화하는 규칙을 쓸 수 있다. 첫 글자가 여러 값이면 접두의 의미를 확인해야 한다.")
    L += ["", "## 판독 가이드",
          "- 월 누락이 있고 수집기 로그에 failed가 있으면 재수집한다. 로그가 없거나 empty(totalCount 0)이면 자료에 없는 것이다.",
          "- 연도별·월별 커버리지의 시기별 변화는 분류별 판정에서 시장 공통 이상 구간과 분류 개별 결손으로 나누어 처리한다 (classify_usability.py).",
          "- 수량 커버리지는 판정에 쓰지 않는다 (수량 단위·정의 변경 이력이 있음)."]
    (out / "audit_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
