#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_usability.py — 약효분류 × 월 단위 "상병 귀속 사용 가능성" 분류

분류 총량(보유 분류 데이터)은 기준값이다. 이 스크립트는 상병 데이터(API 수집분)를 분류 × 월, 분류 × 연,
그리고 YoY 쌍 단위로 "귀속 분석에 쓸 수 있는가"를 판정한다. 상병 금액에 배율을 곱하는 보정은 하지 않는다.

원리
  상병 합계의 성장률은 총계 성장률에서 커버리지 변화율만큼 벗어난다.
  따라서 커버리지의 수준보다 "그 분류의 평소 수준에서 벗어났는가"와 "연도 간 커버리지가 변했는가"가 핵심이다.

월 단위 상태 (분류 × 월)
  TINY        : 보유 금액이 최소 규모 미만 (절삭 영향) → 평가 제외
  ABSENT      : 보유 금액은 있는데 상병 데이터가 없음 (totalCount 0)
  OK          : 커버리지가 그 분류의 평소 수준(중앙값) ±tol 이내
  DIP_MARKET  : 평소보다 tol 넘게 낮고, 그 달에 시장 전체 커버리지도 낮음 (시장 공통 이상 구간)
  DIP_CLASS   : 평소보다 tol 넘게 낮은데 시장은 정상 (분류 개별 결손)
  HIGH        : 평소보다 tol 넘게 높음 (중복 집계 등)

연 단위 (분류 × 연)
  usable_year : 평가 월의 90% 이상이 OK
  yoy_attrib_ok : 연속한 두 연도가 모두 usable이고, 커버리지 변화율 |e| ≤ yoy_tol(3%) 이거나 |e| ≤ 0.2 × |분류 성장률(보유 기준)|
                  → 아니면 그 YoY는 상병 귀속에서 제외(mask). 분류 총량의 YoY에는 영향 없음.

분류 상태 (분류 전체)
  ABSENT_ALL : 평가 월의 95% 이상이 ABSENT → 상병 귀속 불가 (분류 총량은 사용)
  USABLE     : OK 월 비중 ≥ 90% (시장 공통 이상 구간의 달은 분류 자체의 품질이 아니므로 분모에서 제외) 이고 평소 커버리지 ≥ min_level(기본 90%)
               시장 공통 이상 구간은 월·연·YoY 단위 판정(위)에서 따로 처리한다
  LIMITED    : 그 외 (일부 월·연만 사용)

입력  : data_*.pq (api_data), meft_*.csv     (audit_api_data.py와 같은 폴더)
출력  : <out>/usability_class_month.csv, usability_class_year.csv, usability_class_summary.csv, usability_report.md

실행  : python classify_usability.py --api-dir api_data --csv meft_2014-2025.csv --out usability
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import audit_api_data as au


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-dir", default="sick")
    ap.add_argument("--csv", default="meft_2014-2025.csv")
    ap.add_argument("--out", default="usability")
    ap.add_argument("--min-base-amt", type=float, default=100.0, help="평가할 최소 월 보유 금액(백만원)")
    ap.add_argument("--tol", type=float, default=0.03, help="분류 평소 수준 대비 허용 편차 (3%p)")
    ap.add_argument("--market-tol", type=float, default=0.03, help="시장 커버리지 이상 구간 기준 (중앙값 대비 3%p)")
    ap.add_argument("--yoy-tol", type=float, default=0.03)
    ap.add_argument("--min-level", type=float, default=0.90, help="USABLE의 평소 커버리지 하한")
    ap.add_argument("--scale-amt", type=float, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    api = au.load_api(Path(a.api_dir))[0]       # 집계·해당없음 코드는 load 단계에서 제외됨
    base = au.load_base(Path(a.csv))
    months = sorted(api["진료년월"].unique())
    base = base[base["진료년월"].isin(months)]
    m = base.merge(api[["진료년월", "code", "api_amt", "rows"]], on=["진료년월", "code"], how="left")
    m["year"] = m["진료년월"].str[:4]

    ref = m[(m["base_amt"] >= a.min_base_amt) & (m["api_amt"] > 0)]
    scale = a.scale_amt or au.pick_scale(ref["api_amt"] / ref["base_amt"])
    m["eval"] = m["base_amt"] >= a.min_base_amt
    m["absent"] = m["eval"] & (m["api_amt"].isna() | (m["api_amt"] <= 0))
    m["cov"] = m["api_amt"] / (m["base_amt"] * scale)
    m.loc[~m["eval"] | m["absent"], "cov"] = np.nan

    live = m[m["eval"] & ~m["absent"]]
    baseline = live.groupby("code")["cov"].median()
    m["baseline"] = m["code"].map(baseline)

    mk = live.groupby("진료년월").apply(lambda g: g["api_amt"].sum() / (g["base_amt"].sum() * scale))
    mk0 = float(mk.median())
    anomaly = set(mk.index[mk < mk0 - a.market_tol])

    dev = m["cov"] - m["baseline"]
    state = np.where(~m["eval"], "TINY", np.where(m["absent"], "ABSENT",
            np.where(dev.abs() <= a.tol, "OK",
            np.where(dev < -a.tol, np.where(m["진료년월"].isin(anomaly), "DIP_MARKET", "DIP_CLASS"), "HIGH"))))
    m["state"] = state
    m["market_cov"] = m["진료년월"].map(mk)
    m[["진료년월", "code", "base_amt", "api_amt", "cov", "baseline", "market_cov", "state"]].to_csv(
        out / "usability_class_month.csv", index=False, encoding="utf-8-sig")

    # ---- 분류 × 연
    yrs = sorted(m["year"].unique())
    rows = []
    for (code, y), g in m.groupby(["code", "year"]):
        ev = g[g["eval"]]
        n_eval = len(ev)
        n_ok = int((g["state"] == "OK").sum())
        lv = g[g["eval"] & ~g["absent"]]
        cov_y = lv["api_amt"].sum() / (lv["base_amt"].sum() * scale) if len(lv) and lv["base_amt"].sum() > 0 else np.nan
        rows.append({"code": code, "year": y, "base_amt": g["base_amt"].sum(), "n_eval": n_eval, "n_ok": n_ok,
                     "n_absent": int(g["absent"].sum()), "cov_year": cov_y,
                     "usable_year": bool(n_eval > 0 and n_ok / n_eval >= 0.9)})
    cy = pd.DataFrame(rows).sort_values(["code", "year"])
    cy["prev_year_ok"] = cy.groupby("code")["year"].shift(1).astype(float) == cy["year"].astype(float) - 1
    cy["growth_base"] = cy.groupby("code")["base_amt"].pct_change()
    cy["cov_change"] = cy["cov_year"] / cy.groupby("code")["cov_year"].shift(1) - 1
    prev_usable = cy.groupby("code")["usable_year"].shift(1).fillna(False).astype(bool)
    good = cy["cov_change"].abs() <= a.yoy_tol
    rel = cy["cov_change"].abs() <= 0.2 * cy["growth_base"].abs()
    cy["yoy_attrib_ok"] = cy["prev_year_ok"] & cy["usable_year"] & prev_usable & (good | rel)
    cy.to_csv(out / "usability_class_year.csv", index=False, encoding="utf-8-sig")

    # ---- 분류 요약
    tot_base = float(m.loc[m["base_amt"] > 0, "base_amt"].sum())
    sm = []
    for code, g in m.groupby("code"):
        ev = g[g["eval"]]
        n = len(ev)
        if n == 0:
            continue
        ev_ex = ev[~ev["진료년월"].isin(anomaly)]
        ok_share = float((ev_ex["state"] == "OK").mean()) if len(ev_ex) else np.nan
        abs_share = float(ev["absent"].mean())
        lvl = baseline.get(code, np.nan)
        if abs_share >= 0.95:
            st = "ABSENT_ALL"
        elif pd.notna(ok_share) and ok_share >= 0.9 and pd.notna(lvl) and lvl >= a.min_level:
            st = "USABLE"
        else:
            st = "LIMITED"
        sm.append({"code": code, "status": st, "eval_months": n, "ok_share": ok_share, "absent_share": abs_share,
                   "baseline_cov": lvl, "share_of_base": float(g["base_amt"].sum() / tot_base),
                   **{f"n_{s}": int((ev["state"] == s).sum()) for s in ["OK", "ABSENT", "DIP_MARKET", "DIP_CLASS", "HIGH"]}})
    cs = pd.DataFrame(sm).sort_values("share_of_base", ascending=False)
    cs.to_csv(out / "usability_class_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 리포트
    st_n = cs["status"].value_counts()
    st_sh = cs.groupby("status")["share_of_base"].sum()
    ev_all = m[m["eval"]]
    amt_state = ev_all.groupby("state")["base_amt"].sum() / ev_all["base_amt"].sum()
    usable_by_year = ev_all.groupby("year").apply(lambda g: g.loc[g["state"] == "OK", "base_amt"].sum() / g["base_amt"].sum())
    ya = cy[cy["prev_year_ok"]].copy()
    prev_ok = ya["usable_year"] & cy.groupby("code")["usable_year"].shift(1).reindex(ya.index).fillna(False).astype(bool)
    ya["reason"] = np.where(ya["yoy_attrib_ok"], "사용 가능", np.where(~prev_ok, "연도 품질 미달", "커버리지 변화 초과"))
    L = ["# 상병 귀속 사용 가능성 분류 결과\n",
         f"- 대상 {len(months)}개월 ({months[0]} ~ {months[-1]}), 집계·해당없음 코드 제외, 평가 대상 분류 {cs['code'].nunique()}개 "
         f"(월 보유 금액 ≥ {a.min_base_amt:g}백만원인 달이 있는 분류, 전체 {m['code'].nunique()}개 중), 환산계수(추정) {scale:g}",
         f"- 시장 커버리지 중앙값 {mk0:.1%}, 이상 구간 기준: 중앙값 −{a.market_tol:.0%}p 미만",
         "", "## 시장 공통 이상 구간 (상병 커버리지가 시장 전체에서 낮아진 달)", ""]
    if anomaly:
        L += ["| 진료년월 | 시장 커버리지 |", "|---|---:|"] + [f"| {k} | {mk[k]:.1%} |" for k in sorted(anomaly)]
        L.append("\n- 이 구간의 분류별 하락은 DIP_MARKET으로 분류된다. 원인(예: 특정 청구 유형의 상병 집계 제외)은 확인이 필요하다.")
    else:
        L.append("- 없음")
    L += ["", "## 분류 상태", "", "| 상태 | 분류 수 | 보유 금액 비중 |", "|---|---:|---:|"]
    for s_ in ["USABLE", "LIMITED", "ABSENT_ALL"]:
        L.append(f"| {s_} | {int(st_n.get(s_, 0))} | {st_sh.get(s_, 0):.1%} |")
    L += [f"\n- **ABSENT_ALL 분류의 시장 비중: {st_sh.get('ABSENT_ALL', 0):.2%}** — 이 시장은 질환별 귀속 불가 (분류 총량 분석에는 사용)",
          "", "## 분류×월 상태 (보유 금액 가중)", "", "| 상태 | 금액 비중 |", "|---|---:|"]
    L += [f"| {s_} | {amt_state.get(s_, 0):.1%} |" for s_ in ["OK", "ABSENT", "DIP_MARKET", "DIP_CLASS", "HIGH"]]
    L += ["", "## 연도별 귀속 신뢰 금액 비중 (OK 월 금액 ÷ 전체 금액)", "", "| 연도 | 비중 |", "|---|---:|"]
    L += [f"| {y} | {v:.1%} |" for y, v in usable_by_year.items()]
    L += ["", "## YoY 귀속 사용 가능 (연속 연도 쌍, 분류 수 / 금액)", "",
          f"- 대상 쌍 {len(ya)}개 중 귀속 사용 가능 {int(ya['yoy_attrib_ok'].sum())}개 ({ya['yoy_attrib_ok'].mean():.1%}), "
          f"금액 가중 {ya.loc[ya['yoy_attrib_ok'], 'base_amt'].sum() / ya['base_amt'].sum():.1%}",
          "", "### YoY 쌍별 (금액 가중)", "", "| 쌍 | 사용 가능 | 연도 품질 미달 | 커버리지 변화 초과 |", "|---|---:|---:|---:|"]
    for y_, g_ in ya.groupby("year"):
        w = g_["base_amt"].sum()
        sh = g_.groupby("reason")["base_amt"].sum() / w
        L.append(f"| {int(y_) - 1}→{y_} | {sh.get('사용 가능', 0):.1%} | {sh.get('연도 품질 미달', 0):.1%} | {sh.get('커버리지 변화 초과', 0):.1%} |")
    L += ["", "## 비 USABLE 분류 (보유 금액 비중 상위 20)", "",
          "| 분류 | 상태 | 금액 비중 | 평소 커버리지 | OK 월(이상 구간 제외) | 부재 월 | 시장공통 하락 | 개별 하락 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in cs[cs["status"] != "USABLE"].head(20).iterrows():
        bc = "-" if pd.isna(r["baseline_cov"]) else f"{r['baseline_cov']:.1%}"
        L.append(f"| {r['code']} | {r['status']} | {r['share_of_base']:.2%} | {bc} | {r['ok_share']:.0%} | {r['absent_share']:.0%} | {int(r['n_DIP_MARKET'])} | {int(r['n_DIP_CLASS'])} |")
    L += ["", "## 사용 규칙",
          "- 분류 총량·성장률은 항상 보유 분류 데이터를 쓴다.",
          "- 상병 귀속: OK 월과 yoy_attrib_ok 쌍만 쓴다. 상병 금액은 배율 보정하지 않고 '관측 상병 금액 + 미귀속 잔여항'으로 둔다.",
          "- ABSENT_ALL·LIMITED 분류는 상병 경로 대신 성분(공단 파일) 경로로 진단한다.",
          "- 임계값(tol, yoy_tol, min_level)은 운영 초기값이며 결과 분포를 보고 조정한다."]
    (out / "usability_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))
    return cs, cy, m


if __name__ == "__main__":
    main()
