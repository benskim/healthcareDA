
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
screen.py — 구조적 성장 약효분류군 스크리닝 (screen_rule.md v2 구현)

입력 : meft_*.csv  (약효분류코드, 약효분류명, 진료년월, 수량_천개, 사용금액_백만원)
수동 : break_table.csv (확정 break), tags.csv (외부 요인 태깅)
출력 : output/screen_result.csv, report_1차.md, break_candidates.csv, tags_template.csv

실행 : python src/screen.py
필요 : pandas, numpy
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------
EXCLUDE_CODES = {110, 140, 230, 250, 310, 398, 999}
WINDOWS = [3, 5, 7, 10, 12]
REQUIRED_ALL_WINDOWS = [3, 5, 7]     # 모두 3중 통과 필수
AVAIL_PASS_RATIO = 0.8               # 평가 가능 창 중 통과 비율

MIN_AMOUNT_12M = 5_000.0             # 백만원
VOLUME_SHARE_MIN = 0.5
CPI_CAGR = 0.02                      # ★ 실제 CPI 장기 CAGR로 교체할 것
PRICE_MARGIN = 0.02                  # 단가 CAGR < CPI + 2%p
EXCESS_MARGIN = 0.01                 # 초과성장 최소 마진 (벤치마크 대비 +1%p, 조정 가능)
DECEL_RATIO = 0.5                    # 감속 체크: CAGR_3Y >= 0.5 × CAGR_10Y
GENERIC_PRICE = -0.03                # 제네릭 대체 의심: 5Y 단가 CAGR < -3% & 수량 증가
R2_MIN, CV_MAX, MDD_MIN, NEG_YOY_MAX, SPIKE_MAX = 0.70, 0.30, -0.15, 0.20, 0.50
CONTRIB_TOP_PCT = 0.75               # contrib_3Y 상위 25%
ENDPOINT_GAP = 0.03                  # |g - CAGR| 끝점 의존 플래그 (5Y 이상)
STAB_MAX_MONTHS = 121                # 안정성 평가 구간: 최근 10Y(121개월 TTM)
STAB_MIN_MONTHS = 37

# break 후보 탐지
BREAK_MOM = 0.5
BREAK_MIN_MONTHLY = MIN_AMOUNT_12M / 12
CUSUM_K, CUSUM_H = 0.5, 6.0
COVID_WINDOW = (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-12-01"))
COVID_TTM_RANGE = (pd.Timestamp("2020-02-01"), pd.Timestamp("2021-11-01"))  # 이 구간에 끝나는 TTM은 COVID 영향

STRUCTURAL_TAGS = {"적응증확대", "고령화만성질환", "급여확대"}
ONEOFF_TAGS = {"COVID", "약가정책", "특허제네릭", "공급이슈"}

WEIGHTS = {"excess": 0.25, "multi": 0.20, "vol": 0.20, "stab": 0.15, "contrib": 0.10, "tags": 0.10}

# 최초 실행 시 생성되는 break 템플릿 (기존 수동 지정분, 검증 필요)
DEFAULT_BREAKS = [
    (111, "2017-01", "data_def", "cut", "", "전신마취제 수량 급감 (검증 필요)"),
    (116, "2017-01", "data_def", "cut", "", "진훈제 재분류 (검증 필요)"),
    (235, "2017-02", "data_def", "cut", "", "최토제·진토제 (검증 필요)"),
    (332, "2016-04", "data_def", "cut", "", "지혈제 이상치 (검증 필요)"),
    (395, "2016-04", "data_def", "cut", "", "효소제제 수량 (검증 필요)"),
    (634, "2017-01", "data_def", "cut", "", "혈액제제류 (검증 필요)"),
]


# ----------------------------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------------------------
def parse_ym(s: pd.Series) -> pd.Series:
    digits = s.astype(str).str.replace(r"[^0-9]", "", regex=True).str[:6]
    return pd.to_datetime(digits + "01", format="%Y%m%d", errors="coerce")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "약효분류코드": "code", "약효분류명": "name", "진료년월": "ym",
        "수량_천개": "qty", "사용금액_백만원": "amount",
    })
    missing = [c for c in ["code", "name", "ym", "qty", "amount"] if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    for c in ["code", "qty", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["code"])
    df["code"] = df["code"].astype(int)
    df = df[~df["code"].isin(EXCLUDE_CODES)]

    df["date"] = parse_ym(df["ym"])
    n_bad = df["date"].isna().sum()
    if n_bad:
        print(f"    [warn] 날짜 파싱 실패 {n_bad}행 제외")
    df = df.dropna(subset=["date", "qty", "amount"])
    df["name"] = df["name"].astype(str).str.strip()

    n_dup = df.duplicated(["code", "date"]).sum()
    if n_dup:
        print(f"    [warn] (code,date) 중복 {n_dup}행 → 첫 행 유지")
    df = df.sort_values(["code", "date"]).drop_duplicates(["code", "date"])
    return df[["code", "name", "date", "qty", "amount"]]


def resolve_csv_path(csv_arg: str | None) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    if csv_arg:
        path = Path(csv_arg).expanduser()
        candidates = [path] if path.is_absolute() else [path, project_root / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"CSV file not found: {csv_arg}")

    matches = sorted((project_root / "data").glob("meft*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No meft*.csv found in {project_root / 'data'}; pass its path with --csv."
        )
    if len(matches) > 1:
        options = ", ".join(str(path) for path in matches)
        raise ValueError(f"More than one meft*.csv found; choose one with --csv: {options}")
    return matches[0]


# ----------------------------------------------------------------------
# 2. break 처리
# ----------------------------------------------------------------------
def load_break_table(path: Path) -> pd.DataFrame:
    cols = ["code", "date", "type", "action", "partner_code", "note"]
    if not path.exists():
        pd.DataFrame(DEFAULT_BREAKS, columns=cols).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"    [info] {path} 없음 → 기존 수동 break 6건으로 템플릿 생성 (검증 후 사용)")
    bt = pd.read_csv(path, encoding="utf-8-sig", dtype={"partner_code": str}).fillna("")
    bt["code"] = bt["code"].astype(int)
    bt["date"] = parse_ym(bt["date"]) if bt["date"].astype(str).str.len().gt(0).any() else pd.NaT
    return bt


def build_panels(df: pd.DataFrame, bt: pd.DataFrame):
    """merge → 월 reindex → cut 순으로 처리해 (amount, qty) 월별 패널 반환."""
    names = df.sort_values("date").groupby("code")["name"].last()

    merges = bt[bt["action"] == "merge"]
    for _, r in merges.iterrows():
        if str(r["partner_code"]).strip():
            df.loc[df["code"] == int(r["partner_code"]), "code"] = r["code"]
    if len(merges):
        df = df.groupby(["code", "date"], as_index=False)[["qty", "amount"]].sum()
        print(f"    [break] merge {len(merges)}건 적용")

    amt = df.pivot(index="date", columns="code", values="amount")
    qty = df.pivot(index="date", columns="code", values="qty")
    full = pd.date_range(amt.index.min(), amt.index.max(), freq="MS")
    amt, qty = amt.reindex(full), qty.reindex(full)

    cuts = bt[bt["action"] == "cut"]
    n_cut = 0
    for _, r in cuts.iterrows():
        if r["code"] in amt.columns and pd.notna(r["date"]):
            amt.loc[amt.index < r["date"], r["code"]] = np.nan
            qty.loc[qty.index < r["date"], r["code"]] = np.nan
            n_cut += 1
    if n_cut:
        print(f"    [break] cut {n_cut}건 적용 (컷 → rolling 순서)")

    # 활동 구간 내 결측월 보고
    gaps = {}
    for c in amt.columns:
        v = amt[c].dropna()
        if len(v):
            span = amt[c].loc[v.index.min():v.index.max()]
            g = int(span.isna().sum())
            if g:
                gaps[c] = g
    if gaps:
        print(f"    [warn] 활동 구간 내 결측월 있는 분류 {len(gaps)}개 (해당 TTM은 NaN): "
              f"{dict(list(gaps.items())[:5])}{' ...' if len(gaps) > 5 else ''}")
    return amt, qty, names


def detect_break_candidates(amt: pd.DataFrame, qty: pd.DataFrame) -> pd.DataFrame:
    """자동 탐지: MoM 급변 + CUSUM. 후보 생성용이며 적용하지 않는다."""
    rows = {}
    for code in amt.columns:
        a, q = amt[code], qty[code]
        p = a / q.where(q > 0)
        mom_a = a.pct_change(fill_method=None)
        mom_q = q.pct_change(fill_method=None)
        mom_p = p.pct_change(fill_method=None)
        prior = a.shift(1).rolling(12, min_periods=6).mean()

        flag_mom = (mom_a.abs() > BREAK_MOM) & (prior >= BREAK_MIN_MONTHLY)

        logd = np.log(a.where(a > 0)).diff()
        sd = logd.std()
        flag_cusum = pd.Series(False, index=a.index)
        if sd and not np.isnan(sd) and sd > 0:
            z = ((logd - logd.mean()) / sd).values
            sp = sn = 0.0
            for i, zi in enumerate(z):
                if np.isnan(zi):
                    continue
                sp = max(0.0, sp + zi - CUSUM_K)
                sn = max(0.0, sn - zi - CUSUM_K)
                if sp > CUSUM_H or sn > CUSUM_H:
                    flag_cusum.iloc[i] = True
                    sp = sn = 0.0

        for d in a.index[flag_mom | flag_cusum]:
            method = "+".join(m for m, f in [("MoM", flag_mom[d]), ("CUSUM", flag_cusum[d])] if f)
            rows[(code, d)] = {
                "code": code, "date": d.strftime("%Y-%m"), "method": method,
                "mom_amount": mom_a[d], "mom_qty": mom_q[d], "mom_price": mom_p[d],
                "qty_price_jump": bool(abs(mom_q[d]) > BREAK_MOM or abs(mom_p[d]) > BREAK_MOM),
                "covid_window": bool(COVID_WINDOW[0] <= d <= COVID_WINDOW[1]),
            }
    out = pd.DataFrame(list(rows.values()))
    return out.sort_values(["code", "date"]) if len(out) else out


# ----------------------------------------------------------------------
# 3. 창별 지표 / benchmark
# ----------------------------------------------------------------------
def ttm(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.rolling(12, min_periods=12).sum()


def _cagr(end: pd.Series, start: pd.Series, years: int) -> pd.Series:
    ok = (end > 0) & (start > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (end / start) ** (1 / years) - 1
    return out.where(ok)


def window_metrics(ta: pd.DataFrame, tq: pd.DataFrame, w: int) -> pd.DataFrame:
    """TTM 기준 w년 trailing CAGR(금액/수량/단가) + log-linear 연환산 기울기 g."""
    k = 12 * w
    idx = ta.columns
    nan = pd.Series(np.nan, index=idx)
    if len(ta) < k + 1:
        return pd.DataFrame({c: nan for c in ["a_end", "a_start", "cagr", "qcagr", "pcagr", "g"]})

    a_end, a_st = ta.iloc[-1], ta.iloc[-(k + 1)]
    q_end, q_st = tq.iloc[-1], tq.iloc[-(k + 1)]
    p_end = a_end / q_end.where(q_end > 0)
    p_st = a_st / q_st.where(q_st > 0)

    Y = np.log(ta.iloc[-(k + 1):].where(ta.iloc[-(k + 1):] > 0)).values
    col_ok = ~np.isnan(Y).any(axis=0)
    x = np.arange(k + 1, dtype=float)
    xc = x - x.mean()
    with np.errstate(invalid="ignore"):
        slope = (xc @ (Y - Y.mean(axis=0))) / (xc @ xc)
    g = pd.Series(np.where(col_ok, np.expm1(slope * 12), np.nan), index=idx)

    return pd.DataFrame({
        "a_end": a_end, "a_start": a_st,
        "cagr": _cagr(a_end, a_st, w),
        "qcagr": _cagr(q_end, q_st, w),
        "pcagr": _cagr(p_end, p_st, w),
        "g": g,
    })


def loo_median(values: pd.Series, groups: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index)
    for _, members in groups.groupby(groups).groups.items():
        members = list(members)
        for c in members:
            peers = values.loc[[m for m in members if m != c]].dropna()
            if len(peers):
                out[c] = peers.median()
    return out


def loo_agg_cagr(end, start, valid, groups, w) -> pd.Series:
    e, s = end.where(valid, 0.0), start.where(valid, 0.0)
    le = e.groupby(groups).transform("sum") - e
    ls = s.groupby(groups).transform("sum") - s
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (le / ls) ** (1 / w) - 1
    return c.where((le > 0) & (ls > 0))


# ----------------------------------------------------------------------
# 4. 분류별 안정성 (최근 10Y TTM)
# ----------------------------------------------------------------------
def stability(arr: np.ndarray) -> dict:
    nan = dict(R2=np.nan, g_stab=np.nan, CV=np.nan, MDD=np.nan, neg_yoy=np.nan,
               spike_ratio=np.nan, hist_months=0)
    ok = np.isfinite(arr) & (arr > 0)
    if len(arr) == 0 or not ok[-1]:
        return nan
    bad = np.where(~ok)[0]
    tail = arr[(bad[-1] + 1) if len(bad) else 0:]
    hist = len(tail)
    if hist > STAB_MAX_MONTHS:
        tail = tail[-STAB_MAX_MONTHS:]
    if len(tail) < STAB_MIN_MONTHS:
        return {**nan, "hist_months": hist}

    y, x = np.log(tail), np.arange(len(tail), dtype=float)
    slope, icpt = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (slope * x + icpt)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    yoy = tail[12:] / tail[:-12] - 1
    m = yoy.mean()
    cv = yoy.std(ddof=1) / m if m > 0 else np.nan     # 평균 ≤ 0 → 판정 불가
    neg = float((yoy < 0).mean())

    peak = np.maximum.accumulate(tail)
    mdd = float(((tail - peak) / peak).min())

    anchors = tail[::-1][::12][::-1]                   # 최근 시점에서 12개월씩 역산
    spike = np.nan
    if len(anchors) >= 3:
        total = anchors[-1] - anchors[0]
        if total > 0:
            spike = float(np.diff(anchors).max() / total)

    return dict(R2=float(r2), g_stab=float(np.expm1(slope * 12)), CV=float(cv) if not np.isnan(cv) else np.nan,
                MDD=mdd, neg_yoy=neg, spike_ratio=spike, hist_months=hist)


# ----------------------------------------------------------------------
# 5. 요약 테이블 구성
# ----------------------------------------------------------------------
def build_summary(amt, qty, names) -> pd.DataFrame:
    ta, tq = ttm(amt), ttm(qty)
    codes = amt.columns
    s = pd.DataFrame(index=pd.Index(codes, name="code"))
    s["name"] = names.reindex(codes).values
    s["big"] = (s.index // 100 * 100).values
    s["mid"] = (s.index // 10 * 10).values
    s["amount_12m"] = ta.iloc[-1].values
    s["qty_12m"] = tq.iloc[-1].values
    s["unit_price"] = (s["amount_12m"] / s["qty_12m"].where(s["qty_12m"] > 0)) * 1000  # 원/개

    mid, big = s["mid"], s["big"]
    mkt = pd.Series(0, index=s.index)

    n_fb = 0
    s["covid_base"] = False
    for w in WINDOWS:
        m = window_metrics(ta, tq, w)
        m.index = s.index
        valid = m["cagr"].notna()
        s[f"CAGR_{w}Y"], s[f"qty_CAGR_{w}Y"], s[f"price_CAGR_{w}Y"], s[f"g_{w}Y"] = (
            m["cagr"], m["qcagr"], m["pcagr"], m["g"])

        b1 = loo_median(m["cagr"], mid)
        b2 = loo_agg_cagr(m["a_end"], m["a_start"], valid, big, w)
        b3 = loo_agg_cagr(m["a_end"], m["a_start"], valid, mkt, w)
        fb = b1.isna() & valid & b2.notna()
        n_fb += int(fb.sum())
        b1 = b1.where(~fb, b2)
        s[f"b1_fallback_{w}Y"] = fb
        for k, b in enumerate([b1, b2, b3], 1):
            s[f"b{k}_{w}Y"] = b
            s[f"excess_{k}_{w}Y"] = m["cagr"] - b

        ev = valid & s[[f"excess_{k}_{w}Y" for k in (1, 2, 3)]].notna().all(axis=1)
        s[f"ev_{w}Y"] = ev
        s[f"pass_{w}Y"] = ev & (s[[f"excess_{k}_{w}Y" for k in (1, 2, 3)]] > EXCESS_MARGIN).all(axis=1)

        # COVID 기저효과: 시작 TTM이 COVID 구간이고 전년 대비 하락한 경우
        kk = 12 * w
        if len(ta) >= kk + 13 and COVID_TTM_RANGE[0] <= ta.index[-(kk + 1)] <= COVID_TTM_RANGE[1]:
            prev = pd.Series(ta.iloc[-(kk + 13)].values, index=s.index)
            s["covid_base"] |= ((m["a_start"] / prev.where(prev > 0) - 1) < 0).fillna(False)

        # 기울기 기반 3중 초과 (참고용 근사: g의 중앙값 대비)
        gm = m["g"]
        gb2 = loo_median(gm, big)
        gb1 = loo_median(gm, mid).fillna(gb2)
        gb3 = loo_median(gm, mkt)
        s[f"pass_g_{w}Y"] = ev & gm.notna() & (gm > gb1) & (gm > gb2) & (gm > gb3)
        s[f"endpoint_gap_{w}Y"] = (gm - m["cagr"]).abs()
    if n_fb:
        print(f"    [info] benchmark_1 peer 없음 → benchmark_2로 대체: {n_fb}건(창 합산)")

    # ---- [2] 다중 창 판정
    ev = s[[f"ev_{w}Y" for w in WINDOWS]]
    pas = s[[f"pass_{w}Y" for w in WINDOWS]]
    s["n_avail"] = ev.sum(axis=1)
    s["n_pass"] = pas.sum(axis=1)
    need = np.ceil(AVAIL_PASS_RATIO * s["n_avail"])

    req_ok = s[[f"pass_{w}Y" for w in REQUIRED_ALL_WINDOWS]].all(axis=1)
    elig = s[[f"ev_{w}Y" for w in [3, 5, 7, 10]]].all(axis=1)
    trend_ok = pd.Series(True, index=s.index)
    for w in (10, 12):
        trend_ok &= (~s[f"ev_{w}Y"]) | ((s[f"CAGR_{w}Y"] > 0) & (s[f"g_{w}Y"] > 0))
    size_ok = s["amount_12m"] >= MIN_AMOUNT_12M

    s["pass_multi"] = elig & req_ok & (s["n_pass"] >= need) & trend_ok & size_ok
    s["watchlist"] = (~elig) & s[[f"ev_{w}Y" for w in REQUIRED_ALL_WINDOWS]].all(axis=1) & req_ok & size_ok

    later = [w for w in WINDOWS if w >= 5]
    s["robust"] = s["pass_multi"] & pd.concat(
        [(~s[f"ev_{w}Y"]) | s[f"pass_g_{w}Y"] for w in later], axis=1).all(axis=1)
    s["endpoint_dep"] = pd.concat(
        [s[f"ev_{w}Y"] & (s[f"endpoint_gap_{w}Y"] > ENDPOINT_GAP) for w in later], axis=1).any(axis=1)

    # ---- [3] 수량 vs 가격 (10Y)
    a, q, p = s["CAGR_10Y"], s["qty_CAGR_10Y"], s["price_CAGR_10Y"]
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.log1p(q) / np.log1p(a)
    s["volume_share_raw"] = raw.where(a > 0)
    s["volume_share"] = s["volume_share_raw"].clip(0, 1)
    s["vol_ok"] = (q > 0) & (a > 0) & (s["volume_share"] >= VOLUME_SHARE_MIN) & (p < CPI_CAGR + PRICE_MARGIN)

    s["flag_generic"] = ((s["price_CAGR_5Y"] < GENERIC_PRICE) & (s["qty_CAGR_5Y"] > 0)).fillna(False)

    # 감속 체크: 최근 3Y CAGR이 10Y CAGR 대비 일정 비율 이상
    s["decel_ratio"] = (s["CAGR_3Y"] / s["CAGR_10Y"]).where((s["CAGR_3Y"] > 0) & (s["CAGR_10Y"] > 0))
    s["c_decel"] = s["decel_ratio"] >= DECEL_RATIO

    # ---- [4] 최근 3Y 기여도 + Matrix
    m3 = window_metrics(ta, tq, 3)
    m3.index = s.index
    delta = (m3["a_end"] - m3["a_start"]).where(m3["cagr"].notna())
    total = delta.sum()
    s["contrib_3Y"] = delta / total if total > 0 else np.nan
    s["contrib_pct"] = s["contrib_3Y"].rank(pct=True)
    s["contrib_top"] = (s["contrib_3Y"] > 0) & (s["contrib_pct"] >= CONTRIB_TOP_PCT)

    pool = s["ev_3Y"]
    size_med = s.loc[pool, "amount_12m"].median()
    grow_med = s.loc[pool, "excess_3_3Y"].median()
    big_sz = s["amount_12m"] >= size_med
    hi_gr = s["excess_3_3Y"] >= grow_med
    s["matrix"] = np.where(~pool, "", np.where(big_sz & hi_gr, "A", np.where(~big_sz & hi_gr, "B",
                           np.where(big_sz & ~hi_gr, "C", "D"))))

    # ---- [5] 궤적 안정성
    st = {c: stability(ta[c].values) for c in codes}
    for key, col in [("R2", "R2"), ("g_stab", "g_stab"), ("CV", "CV"), ("MDD", "MDD"),
                     ("neg_yoy", "neg_YoY"), ("spike_ratio", "spike_ratio"), ("hist_months", "hist_months")]:
        s[col] = [st[c][key] for c in codes]
    s["c_stab"] = ((s["R2"] > R2_MIN) & (s["g_stab"] > 0) & (s["CV"] < CV_MAX)
                   & (s["neg_YoY"] < NEG_YOY_MAX))            # 하드 컷
    s["flag_mdd"] = ~(s["MDD"] > MDD_MIN)                       # 감점 + 주의 (하드 컷 아님)
    s["flag_spike"] = ~(s["spike_ratio"] < SPIKE_MAX)

    # ---- 판정 (태그 제외)
    core_nonvol = s["pass_multi"] & s["pass_10Y"] & s["c_stab"] & s["c_decel"] & s["contrib_top"]
    s["stage1_pass"] = core_nonvol & s["vol_ok"]
    s["value_track"] = core_nonvol & ~s["vol_ok"] & (a > 0)
    s["track"] = np.where(s["stage1_pass"], "물량형", np.where(s["value_track"], "가치형", ""))
    s["notes"] = [";".join(n for n, f in [("MDD", r.flag_mdd), ("spike", r.flag_spike), ("끝점의존", r.endpoint_dep),
                                        ("제네릭의심", r.flag_generic), ("COVID기저", r.covid_base)] if f)
                  for r in s.itertuples()]
    ev357 = s[[f"ev_{w}Y" for w in REQUIRED_ALL_WINDOWS]].all(axis=1)
    s["status"] = np.where(elig, "평가가능", np.where(s["watchlist"], "이력부족_watchlist",
                           np.where(ev357, "10Y평가불가", "평가불가")))
    return s


# ----------------------------------------------------------------------
# 6. 태그 + 점수
# ----------------------------------------------------------------------
def apply_tags(s: pd.DataFrame, path: Path) -> pd.DataFrame:
    s = s.copy()
    s["tags"], s["tags_n"], s["oneoff_ok"], s["tags_status"] = "", 0, "", "pending"
    if not path.exists():
        return s
    t = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    t["code"] = t["code"].astype(int)
    known = STRUCTURAL_TAGS | ONEOFF_TAGS
    for _, r in t.iterrows():
        if r["code"] not in s.index:
            continue
        tags = {x.strip() for x in r["tags"].replace(",", ";").split(";") if x.strip()}
        unk = tags - known
        if unk:
            print(f"    [warn] code {r['code']} 알 수 없는 태그 무시: {unk}")
        s.loc[r["code"], "tags"] = ";".join(sorted(tags & known))
        s.loc[r["code"], "tags_n"] = len(tags & STRUCTURAL_TAGS)
        s.loc[r["code"], "oneoff_ok"] = r.get("oneoff_ok", "").strip().upper()
        s.loc[r["code"], "tags_status"] = "tagged"
    return s


def add_score(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    pr = lambda x: x.rank(pct=True)
    excess10 = s[[f"excess_{k}_10Y" for k in (1, 2, 3)]].min(axis=1, skipna=False)
    stab = pd.concat([pr(s["R2"]), pr(-s["CV"]), pr(s["MDD"]), pr(-s["neg_YoY"]), pr(-s["spike_ratio"])],
                     axis=1).mean(axis=1, skipna=False)
    comp = pd.DataFrame({
        "excess": pr(excess10),
        "multi": s["n_pass"] / s["n_avail"].replace(0, np.nan),
        "vol": s["volume_share"],
        "stab": stab,
        "contrib": pr(s["contrib_3Y"]),
    })
    w = WEIGHTS
    pre = sum(w[k] * comp[k] for k in comp.columns)          # NaN 하나라도 있으면 NaN
    s["score_pre"] = pre / (1 - w["tags"]) * 100
    tag_score = s["tags_n"].clip(upper=3) / 3
    full = (pre + w["tags"] * tag_score) * 100
    s["score"] = full.where(s["tags_status"] == "tagged")
    return s


def apply_final(s: pd.DataFrame, tags_loaded: bool) -> pd.DataFrame:
    s = s.copy()
    tag_ok = (s["tags_n"] >= 2) & (s["oneoff_ok"] == "Y")
    s["final_pass"] = (s["stage1_pass"] & tag_ok) if tags_loaded else False
    s["final_value"] = (s["value_track"] & tag_ok) if tags_loaded else False
    return s


# ----------------------------------------------------------------------
# 7. 리포트
# ----------------------------------------------------------------------
def fmt(x, d=1, pct=False):
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{d}f}%" if pct else f"{x:.{d}f}"


def table(rows: pd.DataFrame, sortcol: str) -> list[str]:
    out = ["| 순위 | 코드 | 분류명 | 10Y CAGR | g_10Y | vol_share | R² | CV | MDD | contrib_3Y | matrix | robust | 주의 | score |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---|---:|"]
    for i, (c, r) in enumerate(rows.sort_values(sortcol, ascending=False).iterrows(), 1):
        out.append(f"| {i} | {c} | {r['name']} | {fmt(r['CAGR_10Y'], pct=True)} | {fmt(r['g_10Y'], pct=True)} | "
                   f"{fmt(r['volume_share'], 2)} | {fmt(r['R2'], 2)} | {fmt(r['CV'], 2)} | {fmt(r['MDD'], pct=True)} | "
                   f"{fmt(r['contrib_3Y'], pct=True)} | {r['matrix']} | {'Y' if r['robust'] else ''} | "
                   f"{r['notes']} | {fmt(r[sortcol], 0)} |")
    return out


def make_report(s: pd.DataFrame, path: Path, tags_loaded: bool):
    n_all = len(s)
    elig = s[[f"ev_{w}Y" for w in [3, 5, 7, 10]]].all(axis=1)
    f_multi = s["pass_multi"]
    f_vol = f_multi & s["vol_ok"]
    f_stab = f_vol & s["c_stab"]
    f_dec = f_stab & s["c_decel"]
    f_con = f_dec & s["contrib_top"]
    funnel = [("전체 분류", n_all), ("3·5·7·10Y 평가 가능", int(elig.sum())),
              ("다중 창 통과 ([2])", int(f_multi.sum())), ("+ 물량형 ([3])", int(f_vol.sum())),
              ("+ 안정성 ([5])", int(f_stab.sum())), ("+ 감속 없음", int(f_dec.sum())),
              ("+ 기여도 상위 → 1차(기계적)", int(f_con.sum()))]
    if tags_loaded:
        funnel.append(("+ 태그/일회성 → 최종", int(s["final_pass"].sum())))

    L = ["# 구조적 성장 약효분류군 스크리닝 결과\n", "## Funnel (통과 수는 결과 보고용, 목표 아님)\n",
         "| 단계 | 수 |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in funnel]

    stage1 = s[s["stage1_pass"]]
    L += [f"\n## 1차 후보 — 물량형 (n={len(stage1)})\n"]
    L += table(stage1, "score_pre") if len(stage1) else ["(없음)"]

    if tags_loaded:
        fin = s[s["final_pass"]]
        L += [f"\n## 최종 후보 — 태그 반영 (n={len(fin)})\n"]
        L += table(fin, "score") if len(fin) else ["(없음)"]
    val = s[s["value_track"]]
    L += [f"\n## 가치형 트랙 (물량형 조건만 불통과, n={len(val)})\n"]
    L += table(val, "score_pre") if len(val) else ["(없음)"]

    wl = s[s["watchlist"]]
    L += [f"\n## 이력 부족 watchlist (3·5·7Y 통과, 10Y 평가불가, n={len(wl)})\n"]
    L += [f"- {c} {r['name']} (이력 {int(r['hist_months'])}개월, 7Y CAGR {fmt(r['CAGR_7Y'], pct=True)})"
          for c, r in wl.iterrows()] or ["(없음)"]

    L += ["\n## 참고", f"- CPI 가정값: {CPI_CAGR:.1%} (설정값 확인 필요)",
          "- robust: 5Y 이상 모든 창에서 CAGR·기울기 기준 모두 3중 통과",
          "- 주의 플래그: MDD(≤-15%) / spike(≥50%) / 끝점의존(|g−CAGR|>3%p) / 제네릭의심(5Y 단가↓·수량↑) / COVID기저",
          "- 태깅 대상은 tags_template.csv 참고"]
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Input CSV (default: the single data/meft*.csv file)")
    ap.add_argument("--breaks", default="break_table.csv")
    ap.add_argument("--tags", default="tags.csv")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1] 데이터 로드")
    csv_path = resolve_csv_path(args.csv)
    print(f"    CSV: {csv_path}")
    df = load_data(csv_path)
    print(f"    rows={len(df)}, codes={df['code'].nunique()}, {df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}")

    print("[2] break 처리")
    bt = load_break_table(Path(args.breaks))
    amt_raw = df.pivot(index="date", columns="code", values="amount")
    qty_raw = df.pivot(index="date", columns="code", values="qty")
    cand = detect_break_candidates(amt_raw, qty_raw)
    cand.to_csv(out / "break_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"    자동 탐지 후보 {len(cand)}건 → break_candidates.csv (수동 검증 후 break_table.csv 반영)")
    amt, qty, names = build_panels(df, bt)

    print("[3] 지표·benchmark·판정")
    s = build_summary(amt, qty, names)

    print("[4] 태그·점수")
    tags_loaded = Path(args.tags).exists()
    s = apply_tags(s, Path(args.tags))
    s = add_score(s)
    s = apply_final(s, tags_loaded)

    first = ["name", "big", "mid", "status", "amount_12m", "qty_12m", "unit_price"]
    s = s[first + [c for c in s.columns if c not in first]]
    s.to_csv(out / "screen_result.csv", encoding="utf-8-sig")

    tmpl = s[s["stage1_pass"] | s["value_track"]][["name", "track"]].copy()
    tmpl["tags"], tmpl["oneoff_ok"] = "", ""
    tmpl.to_csv(out / "tags_template.csv", encoding="utf-8-sig")

    make_report(s, out / "report_1차.md", tags_loaded)
    print(f"[done] 1차(물량형) {int(s['stage1_pass'].sum())}개, 가치형 {int(s['value_track'].sum())}개, "
          f"watchlist {int(s['watchlist'].sum())}개 → {out}/")


if __name__ == "__main__":
    main()
