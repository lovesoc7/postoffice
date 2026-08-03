"""
예측모델 아웃풋 -> 배차(허브경유 vs 직송) 최적화 연결 모듈
================================================================
lgbm/lstm 예측 결과(노선별·일별 예측중량)를 받아서,
날짜별로 "같은 목적지로 가는 여러 출발지 물량"을 묶어 evaluate_consolidation을
돌리고, 그 결과를 요약한다.

핵심 산출물: "예측 기반 사전 배차(우리 모델)" vs "전부 직송했다고 가정한 baseline"
비용을 비교해서 전체 절감률(%)을 낸다 — 이게 제안서의 핵심 before/after 근거.
"""

import pandas as pd
from dataclasses import asdict

import dispatch


def run_daily_dispatch_optimization(
    df_pred: pd.DataFrame,
    date_col: str = "접수일자",
    origin_col: str = "접수지역",
    dest_col: str = "배달지역",
    weight_col: str = "예측중량_kg",
    min_weight_kg: float = 1.0,  # 이보다 적으면 그날 그 노선은 물량 없음으로 취급
) -> tuple[pd.DataFrame, dict]:
    """
    날짜 x 목적지 단위로 그룹핑해서 [직송] vs [허브경유] 비용을 비교하고,
    최종 선택된 비용을 기록한 요약 테이블을 반환한다.
    """
    df = df_pred[df_pred[weight_col] >= min_weight_kg].copy()

    records = []
    merge_histories = {}
    grouped = df.groupby([date_col, dest_col])

    for (date, dest), g in grouped:
        if dest not in dispatch.REGION_COORDS:
            continue  # 좌표 없는 지역은 스킵 (실제 구현시 좌표 보강 필요)

        origin_weights = {}
        for _, row in g.iterrows():
            o = row[origin_col]
            if o not in dispatch.REGION_COORDS:
                continue
            origin_weights[o] = origin_weights.get(o, 0.0) + row[weight_col]

        if not origin_weights:
            continue

        # result = dispatch.evaluate_consolidation(dest, origin_weights)
        result = dispatch.evaluate_greedy_consolidation(dest, origin_weights)
        

        records.append({
            "날짜": date,
            "목적지": dest,
            "출발지_수": len(origin_weights),
            "총물량_kg": sum(origin_weights.values()),
            "직송_비용": result.direct_cost,
            "허브경유_최적비용": result.best_hub_cost,
            "선택된_허브": result.chosen_hub,
            "최종_결정": result.decision,
            "최종_비용": result.best_hub_cost if result.decision == "hub" else result.direct_cost,
            "절감액": result.savings if result.decision == "hub" else 0,
        })
        if merge_histories.get(str(date))==None:
            merge_histories[str(date)] = {}
        merge_histories[str(date)][dest] = result.detail['merge_history']

    return pd.DataFrame.from_records(records), merge_histories


def summarize_savings(summary_df: pd.DataFrame) -> dict:
    """
    전체 기간 합계: [우리 모델(예측+허브경유 판단 적용)] vs
    [baseline: 전부 직송했다고 가정] 총비용 비교.
    """
    total_direct_baseline = summary_df["직송_비용"].sum()  # 모든 건을 직송했다면의 비용
    total_optimized = summary_df["최종_비용"].sum()          # 우리 모델이 실제로 선택한 비용
    total_savings = total_direct_baseline - total_optimized
    savings_pct = (total_savings / total_direct_baseline * 100) if total_direct_baseline > 0 else 0

    n_hub_chosen = (summary_df["최종_결정"] == "hub").sum()
    n_total = len(summary_df)

    return {
        "총_직송기준_비용(baseline)": total_direct_baseline,
        "총_최적화_비용(우리모델)": total_optimized,
        "총_절감액": total_savings,
        "절감률(%)": round(savings_pct, 2),
        "허브경유_선택_건수": int(n_hub_chosen),
        "전체_건수": int(n_total),
        "허브경유_선택비율(%)": round(n_hub_chosen / n_total * 100, 2) if n_total else 0,
    }


def print_savings_report(summary: dict):
    print("=" * 60)
    print("📊 배차 최적화 결과 요약 (예측기간 전체)")
    print("=" * 60)
    print(f" 직송 기준(baseline) 총비용   : {summary['총_직송기준_비용(baseline)']:,.0f} 원")
    print(f" 우리 모델(예측+허브판단) 총비용: {summary['총_최적화_비용(우리모델)']:,.0f} 원")
    print(f" 총 절감액                  : {summary['총_절감액']:,.0f} 원")
    print(f" 절감률                    : {summary['절감률(%)']}%")
    print(f" 허브경유 선택 건수         : {summary['허브경유_선택_건수']} / {summary['전체_건수']} "
          f"({summary['허브경유_선택비율(%)']}%)")
    print("=" * 60)
