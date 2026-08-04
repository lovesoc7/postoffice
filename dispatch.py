"""
배차 최적화 모듈
=================
2단계 구조로 동작함:

1) 물량재배치(consolidation) 의사결정
   - 목적지가 같고 출발지가 다른 여러 물량을 놓고
   - [직송] 출발지별로 각자 목적지까지 보내는 것과
   - [허브경유] 출발지들을 허브로 몰아 한 번에 합쳐 보내는 것을 비교해서
   - 더 싼 쪽을 선택한다.

2) 트럭 배차(bin-packing)
   - 주어진 구간(출발지->목적지 또는 출발지->허브, 허브->목적지)에서
   - 실제 존재하는 톤급(1/2.5/4.5/5/8/11/18/25톤)의 트럭을 몇 대씩 조합해야
   - 예측 물량을 다 싣고 총비용이 최소가 되는지 정수계획법(OR-Tools CP-SAT)으로 계산.

⚠️ 비용 가정에 대하여
- 우정사업본부 실제 운임/하역비/분류비 단가는 비공개라 공개된 적이 없음.
- 아래 COST_ASSUMPTIONS는 국토교통부 화물자동차 안전운송원가 고시 등을 참고한
  '투명한 가정치'이며, 실제 제출 시 이 숫자들을 바꿔가며 민감도 분석을 해야 함
  (숫자 자체보다 "허브경유가 어떤 조건에서 유리해지는가"라는 구조가 핵심 결과물).
"""

import math
from dataclasses import dataclass
from functools import lru_cache
import pandas as pd

from ortools.sat.python import cp_model

import os
import time
from dotenv import load_dotenv
import pandas as pd
import requests

FILE_PATH = "real_hub_distance_matrix.csv"
REGION_TO_HUB = {
        "서울특별시": "동서울우편집중국",
        "경기도": "수원우편집중국",
        "인천광역시": "부평물류센터",
        "강원특별자치도": "원주우편집중국",
        "충청북도": "청주우편집중국",
        "충청남도": "천안우편집중국",
        "대전광역시": "대전교환센터",
        "세종특별자치시": "대전교환센터",
        "전북특별자치도": "전주우편집중국",
        "전라남도": "순천집중과",
        "광주광역시": "광주우편집중국",
        "경상북도": "안동우편집중국",
        "경상남도": "창원우편집중국",
        "대구광역시": "대구우편집중국",
        "울산광역시": "울산우편집중국",
        "부산광역시": "부산우편집중국",
        "제주특별자치도": "제주우편집중국",
}
if os.path.exists(FILE_PATH):
    print(f"✅ '{FILE_PATH}' 파일이 이미 존재합니다. 기존 데이터를 로드합니다.")
    df_real_matrix = pd.read_csv(FILE_PATH, encoding="cp949")
else:
    print(
        f"🚨 '{FILE_PATH}' 파일이 없습니다. 카카오 API를 호출하여 매트릭스를"
        " 생성합니다..."
    )
    load_dotenv()
    KAKAO_API_KEY = os.getenv("KAKAO_REST_API_KEY")

    if not KAKAO_API_KEY:
        raise ValueError(
            "카카오 API 키를 찾을 수 없습니다. .env 파일을 확인해 주세요!"
        )
    HUB_COORDS = {
        "동서울우편집중국": (127.0963, 37.5350),
        "수원우편집중국": (127.0205, 37.2642),
        "부평물류센터": (126.7210, 37.5075),
        "원주우편집중국": (127.9450, 37.3421),
        "청주우편집중국": (127.4610, 36.6350),
        "천안우편집중국": (127.1510, 36.8150),
        "대전교환센터": (127.3850, 36.3620),
        "전주우편집중국": (127.1350, 35.8250),
        "순천집중과": (127.4890, 34.9510),
        "광주우편집중국": (126.8850, 35.1750),
        "안동우편집중국": (128.7290, 36.5650),
        "창원우편집중국": (128.6810, 35.2280),
        "대구우편집중국": (128.5830, 35.8850),
        "울산우편집중국": (129.3320, 35.5680),
        "부산우편집중국": (128.9830, 35.1580),
        "제주우편집중국": (126.5280, 33.5080),
    }
    matrix_data = []
    df_real_matrix = None
    try:
        for orig_name, orig_coord in HUB_COORDS.items():
            for dest_name, dest_coord in HUB_COORDS.items():
                if orig_name == dest_name:
                    dist = 15.0  # 동일 지역 내 수송거리 기본값
                else:
                    # 🔑 [수정] 카카오모빌리티 길찾기 전용 URL 사용
                    url = f"https://apis-navi.kakaomobility.com/v1/directions?origin={orig_coord[0]},{orig_coord[1]}&destination={dest_coord[0]},{dest_coord[1]}"
                    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
                    res = requests.get(
                        url, headers=headers, timeout=5
                    ).json()
                    dist = round(
                        res["routes"][0]["summary"]["distance"] / 1000.0, 2
                    )
                    time.sleep(0.05)

                matrix_data.append({
                    "발송집중국": orig_name,
                    "도착집중국": dest_name,
                    "운송거리(km)": dist,
                })

        df_real_matrix = pd.DataFrame(matrix_data)
        df_real_matrix.to_csv(FILE_PATH, index=False, encoding="cp949")
        print(f"🎉 파일 생성 완료: '{FILE_PATH}'에 저장되었습니다!")
    except Exception as e:
        print(f"파일 생성 실패, 좌표근사치로 거리를 사용함")
if df_real_matrix is not None:
    df_real_matrix = df_real_matrix.set_index(['발송집중국','도착집중국'])
# ---------------------------------------------------------------------------
# 1. 비용 가정 (투명하게 명시 — 제출 시 민감도 분석 대상)
# ---------------------------------------------------------------------------
TRUCK_CLASSES = [1, 1.4, 2, 2.5, 4.5, 5, 8, 11, 18, 25]  # 톤

COST_ASSUMPTIONS = {
    # 톤급별 (기본출차비/고정비, km당 변동비) - 톤급이 클수록 기본비는 높지만 kg당 단가는 낮아짐
    "fixed_cost_per_dispatch": {
        1: 30_000, 1.4: 33_000, 2: 37_000, 2.5: 40_000, 4.5: 55_000, 5: 60_000,
        8: 80_000, 11: 100_000, 18: 140_000, 25: 180_000,
    },
    "variable_cost_per_km": {
        1: 300, 1.4: 330, 2: 370, 2.5: 400, 4.5: 550, 5: 600,
        8: 800, 11: 1_000, 18: 1_400, 25: 1_800,
    },
    # 허브 경유 시 1회 접촉(터치)당 추가 비용: 하역비+분류비+재적재비+대기시간 인건비 근사
    "hub_handling_cost_per_touch": 25_000,  # 원/건(출발지 1곳이 허브를 거치는 1회 접촉당)
}

TRUCK_CAPACITY_KG = {t: t * 1000 for t in TRUCK_CLASSES}  # 톤 -> kg (만재 기준, 안전마진은 아래서 반영)
LOAD_SAFETY_MARGIN = 0.9  # 실제로는 만재적재 안 하고 90%까지만 채운다고 가정


# ---------------------------------------------------------------------------
# 2. 지역/허브 좌표 (근사치 — 실제 제출본에서는 카카오/네이버 API 좌표로 교체 권장)
#    ※ 여기 좌표는 도청 소재지 근사 위경도이며, 정밀 거리 계산용이 아니라
#       "허브경유가 유리해지는 구조"를 보여주는 데모용입니다.
# ---------------------------------------------------------------------------
REGION_COORDS = {
    "서울특별시": (37.5665, 126.9780), "경기도": (37.4138, 127.5183),
    "인천광역시": (37.4563, 126.7052), "강원특별자치도": (37.8228, 128.1555),
    "충청북도": (36.6357, 127.4917), "충청남도": (36.5184, 126.8000),
    "대전광역시": (36.3504, 127.3845), "세종특별자치시": (36.4800, 127.2890),
    "경상북도": (36.5760, 128.5056), "경상남도": (35.4606, 128.2132),
    "대구광역시": (35.8714, 128.6014), "부산광역시": (35.1796, 129.0756),
    "울산광역시": (35.5384, 129.3114), "전라북도": (35.8242, 127.1480),
    "전북특별자치도": (35.8242, 127.1480), "전라남도": (34.8161, 126.4630),
    "광주광역시": (35.1595, 126.8526), "제주특별자치도": (33.4996, 126.5312),
}

def haversine_km(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lon1 = p1
    lat2, lon2 = p2
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def region_distance_km(region_a: str, region_b: str) -> float:
    if df_real_matrix is None: # 실제 데이터 사용이 불가하다면 근사좌표값을 사용.
        return haversine_km(REGION_COORDS[region_a], REGION_COORDS[region_b])
    # 지역명을 대표 집중국명으로 변환
    hub_a = REGION_TO_HUB.get(region_a)
    hub_b = REGION_TO_HUB.get(region_b)
    # 거리 검색
    try:
        distance = df_real_matrix.loc[(hub_a, hub_b), '운송거리(km)']
    except: # 데이터에서 해당 경로를 못찾은경우 근사치 이용.
        distance = haversine_km(REGION_COORDS[region_a], REGION_COORDS[region_b])
    return distance

# ---------------------------------------------------------------------------
# 3. 트럭 배차(bin-packing) — OR-Tools CP-SAT
# ---------------------------------------------------------------------------
@dataclass
class DispatchResult:
    total_cost: int
    truck_counts: dict          # {톤급: 대수}
    total_capacity_kg: float
    weight_kg: float
    distance_km: float


def optimize_truck_dispatch(weight_kg: float, distance_km: float) -> DispatchResult:
    """
    주어진 구간(거리)에서 weight_kg를 실어 나르는 데 필요한 톤급별 트럭 대수를
    총비용 최소화 기준으로 계산한다 (정수계획법, bin-packing).

    ⚠️ 성능 최적화: 탐욕 알고리즘(evaluate_greedy_consolidation)이 같은 (물량,거리)
    조합을 반복 평가에서 여러 번 다시 계산하는 경우가 많아, 1kg/1km 단위로 반올림한
    캐시를 앞단에 둔다. OR-Tools는 모델 생성+solver 호출 자체의 오버헤드가 커서
    이 캐싱만으로도 대량 반복 호출 시 체감 속도가 크게 개선된다.
    """
    weight_key = round(weight_kg)
    distance_key = round(distance_km)
    return _optimize_truck_dispatch_cached(weight_key, distance_key)


@lru_cache(maxsize=200_000)
def _optimize_truck_dispatch_cached(weight_kg: int, distance_km: int) -> DispatchResult:
    if weight_kg <= 0:
        return DispatchResult(0, {t: 0 for t in TRUCK_CLASSES}, 0.0, 0.0, distance_km)

    model = cp_model.CpModel()
    max_trucks_per_class = int(weight_kg / (min(TRUCK_CAPACITY_KG.values()) * LOAD_SAFETY_MARGIN)) + 2

    x = {
        t: model.NewIntVar(0, max_trucks_per_class, f"x_{t}")
        for t in TRUCK_CLASSES
    }

    # 용량 제약: 안전마진 반영한 실제 가용 용량의 합이 물량 이상이어야 함
    model.Add(
        sum(int(TRUCK_CAPACITY_KG[t] * LOAD_SAFETY_MARGIN) * x[t] for t in TRUCK_CLASSES)
        >= math.ceil(weight_kg)
    )

    # 목적함수: 톤급별 (고정비 + 거리*변동비) * 대수 총합 최소화
    cost_per_truck = {
        t: COST_ASSUMPTIONS["fixed_cost_per_dispatch"][t]
        + COST_ASSUMPTIONS["variable_cost_per_km"][t] * distance_km
        for t in TRUCK_CLASSES
    }
    model.Minimize(sum(int(cost_per_truck[t]) * x[t] for t in TRUCK_CLASSES))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"배차 최적화 실패 (status={status}), weight={weight_kg}, dist={distance_km}")

    truck_counts = {t: solver.Value(x[t]) for t in TRUCK_CLASSES}
    total_cost = int(solver.ObjectiveValue())
    total_capacity = sum(TRUCK_CAPACITY_KG[t] * LOAD_SAFETY_MARGIN * truck_counts[t] for t in TRUCK_CLASSES)

    return DispatchResult(total_cost, truck_counts, total_capacity, weight_kg, distance_km)


# ---------------------------------------------------------------------------
# 4. 물량재배치(허브경유 vs 직송) 의사결정
# ---------------------------------------------------------------------------
@dataclass
class ConsolidationResult:
    decision: str                 # "direct" or "hub"
    chosen_hub: str | None
    direct_cost: int
    hub_cost: dict                # {허브명: 총비용}
    best_hub_cost: int | None
    savings: int                  # 직송 대비 허브경유 절감액 (양수면 허브가 더 쌈)
    detail: dict


def evaluate_greedy_consolidation(
    destination: str,
    origin_weights_kg: dict[str, float],
) -> ConsolidationResult:
    """탐욕법(Greedy) 기반 단계적 몰아주기 배차 최적화.

    1. 모든 출발지 쌍 (i -> j)에 대해 [각자 직송] vs [i를 j로 몰아주기] 절감액 계산
    2. 가장 절감 금액이 큰 (i, j) 조합을 선택하여 i 물량을 j로 통합 (j 물동량 갱신, i 제거)
    3. 이 과정을 더 이상 절감되는 조합이 없을 때까지 반복
    """
    # 물량 복사본 준비
    weights = origin_weights_kg.copy()

    # ------------------------------------------------------------------
    # 1. Baseline: 처음에 모두 각자 직송할 때의 총비용 계산
    # ------------------------------------------------------------------
    direct_details = {}
    direct_total_cost = 0
    for origin, w in weights.items():
        if w > 0:
            dist = region_distance_km(origin, destination)
            r = optimize_truck_dispatch(w, dist)
            direct_details[origin] = r
            direct_total_cost += r.total_cost

    # 상태 추적 변수
    active_origins = [o for o, w in weights.items() if w > 0]
    total_feed_cost = 0  # 몰아주는 1구간 트럭 비용 누적
    total_handling_cost = 0  # 하역/분류비 누적
    merge_history = []  # 병합 히스토리 기록

    handling_cost_per_touch = COST_ASSUMPTIONS.get(
        "hub_handling_cost_per_touch", 10000
    )

    # ------------------------------------------------------------------
    # 2. 탐욕적 반복 병합 (Greedy Loop)
    # ------------------------------------------------------------------
    while len(active_origins) > 1:
        best_savings = 0
        best_pair = None  # (feeder_i, target_j)
        best_feed_dispatch = None

        # 현재 살아있는 모든 출발지 쌍 (i -> j) 전수 조사
        for i_origin in active_origins:
            for j_origin in active_origins:
                if i_origin == j_origin:
                    continue

                w_i = weights[i_origin]
                w_j = weights[j_origin]

                # (A) 두 지역이 지금 상태에서 각각 목적지로 직송할 때의 비용 (OR-Tools)
                dist_i_dest = region_distance_km(i_origin, destination)
                dist_j_dest = region_distance_km(j_origin, destination)
                cost_i_direct = optimize_truck_dispatch(
                    w_i, dist_i_dest
                ).total_cost
                cost_j_direct = optimize_truck_dispatch(
                    w_j, dist_j_dest
                ).total_cost
                pair_direct_cost = cost_i_direct + cost_j_direct

                # (B) i를 j로 몰아주고, j에서 합쳐서 목적지로 쏠 때의 비용 (OR-Tools)
                dist_i_j = region_distance_km(i_origin, j_origin)
                r_feed = optimize_truck_dispatch(
                    w_i, dist_i_j
                )  # i -> j 피딩 트럭비

                cost_merged = optimize_truck_dispatch(
                    w_i + w_j, dist_j_dest
                ).total_cost  # (i+j) -> D 합침 트럭비

                consolidated_cost = (
                    r_feed.total_cost + cost_merged + handling_cost_per_touch
                )

                # (C) 절감 금액 계산
                savings = pair_direct_cost - consolidated_cost

                # 가장 절감 금액이 큰 쌍으로 갱신
                if savings > best_savings:
                    best_savings = savings
                    best_pair = (i_origin, j_origin)
                    best_feed_dispatch = r_feed

        # ⚠️ 더 이상 돈이 절약되는 조합이 없으면 탐욕 반복 종료!
        if best_pair is None or best_savings <= 0:
            break

        # 가장 이득이 큰 조합으로 물량 병합 실행
        feeder, target = best_pair

        # 비용 및 병합 내역 기록
        total_feed_cost += best_feed_dispatch.total_cost
        total_handling_cost += handling_cost_per_touch
        merge_history.append({
            "feeder": feeder,
            "target": target,
            "moved_weight_kg": weights[feeder],
            "feed_cost": best_feed_dispatch.total_cost,
            "savings": best_savings,
        })

        # 🔑 물류량 갱신: feeder의 물량을 target에 더해주고, feeder는 목록에서 제거
        weights[target] += weights[feeder]
        weights[feeder] = 0
        active_origins.remove(feeder)

    # ------------------------------------------------------------------
    # 3. 최종 배차 및 절감액 계산
    # ------------------------------------------------------------------
    # 최종 남아있는 거점 노드들 -> 목적지(D) 배차 계산 (OR-Tools)
    final_direct_cost = 0
    final_dispatches = {}
    for o in active_origins:
        dist = region_distance_km(o, destination)
        r = optimize_truck_dispatch(weights[o], dist)
        final_dispatches[o] = r
        final_direct_cost += r.total_cost

    final_total_cost = (
        total_feed_cost + total_handling_cost + final_direct_cost
    )
    total_savings = direct_total_cost - final_total_cost

    return ConsolidationResult(
        decision="hub" if merge_history else "direct",
        chosen_hub=merge_history[-1]["target"] if merge_history else None,
        direct_cost=direct_total_cost,
        hub_cost={"greedy_merged": final_total_cost},
        best_hub_cost=final_total_cost,
        savings=total_savings,
        detail={
            "direct": direct_details,
            "merge_history": merge_history,  # 👈 어떤 순서로 합쳐졌는지 단계별 히스토리
            "final_dispatches": final_dispatches,
            "active_hubs": active_origins,
        },
    )


def print_consolidation_result(destination: str, origin_weights_kg: dict, result: ConsolidationResult):
    print("=" * 60)
    print(f"📦 목적지: {destination}")
    print(f"   출발지별 물량: { {k: f'{v:,.0f}kg' for k, v in origin_weights_kg.items()} }")
    print("-" * 60)
    print(f" [직송] 총비용         : {result.direct_cost:,} 원")
    for hub, cost in result.hub_cost.items():
        mark = " ⭐" if hub == result.chosen_hub else ""
        print(f" [허브경유:{hub}] 총비용 : {cost:,} 원{mark}")
    print("-" * 60)
    if result.decision == "hub":
        print(f" ✅ 결정: 허브경유({result.chosen_hub})가 유리 (절감액 {result.savings:,.0f}원)")
    else:
        print(f" ✅ 결정: 직송이 유리 (허브경유 대비 {abs(result.savings):,.0f}원 더 저렴)")
    print("=" * 60)
