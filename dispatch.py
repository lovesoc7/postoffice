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

from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# 0. 트럭 톤급 카탈로그 (2021년 우정사업본부 실데이터에서 확인된 실제 운용 톤급)
# ---------------------------------------------------------------------------
TRUCK_CLASSES = [1, 2.5, 4.5, 5, 8, 11, 18, 25]  # 톤

# ---------------------------------------------------------------------------
# 1. 비용 가정 (투명하게 명시 — 제출 시 민감도 분석 대상)
# ---------------------------------------------------------------------------
COST_ASSUMPTIONS = {
    # 톤급별 (기본출차비/고정비, km당 변동비) - 톤급이 클수록 기본비는 높지만 kg당 단가는 낮아짐
    "fixed_cost_per_dispatch": {  # 원/1회 출차 (상하차 준비, 기사 기본수당 등)
        1: 30_000, 2.5: 40_000, 4.5: 55_000, 5: 60_000,
        8: 80_000, 11: 100_000, 18: 140_000, 25: 180_000,
    },
    "variable_cost_per_km": {  # 원/km (연료비+통행료+감가상각 근사)
        1: 300, 2.5: 400, 4.5: 550, 5: 600,
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
#      "허브경유가 유리해지는 구조"를 보여주는 데모용입니다.
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

# 실제 3개 물류사업소(우체국물류지원단 공개 위치정보 기준 대략 좌표)
HUB_COORDS = {
    "동서울물류센터": (37.5504, 127.2495),  # 하남시 인근
    "안양물류센터": (37.3943, 126.9568),
    "부평물류센터": (37.5069, 126.7218),
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
    return haversine_km(REGION_COORDS[region_a], REGION_COORDS[region_b])


def region_to_hub_distance_km(region: str, hub: str) -> float:
    return haversine_km(REGION_COORDS[region], HUB_COORDS[hub])


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
    """
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


def evaluate_consolidation(
    destination: str,
    origin_weights_kg: dict,      # {출발지역: 물량kg}
    hub_candidates: list = None,
) -> ConsolidationResult:
    """
    같은 목적지로 가는 여러 출발지 물량에 대해
    [직송] vs [허브경유] 비용을 비교하여 더 싼 쪽을 결정한다.
    """
    if hub_candidates is None:
        hub_candidates = list(HUB_COORDS.keys())

    # --- 옵션 A: 직송 (출발지별로 각자 목적지까지) ---
    direct_detail = {}
    direct_total = 0
    for origin, w in origin_weights_kg.items():
        dist = region_distance_km(origin, destination)
        r = optimize_truck_dispatch(w, dist)
        direct_detail[origin] = r
        direct_total += r.total_cost

    # --- 옵션 B: 허브경유 (출발지들 -> 허브 -> 목적지, 허브에서 합쳐서 1번에) ---
    hub_costs = {}
    hub_details = {}
    for hub in hub_candidates:
        # 1구간: 각 출발지 -> 허브 (각자 트럭배차)
        leg1_cost = 0
        leg1_detail = {}
        for origin, w in origin_weights_kg.items():
            dist = region_to_hub_distance_km(origin, hub)
            r = optimize_truck_dispatch(w, dist)
            leg1_detail[origin] = r
            leg1_cost += r.total_cost

        # 허브 접촉(하역+분류+재적재+대기) 비용: 출발지 수만큼 발생
        handling_cost = COST_ASSUMPTIONS["hub_handling_cost_per_touch"] * len(origin_weights_kg)

        # 2구간: 허브 -> 목적지 (합쳐진 물량을 "한 번에" 배차 = 트럭 대수 절감 효과 발생)
        total_weight = sum(origin_weights_kg.values())
        dist_hub_to_dest = region_distance_km(destination, destination) if hub == destination else haversine_km(
            HUB_COORDS[hub], REGION_COORDS[destination]
        )
        leg2 = optimize_truck_dispatch(total_weight, dist_hub_to_dest)

        total = leg1_cost + handling_cost + leg2.total_cost
        hub_costs[hub] = total
        hub_details[hub] = {
            "leg1_구간별": leg1_detail,
            "handling_cost": handling_cost,
            "leg2_허브to목적지": leg2,
        }

    best_hub = min(hub_costs, key=hub_costs.get) if hub_costs else None
    best_hub_cost = hub_costs[best_hub] if best_hub else None

    if best_hub_cost is not None and best_hub_cost < direct_total:
        decision = "hub"
        savings = direct_total - best_hub_cost
    else:
        decision = "direct"
        savings = direct_total - (best_hub_cost or direct_total)

    return ConsolidationResult(
        decision=decision,
        chosen_hub=best_hub if decision == "hub" else None,
        direct_cost=direct_total,
        hub_cost=hub_costs,
        best_hub_cost=best_hub_cost,
        savings=savings,
        detail={"direct": direct_detail, "hub": hub_details},
    )


def print_consolidation_result(destination: str, origin_weights_kg: dict, result: ConsolidationResult):
    print("=" * 60)
    print(f"📦 목적지: {destination}")
    print(f"   출발지별 물량: { {k: f'{v:,.0f}kg' for k, v in origin_weights_kg.items()} }")
    print("-" * 60)
    print(f" [직송] 총비용        : {result.direct_cost:,} 원")
    for hub, cost in result.hub_cost.items():
        mark = " ⭐" if hub == result.chosen_hub else ""
        print(f" [허브경유:{hub}] 총비용 : {cost:,} 원{mark}")
    print("-" * 60)
    if result.decision == "hub":
        print(f" ✅ 결정: 허브경유({result.chosen_hub})가 유리 (절감액 {result.savings:,.0f}원)")
    else:
        print(f" ✅ 결정: 직송이 유리 (허브경유 대비 {abs(result.savings):,.0f}원 더 저렴)")
    print("=" * 60)
