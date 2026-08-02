from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def get_wape(y_test, y_pred):
    return (np.sum(np.abs(y_test - y_pred)) / np.sum(y_test)) * 100

def find_best_ensemble_weight(y_valid, pred_lstm_valid, pred_lgb_valid, step=0.01):
    """
    검증셋(validation, 훈련기간 내부에서 뗀 세트)에서 WAPE가 최소가 되는
    LSTM 가중치(w_lstm)를 탐색한다. 테스트셋은 절대 사용하지 않는다.

    기존 코드는 (0.585, 0.415)라는 가중치가 어디서 나왔는지 근거가 없었고,
    만약 테스트셋 성능을 보면서 손으로 튜닝한 것이라면 그 자체가 누수임.
    """
    best_w, best_wape = 0.5, float("inf")
    for w in np.arange(0.0, 1.0 + step, step):
        blended = w * pred_lstm_valid + (1 - w) * pred_lgb_valid
        wape = get_wape(y_valid, blended)
        if wape < best_wape:
            best_wape = wape
            best_w = w
    print(f"🔍 검증셋 기준 최적 앙상블 가중치: w_lstm={best_w:.2f}, w_lgb={1-best_w:.2f} (valid WAPE={best_wape:.2f}%)")
    return best_w, 1 - best_w
def evaluate_outputs(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    wape = get_wape(y_test, y_pred)
    print("\n" + "=" * 50)
    print("📈 모델 평가 결과")
    print("=" * 50)
    print(f" MAE  (평균 절대 오차) : {mae:,.2f} kg")
    print(f" RMSE (평균 제곱근 오차): {rmse:,.2f} kg")
    print(f" WAPE (가중 오차율)    : {wape:.2f} %")
    print("=" * 50)