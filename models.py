from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def get_wape(y_test, y_pred):
    return (np.sum(np.abs(y_test - y_pred)) / np.sum(y_test)) * 100
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