import lightgbm as lgb
import numpy as np
import pandas as pd

def _additional_preprocess(df:pd.DataFrame,split=True)->tuple[pd.DataFrame,pd.DataFrame]|pd.DataFrame:
    # lgbm모델에 맞춰 전처리: 필요없는 열 떨구고 X, y 분리
    d = "접수일자"
    df = df.drop(columns=[d])
    if split:
        l = "총중량_kg"
        X = df.drop(columns=[l])
        y = df[l]
        return X,y
    return df
###################################################################
def train_lgbm(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> lgb.LGBMRegressor:
    """
    LightGBM 시계열 회귀 모델을 학습시켜서 반환함.

    ⚠️ 중요: df_eval은 반드시 '훈련기간 내부'에서 떼어낸 검증(validation)셋이어야 합니다.
    최종 테스트셋(df_test)을 여기에 그대로 넣으면 early stopping이 테스트셋을
    보고 멈출 시점을 정하게 되어 테스트 성능이 실제보다 낙관적으로 나옵니다(누수).
    사용 예:
        df_train2, df_valid = data.split_timeseries_data(df_train, test_days=30)
        model = train_lgbm(df_train2, df_valid)   # df_test는 여기 쓰지 않음
    """
    # lgbm모델에 맞춰 전처리: 필요없는 열 떨구고 X, y 분리
    X_train, y_train = _additional_preprocess(df_train)
    X_eval, y_eval = _additional_preprocess(df_eval)
    
    # LightGBM 회귀 모델 정의
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    # 모델 학습 (Early Stopping 적용)
    print("LightGBM 모델 학습 시작...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_eval, y_eval)],
        eval_names=['train', 'valid'],
        eval_metric='mae',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    return model
def predict_lgbm(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    예측 및 후처리 (물량은 음수가 나올 수 없음).
    타겟 컬럼('총중량_kg') 존재 여부를 명시적으로 확인해서 분기한다.
    (기존 try/except는 어떤 에러든 다 삼켜서 진짜 버그를 숨길 수 있었음)
    """
    label_col = "총중량_kg"
    if label_col in X.columns:
        X_processed, _ = _additional_preprocess(X, split=True)
    else:
        X_processed = _additional_preprocess(X, split=False)

    y_pred = model.predict(X_processed)
    y_pred = np.maximum(y_pred, 0)
    return y_pred