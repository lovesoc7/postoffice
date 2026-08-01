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
def train_lgbm(df_train:pd.DataFrame, df_eval:pd.DataFrame)->lgb.LGBMRegressor:
    """
    LightGBM 시계열 회귀 모델을 학습시켜서 반환함.
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
def predict_lgbm(model:lgb.DaskLGBMRegressor, X:pd.DataFrame)->pd.DataFrame:
    # 예측 및 후처리 (물량은 음수가 나올 수 없음)
    
    try:
        X = _additional_preprocess(X)[0]
    except:
        X = _additional_preprocess(X,False)
    y_pred = model.predict(X)
    y_pred = np.maximum(y_pred, 0)
    return y_pred