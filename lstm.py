import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

SEQ_LEN = 14
CAT_COLS = ["접수지역", "배달지역", "월", "요일", "공휴일여부"]
CONT_COLS = ["총중량_kg"]


# ----------------------------------------------------
# 1. Dataset 정의 (X와 Y를 1:1 세트로 묶어서 보관)
# ----------------------------------------------------
class RouteEmbeddingDataset(Dataset):

    def __init__(
        self,
        df,
        cat_cols=CAT_COLS,
        cont_cols=CONT_COLS,
        label_encoders=None,
        scaler=None,
        is_train=True,
        has_label=None,
    ):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols

        # 타겟 컬럼 존재 여부로 라벨 유무 자동 판별
        if has_label is None:
            self.has_label = "총중량_kg" in df.columns
        else:
            self.has_label = has_label

        df_processed = df.copy()

        # ⚠️ [핵심] groupby/sort 전에 원본 인덱스를 'orig_idx' 컬럼으로 보존
        df_processed["orig_idx"] = df.index.values

        # 1-1. Label Encoding
        self.label_encoders = (
            label_encoders if label_encoders is not None else {}
        )
        for col in cat_cols:
            if is_train:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(
                    df_processed[col].astype(str)
                )
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df_processed[col] = (
                    df_processed[col]
                    .astype(str)
                    .map(
                        lambda s: (
                            le.transform([s])[0] if s in le.classes_ else 0
                        )
                    )
                )

        # 1-2. StandardScaler
        if is_train:
            self.scaler = StandardScaler()
            df_processed[cont_cols] = self.scaler.fit_transform(
                df_processed[cont_cols]
            )
        else:
            self.scaler = scaler
            df_processed[cont_cols] = self.scaler.transform(
                df_processed[cont_cols]
            )

        self.X_cont = []
        self.X_cat = []
        self.target_indices = []  # 👈 예측 대상의 원본 인덱스 저장용
        if self.has_label:
            self.y = []

        # 1-3. 시퀀스 데이터 생성
        groups = df_processed.groupby(["접수지역", "배달지역"])

        for _, g in groups:
            # ⚠️ reset_index(drop=True)를 하지 않고 날짜로만 정렬하여 orig_idx 유지
            g = g.sort_values("접수일자")

            if len(g) <= SEQ_LEN:
                continue

            cont_values = g[cont_cols].values.astype(np.float32)
            cat_values = g[cat_cols].values.astype(np.int64)
            orig_idxs = g["orig_idx"].values  # 원본 인덱스 배열

            if self.has_label:
                target_values = g["총중량_kg"].values.astype(np.float32)
                for i in range(len(g) - SEQ_LEN):
                    self.X_cont.append(cont_values[i : i + SEQ_LEN])
                    self.X_cat.append(cat_values[i : i + SEQ_LEN])
                    self.y.append(target_values[i + SEQ_LEN])
                    # ⚠️ 예측 대상이 되는 (i + SEQ_LEN) 위치의 원본 인덱스 기록
                    self.target_indices.append(orig_idxs[i + SEQ_LEN])
            else:
                for i in range(len(g) - SEQ_LEN):
                    self.X_cont.append(cont_values[i : i + SEQ_LEN])
                    self.X_cat.append(cat_values[i : i + SEQ_LEN])
                    self.target_indices.append(orig_idxs[i + SEQ_LEN])

        self.X_cont = np.array(self.X_cont, dtype=np.float32)
        self.X_cat = np.array(self.X_cat, dtype=np.int64)
        self.target_indices = np.array(self.target_indices)
        if self.has_label:
            self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        if self.has_label:
            return (
                torch.tensor(self.X_cont[idx]),
                torch.tensor(self.X_cat[idx]),
                torch.tensor(self.y[idx]),
            )
        else:
            return torch.tensor(self.X_cont[idx]), torch.tensor(self.X_cat[idx])


# ----------------------------------------------------
# 2. PyTorch 모델 정의 (Embedding + LSTM)
# ----------------------------------------------------
class CategoricalLSTMRegressor(nn.Module):

    def __init__(
        self,
        cat_cardinalities,
        cat_cols=CAT_COLS,
        num_cont=1,
        hidden_dim=64,
        num_layers=2,
    ):
        super().__init__()
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict()

        total_emb_dim = 0
        for col in cat_cols:
            num_classes = cat_cardinalities[col]
            emb_dim = max(2, min(50, (num_classes + 1) // 2))
            self.embeddings[col] = nn.Embedding(num_classes + 1, emb_dim)
            total_emb_dim += emb_dim

        lstm_input_dim = num_cont + total_emb_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x_cont, x_cat):
        emb_outputs = []
        for idx, col in enumerate(self.cat_cols):
            cat_tensor = x_cat[:, :, idx]
            emb_out = self.embeddings[col](cat_tensor)
            emb_outputs.append(emb_out)

        x = torch.cat([x_cont] + emb_outputs, dim=-1)
        out, (h, c) = self.lstm(x)
        h_last = h[-1]
        out = self.fc(h_last).squeeze(-1)
        return out


# ----------------------------------------------------
# 3. 모델 학습 함수 (Train 전용)
# ----------------------------------------------------
def train_lstm(
    train_df: pd.DataFrame,
    epochs=30,
    batch_size=64,
    lr=1e-3,
):
    train_dataset = RouteEmbeddingDataset(
        train_df, cat_cols=CAT_COLS, cont_cols=CONT_COLS, is_train=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    cat_cardinalities = {
        col: len(train_dataset.label_encoders[col].classes_)
        for col in CAT_COLS
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 장치: {device}")

    model = CategoricalLSTMRegressor(
        cat_cardinalities=cat_cardinalities,
        cat_cols=CAT_COLS,
        num_cont=len(CONT_COLS),
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n🚀 Categorical Embedding + LSTM 모델 학습 시작...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = (
                x_cont.to(device),
                x_cat.to(device),
                y.to(device),
            )

            optimizer.zero_grad()
            pred = model(x_cont, x_cat)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Loss (Huber): {avg_loss:.5f}"
            )

    return model, train_dataset.label_encoders, train_dataset.scaler


# ----------------------------------------------------
# 4. 모델 추론/평가 함수 (Predict 전용)
# ----------------------------------------------------
def predict_lstm(
    model: CategoricalLSTMRegressor,
    df: pd.DataFrame,
    label_encoders: dict,
    scaler: StandardScaler,
    batch_size=64,
):
    dataset = RouteEmbeddingDataset(
        df,
        cat_cols=CAT_COLS,
        cont_cols=CONT_COLS,
        label_encoders=label_encoders,
        scaler=scaler,
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = next(model.parameters()).device
    model.eval()

    preds_scaled = []

    with torch.no_grad():
        if dataset.has_label:
            for x_cont, x_cat, _ in loader:
                x_cont, x_cat = x_cont.to(device), x_cat.to(device)
                pred = model(x_cont, x_cat).cpu()
                preds_scaled.extend(pred.numpy())
        else:
            for x_cont, x_cat in loader:
                x_cont, x_cat = x_cont.to(device), x_cat.to(device)
                pred = model(x_cont, x_cat).cpu()
                preds_scaled.extend(pred.numpy())

    preds_scaled = np.array(preds_scaled)

    # kg 단위로 역변환 (Inverse Transform)
    weight_mean = scaler.mean_[0]
    weight_std = scaler.scale_[0]

    preds_kg = np.clip(
        preds_scaled * weight_std + weight_mean, a_min=0, a_max=None
    )

    # ⚠️ [핵심] 예측값(preds_kg)과 1:1로 매칭되는 원본 인덱스(target_indices) 반환
    return preds_kg, dataset.target_indices
