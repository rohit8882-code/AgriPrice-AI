"""
Deep Learning Architecture: Bidirectional LSTM with Temporal Attention and Quantile Loss.
Captures non-linear long-term cyclical dependencies and provides P10, P50, and P90 forecast bands.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TemporalAttention(nn.Module):
        """Bahdanau-style temporal attention over recurrent hidden states."""
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, r_outputs):
            # r_outputs: [batch_size, seq_len, hidden_dim]
            weights = self.attention_weights(r_outputs) # [batch_size, seq_len, 1]
            normalized_weights = torch.softmax(weights, dim=1)
            context = torch.sum(normalized_weights * r_outputs, dim=1) # [batch_size, hidden_dim]
            return context, normalized_weights


    class PinballLoss(nn.Module):
        """Multi-Quantile Pinball Loss for P10, P50, and P90 interval estimation."""
        def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90]):
            super().__init__()
            self.quantiles = quantiles

        def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # preds: [batch, len(quantiles)], target: [batch, 1]
            losses = []
            for i, q in enumerate(self.quantiles):
                error = target - preds[:, i:i+1]
                loss = torch.max((q - 1) * error, q * error)
                losses.append(loss)
            return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


    class AgriculturalBiLSTM(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, output_quantiles: int = 3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0.0
            )
            self.attention = TemporalAttention(hidden_dim * 2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, output_quantiles) # [P10, P50, P90]
            )

        def forward(self, x):
            # x: [batch, seq_len, input_dim]
            lstm_out, _ = self.lstm(x)
            context, attn_weights = self.attention(lstm_out)
            out = self.fc(context)
            return out, attn_weights

class DeepLearningPipeline:
    def __init__(self, seq_len: int = 30, hidden_dim: int = 64, num_layers: int = 2):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.feature_cols: List[str] = []
        self.target_col: str = "target_price_15d"
        self.x_mean = None
        self.x_std = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def is_available(self) -> bool:
        return TORCH_AVAILABLE

    def prepare_sequences(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts tabular series into 3D sliding window tensors (Batch x SeqLen x Features)."""
        X_seqs, y_seqs = [], []
        for i in range(len(X_scaled) - self.seq_len):
            X_seqs.append(X_scaled[i : i + self.seq_len])
            y_seqs.append(y_scaled[i + self.seq_len])
        return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32).reshape(-1, 1)

    def train(
        self,
        df_features: pd.DataFrame,
        feature_cols: List[str] = None,
        target_col: str = "target_price_15d",
        epochs: int = 20,
        lr: float = 0.005,
        batch_size: int = 32,
        test_days: int = 30
    ) -> Dict[str, float]:
        """
        Trains the PyTorch Bi-LSTM with Temporal Attention and Pinball Loss.
        Evaluates on an out-of-time test window.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DeepLearningPipeline.")

        self.target_col = target_col
        df = df_features.dropna(subset=[target_col]).copy()
        if len(df) < self.seq_len + test_days + 10:
            raise ValueError(f"Insufficient historical data ({len(df)} rows) for sequence length {self.seq_len}.")

        if feature_cols is None:
            ignore = ["record_date", "mandi_name", "commodity", "state", "district", "data_source"] + [
                c for c in df.columns if c.startswith("target_")
            ]
            self.feature_cols = [c for c in df.columns if c not in ignore and np.issubdtype(df[c].dtype, np.number)]
        else:
            self.feature_cols = [c for c in feature_cols if c in df.columns]

        # Feature normalization
        X_raw = df[self.feature_cols].fillna(0).values.astype(np.float32)
        y_raw = df[target_col].values.astype(np.float32)

        split_idx = len(df) - test_days
        X_train_raw, y_train_raw = X_raw[:split_idx], y_raw[:split_idx]
        X_test_raw, y_test_raw = X_raw[split_idx:], y_raw[split_idx:]

        self.x_mean = np.nanmean(X_train_raw, axis=0)
        self.x_std = np.nanstd(X_train_raw, axis=0)
        self.x_std[self.x_std == 0] = 1.0

        self.y_mean = float(np.mean(y_train_raw))
        self.y_std = float(np.std(y_train_raw)) or 1.0

        X_train_scaled = (X_train_raw - self.x_mean) / self.x_std
        y_train_scaled = (y_train_raw - self.y_mean) / self.y_std

        X_test_scaled = (X_test_raw - self.x_mean) / self.x_std
        y_test_scaled = (y_test_raw - self.y_mean) / self.y_std

        X_tr_seq, y_tr_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        
        # Build sequences for test set with context from train end
        combined_scaled_X = np.vstack([X_train_scaled[-self.seq_len:], X_test_scaled])
        combined_scaled_y = np.concatenate([y_train_scaled[-self.seq_len:], y_test_scaled])
        X_te_seq, y_te_seq = self.prepare_sequences(combined_scaled_X, combined_scaled_y)

        # Initialize Model
        input_dim = len(self.feature_cols)
        self.model = AgriculturalBiLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_quantiles=3
        )
        self.model.train()

        criterion = PinballLoss(quantiles=[0.10, 0.50, 0.90])
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Dataset & Loader
        t_x = torch.from_numpy(X_tr_seq)
        t_y = torch.from_numpy(y_tr_seq)
        dataset = torch.utils.data.TensorDataset(t_x, t_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds, _ = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()

        # Evaluate on Test Set
        self.model.eval()
        with torch.no_grad():
            te_x_tensor = torch.from_numpy(X_te_seq)
            raw_preds, _ = self.model(te_x_tensor)
            preds_arr = raw_preds.cpu().numpy()

        # Unscale P50 predictions
        p50_preds = (preds_arr[:, 1] * self.y_std) + self.y_mean
        p50_preds = np.maximum(50.0, p50_preds)

        y_actual = y_test_raw[:len(p50_preds)]
        y_prior = df["modal_price"].iloc[split_idx : split_idx + len(p50_preds)].values

        from models.evaluator import evaluate_forecast
        metrics = evaluate_forecast(y_actual, p50_preds, y_prior)
        return metrics

    def predict_latest(self, recent_features: pd.DataFrame) -> Dict[str, float]:
        """
        Takes recent historical feature rows (at least seq_len rows) and outputs
        forecast quantiles: P10, P50, P90, along with attention weights.
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before prediction.")

        self.model.eval()
        tail_df = recent_features[self.feature_cols].tail(self.seq_len).fillna(0)
        if len(tail_df) < self.seq_len:
            # Pad with first available row if sequence is slightly shorter
            pad_needed = self.seq_len - len(tail_df)
            pad_rows = pd.concat([tail_df.iloc[[0]]] * pad_needed, ignore_index=True)
            tail_df = pd.concat([pad_rows, tail_df], ignore_index=True)

        X_raw = tail_df.values.astype(np.float32)
        X_scaled = (X_raw - self.x_mean) / self.x_std
        X_tensor = torch.from_numpy(X_scaled).unsqueeze(0) # [1, seq_len, input_dim]

        with torch.no_grad():
            preds, attn = self.model(X_tensor)
            preds = preds.squeeze(0).cpu().numpy()
            attn = attn.squeeze().cpu().numpy()

        p10 = float((preds[0] * self.y_std) + self.y_mean)
        p50 = float((preds[1] * self.y_std) + self.y_mean)
        p90 = float((preds[2] * self.y_std) + self.y_mean)

        p10 = max(50.0, min(p10, p50))
        p90 = max(p50, p90)

        return {
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "attention_weights": attn.tolist() if hasattr(attn, "tolist") else []
        }
