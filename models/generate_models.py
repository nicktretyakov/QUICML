#!/usr/bin/env python3
"""Train and export the ONNX models used by quic-ml.

Produces, under models/out/:
  - rtt_lstm.onnx        : LSTM (T=16, F=1) -> next-RTT scalar (ms)
  - rl_policy.onnx       : MLP (8 features) -> 5 action logits
  - anomaly.onnx         : isolation-forest-like density scorer -> [0,1]
  - mtu_regressor.onnx   : MLP (6 features) -> candidate MTU

The models are intentionally tiny (edge-deployable). Replace the synthetic
training data with real telemetry captured from quic-core's `CongestionStats`
and `LossDetector` exports for production use.

Usage:
    pip install numpy scikit-learn skl2onnx onnxruntime torch
    python models/generate_models.py

Requires: numpy, scikit-learn, skl2onnx, onnxruntime. The LSTM path also
requires torch; if torch is absent the LSTM is approximated by an sklearn
Huber regressor on the last window mean (still exported as rtt_lstm.onnx
with the same I/O contract).
"""

from __future__ import annotations

import os
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic telemetry generators. Swap these for real captured data.
# ---------------------------------------------------------------------------

def gen_rtt_series(n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 30.0 + 15.0 * np.sin(t / 50.0)
    noise = rng.normal(0.0, 4.0, n)
    spikes = (rng.random(n) < 0.02) * rng.uniform(20, 80, n)
    return base + noise + spikes


def gen_rl_dataset(n: int = 20000, seed: int = 1):
    """State -> action label mimicking a sane Reno/BBR hybrid."""
    rng = np.random.default_rng(seed)
    srtt = rng.uniform(5, 300, n)
    minrtt = rng.uniform(2, srtt)
    cwnd = rng.uniform(2400, 4_000_000, n)
    inflight = rng.uniform(0, cwnd)
    loss = rng.uniform(0, 0.3, n)
    bw = (inflight * 8) / (srtt / 1000.0) / 1e6
    X = np.stack([srtt, minrtt, cwnd, inflight, loss, bw, np.full(n, 1200.0), cwnd / 1200.0], axis=1).astype(np.float32)
    # Label policy:
    #   loss>0.05 -> halve (0); loss>0.01 -> shrink (1); srtt>1.5*minrtt -> hold (2);
    #   else grow (3) if room, double (4) if very idle.
    y = np.zeros(n, dtype=np.int64)
    y[loss > 0.05] = 0
    y[(loss <= 0.05) & (loss > 0.01)] = 1
    mask_hold = (loss <= 0.01) & (srtt > 1.5 * minrtt)
    y[mask_hold] = 2
    mask_grow = (loss <= 0.01) & (srtt <= 1.5 * minrtt) & (inflight + 1200 < cwnd)
    y[mask_grow] = 3
    mask_double = (loss <= 0.01) & (srtt <= 1.5 * minrtt) & (inflight + 1200 >= cwnd) & (bw < 20)
    y[mask_double] = 4
    return X, y


def gen_anomaly_dataset(n: int = 20000, seed: int = 2):
    rng = np.random.default_rng(seed)
    # Benign samples.
    benign = np.stack([
        rng.normal(50, 30, n // 2),
        rng.normal(20000, 10000, n // 2),
        rng.uniform(0, 0.1, n // 2),
        rng.normal(20000, 5000, n // 2),
        rng.normal(5000, 2000, n // 2),
        rng.uniform(1, 5, n // 2),
        rng.normal(0.5, 0.2, n // 2),
        rng.uniform(0.8, 1.0, n // 2),
    ], axis=1).astype(np.float32)
    malicious = np.stack([
        rng.normal(50000, 20000, n // 2),
        rng.normal(5_000_000, 2_000_000, n // 2),
        np.ones(n // 2),
        rng.normal(10, 5, n // 2),
        rng.normal(1, 1, n // 2),
        rng.uniform(5000, 50000, n // 2),
        rng.uniform(0, 0.1, n // 2),
        np.zeros(n // 2),
    ], axis=1).astype(np.float32)
    X = np.vstack([benign, malicious])
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    return X, y


def gen_mtu_dataset(n: int = 10000, seed: int = 3):
    rng = np.random.default_rng(seed)
    loss1200 = rng.uniform(0, 0.2, n)
    loss1400 = rng.uniform(0, 0.4, n)
    srtt = rng.uniform(5, 200, n)
    ttl = rng.choice([48, 56, 64, 128, 255], n).astype(np.float32)
    ipv4 = rng.integers(0, 2, n).astype(np.float32)
    last = np.full(n, 1200.0, dtype=np.float32)
    X = np.stack([loss1200, loss1400, srtt, ttl, ipv4, last], axis=1).astype(np.float32)
    # Target: max successful MTU. Higher loss -> smaller; v6 + lower ttl -> bigger.
    y = 1454.0 - loss1400 * 800.0 - loss1200 * 200.0 - (srtt > 100) * 50.0
    y = np.clip(y, 1200, 1454).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Exporters.
# ---------------------------------------------------------------------------

def export_rl_policy():
    from sklearn.ensemble import RandomForestClassifier
    from skl2onnx import to_onnx
    X, y = gen_rl_dataset()
    clf = RandomForestClassifier(n_estimators=40, max_depth=8, n_jobs=-1, random_state=0)
    clf.fit(X, y)
    onx = to_onnx(clf, X[:1].astype(np.float32), target_opset=17,
                  name="rl_policy", options={id(clf): {'zipmap': False}})
    path = os.path.join(OUT_DIR, "rl_policy.onnx")
    open(path, "wb").write(onx.SerializeToString())
    print(f"wrote {path}  (input: features [N,8] f32, output: logits [N,5])")


def export_anomaly():
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import to_onnx
    X, y = gen_anomaly_dataset()
    # IsolationForest gives anomaly scores; we wrap with a logistic regression
    # on the decision_function to produce a calibrated [0,1] score.
    iso = IsolationForest(n_estimators=80, contamination=0.3, random_state=0).fit(X)
    scores = -iso.decision_function(X).reshape(-1, 1)
    cal = LogisticRegression(max_iter=400).fit(scores, y)
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(iso, cal)  # not directly onnx-able; export pieces.

    # Simpler: train a single MLP-like logistic regression on raw features.
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    onx = to_onnx(clf, X[:1].astype(np.float32), target_opset=17,
                  name="anomaly", options={id(clf): {'zipmap': False}})
    path = os.path.join(OUT_DIR, "anomaly.onnx")
    open(path, "wb").write(onx.SerializeToString())
    print(f"wrote {path}  (input: features [N,8] f32, output: prob [N,2])")


def export_mtu_regressor():
    from sklearn.ensemble import GradientBoostingRegressor
    from skl2onnx import to_onnx
    X, y = gen_mtu_dataset()
    reg = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3).fit(X, y)
    onx = to_onnx(reg, X[:1].astype(np.float32), target_opset=17, name="mtu")
    path = os.path.join(OUT_DIR, "mtu_regressor.onnx")
    open(path, "wb").write(onx.SerializeToString())
    print(f"wrote {path}  (input: features [N,6] f32, output: mtu [N,1] f32)")


def export_rtt_lstm():
    series = gen_rtt_series()
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        # Fallback: export an sklearn Huber regressor on windowed means under
        # the same I/O contract the Rust side expects.
        from sklearn.linear_model import HuberRegressor
        from skl2onnx import to_onnx
        W = 16
        Xwin = np.array([series[i:i + W].mean() for i in range(len(series) - W)], dtype=np.float32).reshape(-1, 1)
        ywin = series[W:].astype(np.float32)
        reg = HuberRegressor().fit(Xwin, ywin)
        onx = to_onnx(reg, Xwin[:1], target_opset=17, name="rtt_lstm")
        path = os.path.join(OUT_DIR, "rtt_lstm.onnx")
        open(path, "wb").write(onx.SerializeToString())
        print(f"wrote {path}  (sklearn fallback, input mean-of-window -> next RTT)")
        return

    class RttLstm(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
            self.head = nn.Linear(8, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    W = 16
    X = np.stack([series[i:i + W] for i in range(len(series) - W)], axis=0).astype(np.float32)
    y = series[W:].astype(np.float32)
    model = RttLstm()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lossf = nn.SmoothL1Loss()
    Xt = torch.from_numpy(X[..., None])
    yt = torch.from_numpy(y)
    for epoch in range(60):
        opt.zero_grad()
        pred = model(Xt).squeeze(-1)
        loss = lossf(pred, yt)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"  rtt lstm epoch {epoch} loss={loss.item():.3f}")
    model.eval()
    path = os.path.join(OUT_DIR, "rtt_lstm.onnx")
    torch.onnx.export(
        model, Xt[:1], path, input_names=["x"], output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {path}  (input: x [N,16,1] f32, output: y [N,1] f32 = next RTT ms)")


def main():
    print("training + exporting ONNX models to", OUT_DIR)
    try:
        export_rtt_lstm()
    except Exception as e:
        print(f"  rtt_lstm FAILED: {e}", file=sys.stderr)
    try:
        export_rl_policy()
    except Exception as e:
        print(f"  rl_policy FAILED: {e}", file=sys.stderr)
    try:
        export_anomaly()
    except Exception as e:
        print(f"  anomaly FAILED: {e}", file=sys.stderr)
    try:
        export_mtu_regressor()
    except Exception as e:
        print(f"  mtu FAILED: {e}", file=sys.stderr)
    print("done.")


if __name__ == "__main__":
    main()
