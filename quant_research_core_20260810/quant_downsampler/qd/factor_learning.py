"""Two factor-only baselines: return regression and direct portfolio-Sharpe learning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import OUTPUT_DIR


DAILY_FACTOR_NAMES = (
    "example_factor",
    "momentum_5d", "momentum_10d", "momentum_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "buy_sell_imbalance", "intraday_range", "volume_price_corr_20d",
)


def load_factor_panel() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return standardized daily factor panels and next-day returns without look-ahead."""
    daily = OUTPUT_DIR / "daily"
    factors_dir = OUTPUT_DIR / "factors"
    close = pd.read_csv(daily / "close.csv", index_col=0).astype(float)
    open_price = pd.read_csv(daily / "open.csv", index_col=0).astype(float).reindex_like(close)
    volume = pd.read_csv(daily / "volume.csv", index_col=0).astype(float).reindex_like(close)
    # Daily matrices use YYYYMMDD while saved factor files use ISO dates.
    close.index = pd.to_datetime(close.index.astype(str), format="mixed").strftime("%Y%m%d")
    open_price.index = close.index
    volume.index = close.index
    dates, stocks = close.index.astype(str).tolist(), close.columns.tolist()
    # Factor at close t is executed at open t+1 and held to open t+2.
    forward = open_price.shift(-2).div(open_price.shift(-1)).sub(1.0).to_numpy(np.float32)
    valid = (
        np.isfinite(forward)
        & (volume.shift(-1).to_numpy() > 0)
        & (volume.shift(-2).to_numpy() > 0)
    )
    arrays = []
    for name in DAILY_FACTOR_NAMES:
        f = pd.read_csv(factors_dir / f"{name}.csv", index_col=0)
        f.index = pd.to_datetime(f.index).strftime("%Y%m%d")
        f = f.reindex(index=close.index, columns=close.columns).astype(float)
        values = f.to_numpy(np.float32)
        # Cross-sectional winsorization and z-score.  Each row uses data available at its close only.
        normalized = np.zeros_like(values, dtype=np.float32)
        valid_rows = np.isfinite(values).any(axis=1)
        usable_values = values[valid_rows]
        median = np.nanmedian(usable_values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(usable_values - median), axis=1, keepdims=True)
        clipped = np.clip(
            usable_values, median - 5 * (mad + 1e-8), median + 5 * (mad + 1e-8),
        )
        mean = np.nanmean(clipped, axis=1, keepdims=True)
        std = np.nanstd(clipped, axis=1, keepdims=True)
        normalized[valid_rows] = np.nan_to_num(
            (clipped - mean) / (std + 1e-8), nan=0.0, posinf=0.0, neginf=0.0,
        )
        arrays.append(normalized)
    return np.stack(arrays, axis=-1), forward, valid, dates, stocks


def _weights(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Dollar-neutral, unit-gross-exposure portfolio weights per day."""
    scores = scores * mask
    scores = scores - (scores.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)) * mask
    raw = torch.tanh(scores) * mask
    return raw / raw.abs().sum(1, keepdim=True).clamp_min(1e-8)


def _portfolio_metrics(scores: np.ndarray, returns: np.ndarray, mask: np.ndarray, cost_bps: float) -> tuple[dict, pd.DataFrame]:
    score = np.where(mask, scores, 0.0)
    score -= np.divide(score.sum(1, keepdims=True), mask.sum(1, keepdims=True), out=np.zeros_like(score), where=mask.sum(1, keepdims=True) > 0) * mask
    weights = np.tanh(score) * mask
    weights /= np.maximum(np.abs(weights).sum(1, keepdims=True), 1e-8)
    gross = np.nansum(weights * np.where(mask, returns, 0.0), axis=1)
    turnover = np.r_[0.0, np.abs(weights[1:] - weights[:-1]).sum(1) / 2]
    net = gross - turnover * cost_bps / 1e4
    annual_sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252)) if net.std(ddof=1) else 0.0
    metrics = {"annualized_sharpe_net": annual_sharpe, "mean_daily_net_bp": float(net.mean()*1e4),
               "daily_volatility_bp": float(net.std(ddof=1)*1e4), "average_turnover": float(turnover.mean()),
               "annualized_return_net": float((1 + net).prod() ** (252 / len(net)) - 1),
               "max_drawdown": float((np.cumprod(1 + net) / np.maximum.accumulate(np.cumprod(1 + net)) - 1).min())}
    return metrics, pd.DataFrame({"gross_return": gross, "turnover": turnover, "net_return": net, "nav": np.cumprod(1 + net)})


def run_factor_models(epochs: int = 300, learning_rate: float = .03, cost_bps: float = 5.0,
                      out_dir: Path | None = None, seed: int = 42) -> dict:
    """Fit a next-day return model and a direct Sharpe-maximizing factor portfolio."""
    np.random.seed(seed); torch.manual_seed(seed)
    out_dir = out_dir or OUTPUT_DIR / "factor_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, mask, dates, _stocks = load_factor_panel()
    # Last day has no forward return.  Chronological 70/15/15 split prevents leakage.
    usable = len(dates) - 2; train_end, val_end = int(usable*.70), int(usable*.85)
    splits = {"train": slice(0, train_end), "validation": slice(train_end, val_end), "test": slice(val_end, usable)}
    tx = torch.tensor(x[:usable]); ty = torch.tensor(np.nan_to_num(y[:usable] * 100.0)); tm = torch.tensor(mask[:usable])
    train_x, train_y, train_m = tx[:train_end], ty[:train_end], tm[:train_end]

    # Model A: infer individual next-day returns from the ten factors.
    reg = torch.nn.Linear(x.shape[-1], 1, bias=True)
    opt = torch.optim.AdamW(reg.parameters(), lr=learning_rate, weight_decay=.05)
    for _ in range(epochs):
        pred = reg(train_x).squeeze(-1)
        loss = ((pred - train_y).square() * train_m).sum() / train_m.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): reg_scores = reg(tx).squeeze(-1).numpy()

    # Model B: only learn factor weights, but optimize the actual net portfolio Sharpe.
    sharpe = torch.nn.Linear(x.shape[-1], 1, bias=False)
    opt = torch.optim.AdamW(sharpe.parameters(), lr=learning_rate, weight_decay=.01)
    history = []
    for _ in range(epochs):
        scores = sharpe(train_x).squeeze(-1)
        w = _weights(scores, train_m)
        gross = (w * train_y / 100.0).sum(1)
        turnover = torch.cat((torch.zeros(1), (w[1:] - w[:-1]).abs().sum(1) / 2))
        net = gross - turnover * cost_bps / 1e4
        objective = net.mean() / net.std(unbiased=True).clamp_min(1e-8)
        loss = -objective
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(float(objective.detach()))
    with torch.no_grad(): sharpe_scores = sharpe(tx).squeeze(-1).numpy()

    result = {"settings": {"cost_bps": cost_bps, "epochs": epochs, "split": "70% train / 15% validation / 15% test"}, "models": {}}
    for model_name, scores in (("return_regression", reg_scores), ("direct_sharpe", sharpe_scores)):
        result["models"][model_name] = {}
        for split_name, index in splits.items():
            metrics, pnl = _portfolio_metrics(scores[index], y[:usable][index], mask[:usable][index], cost_bps)
            pnl.index = dates[:usable][index]
            pnl.to_csv(out_dir / f"{model_name}_{split_name}_daily.csv", float_format="%.8f")
            result["models"][model_name][split_name] = metrics
    result["factor_weights"] = {"return_regression": dict(zip(DAILY_FACTOR_NAMES, reg.weight.detach().numpy().ravel().tolist())),
                                "direct_sharpe": dict(zip(DAILY_FACTOR_NAMES, sharpe.weight.detach().numpy().ravel().tolist()))}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "sharpe_training_history.json").write_text(json.dumps(history), encoding="utf-8")
    _plot_results(out_dir)
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))
    return result


def run_rolling_factor_models(train_window: int = 120, rebalance_days: int = 20,
                              epochs: int = 120, learning_rate: float = .03,
                              cost_bps: float = 5.0, out_dir: Path | None = None,
                              seed: int = 42) -> dict:
    """Walk-forward re-estimation: each block is fit using only preceding dates."""
    np.random.seed(seed); torch.manual_seed(seed)
    out_dir = out_dir or OUTPUT_DIR / "factor_models_rolling"
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, mask, dates, _stocks = load_factor_panel()
    usable = len(dates) - 2
    tx = torch.tensor(x[:usable]); ty = torch.tensor(np.nan_to_num(y[:usable] * 100.0)); tm = torch.tensor(mask[:usable])
    result_scores = {"return_regression": np.zeros((usable - train_window, x.shape[1]), dtype=np.float32),
                     "direct_sharpe": np.zeros((usable - train_window, x.shape[1]), dtype=np.float32)}
    fitted_weights = []
    for start in range(train_window, usable, rebalance_days):
        end = min(start + rebalance_days, usable)
        hx, hy, hm = tx[start - train_window:start], ty[start - train_window:start], tm[start - train_window:start]
        # Return regression, trained strictly on the trailing window.
        reg = torch.nn.Linear(x.shape[-1], 1)
        opt = torch.optim.AdamW(reg.parameters(), lr=learning_rate, weight_decay=.05)
        for _ in range(epochs):
            loss = ((reg(hx).squeeze(-1) - hy).square() * hm).sum() / hm.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
        # Direct net-Sharpe optimizer on exactly the same historical window.
        direct = torch.nn.Linear(x.shape[-1], 1, bias=False)
        opt = torch.optim.AdamW(direct.parameters(), lr=learning_rate, weight_decay=.01)
        for _ in range(epochs):
            w = _weights(direct(hx).squeeze(-1), hm)
            gross = (w * hy / 100.0).sum(1)
            turnover = torch.cat((torch.zeros(1), (w[1:] - w[:-1]).abs().sum(1) / 2))
            net = gross - turnover * cost_bps / 1e4
            loss = -(net.mean() / net.std(unbiased=True).clamp_min(1e-8))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            block = slice(start - train_window, end - train_window)
            result_scores["return_regression"][block] = reg(tx[start:end]).squeeze(-1).numpy()
            result_scores["direct_sharpe"][block] = direct(tx[start:end]).squeeze(-1).numpy()
        fitted_weights.append({"rebalance_date": dates[start], "return_regression": reg.weight.detach().numpy().ravel().tolist(),
                               "direct_sharpe": direct.weight.detach().numpy().ravel().tolist()})
        print(f"Walk-forward fit through {dates[start-1]}, applied {dates[start]} to {dates[end-1]}")

    out_y, out_mask, out_dates = y[train_window:usable], mask[train_window:usable], dates[train_window:usable]
    summary = {"settings": {"train_window_days": train_window, "rebalance_days": rebalance_days,
                             "epochs_per_refit": epochs, "cost_bps": cost_bps}, "models": {}}
    recent_start = max(0, len(out_dates) - 45)
    for name, scores in result_scores.items():
        metrics, pnl = _portfolio_metrics(scores, out_y, out_mask, cost_bps)
        pnl.index = out_dates; pnl.to_csv(out_dir / f"{name}_walk_forward_daily.csv", float_format="%.8f")
        recent, _ = _portfolio_metrics(scores[recent_start:], out_y[recent_start:], out_mask[recent_start:], cost_bps)
        summary["models"][name] = {"full_walk_forward": metrics, "last_45_days": recent}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "rolling_weights.json").write_text(json.dumps(fitted_weights, indent=2), encoding="utf-8")
    _plot_rolling_results(out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _plot_results(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, color in (("return_regression", "#1f77b4"), ("direct_sharpe", "#ff7f0e")):
        frame = pd.concat([pd.read_csv(out_dir / f"{name}_{s}_daily.csv", index_col=0) for s in ("train", "validation", "test")])
        dates = pd.to_datetime(frame.index.astype(str), format="mixed")
        ax.plot(dates, frame["nav"], label=name, color=color)
    ax.set(title="Factor portfolio NAV (net of assumed costs)", ylabel="NAV", xlabel="Date"); ax.grid(alpha=.3); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out_dir / "portfolio_nav.png", dpi=150); plt.close(fig)


def _plot_rolling_results(out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, color in (("return_regression", "#1f77b4"), ("direct_sharpe", "#ff7f0e")):
        frame = pd.read_csv(out_dir / f"{name}_walk_forward_daily.csv", index_col=0)
        dates = pd.to_datetime(frame.index.astype(str), format="mixed")
        ax.plot(dates, frame["nav"], label=name, color=color)
    ax.set(title="Walk-forward factor portfolio NAV (net of assumed costs)", ylabel="NAV", xlabel="Date"); ax.grid(alpha=.3); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out_dir / "portfolio_nav.png", dpi=150); plt.close(fig)
