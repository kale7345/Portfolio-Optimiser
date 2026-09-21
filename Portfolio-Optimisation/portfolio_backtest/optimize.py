"""Download data, find a maximum-Sharpe allocation, and backtest it."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "AMZN"]
BENCHMARK = "^GSPC"
START = "2020-01-01"
END = "2025-12-31"
SPLIT = "2024-01-01"
RISK_FREE_RATE = 0.045
MAX_WEIGHT = 0.30
TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestResult:
    allocation: pd.Series
    train_stats: tuple[float, float, float]
    test_stats: pd.DataFrame
    nav: pd.DataFrame


def download_prices(
    tickers: list[str], benchmark: str = BENCHMARK,
    start: str = START, end: str = END,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted close prices and remove incomplete asset rows."""
    symbols = list(dict.fromkeys([*tickers, benchmark]))
    data = yf.download(
        symbols, start=start, end=end, auto_adjust=True, progress=False,
    )
    if data.empty:
        raise ValueError("No price data was downloaded.")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    missing = set(symbols) - set(close.columns)
    if missing:
        raise ValueError(f"Missing downloaded symbols: {sorted(missing)}")
    prices = close[tickers].dropna()
    bench = close[benchmark].dropna()
    if prices.empty or bench.empty:
        raise ValueError("Downloaded data contains no complete observations.")
    return prices, bench


def annualized_inputs(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Convert daily simple returns into annualized mean and covariance."""
    if returns.empty or returns.isna().any().any():
        raise ValueError("Returns must be non-empty and contain no NaNs.")
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def portfolio_stats(
    weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple[float, float, float]:
    """Return annualized return, volatility, and Sharpe ratio."""
    expected_return = float(weights @ mu.to_numpy())
    volatility = float(np.sqrt(weights @ cov.to_numpy() @ weights))
    if volatility <= 0:
        raise ValueError("Portfolio volatility must be positive.")
    sharpe = (expected_return - risk_free_rate) / volatility
    return expected_return, volatility, float(sharpe)


def optimize_weights(
    returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float | None = MAX_WEIGHT,
) -> tuple[pd.Series, tuple[float, float, float]]:
    """Find the long-only maximum-Sharpe allocation using annualized inputs."""
    mu, cov = annualized_inputs(returns)
    asset_count = len(returns.columns)
    upper_bound = 1.0 if max_weight is None else max_weight
    if upper_bound * asset_count < 1:
        required = 1 / asset_count
        raise ValueError(
            f"max_weight={upper_bound} is too small for {asset_count} assets; "
            f"a fully invested long-only portfolio requires max_weight >= {required:.6f}."
        )
    initial = np.full(asset_count, 1 / asset_count)
    result = minimize(
        lambda weights: -portfolio_stats(weights, mu, cov, risk_free_rate)[2],
        initial,
        method="SLSQP",
        bounds=[(0, upper_bound)] * asset_count,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    weights = pd.Series(result.x, index=returns.columns, name="Weight")
    return weights.sort_values(ascending=False), portfolio_stats(result.x, mu, cov, risk_free_rate)


def performance_stats(
    daily_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """Calculate CAGR, volatility, Sharpe, and maximum drawdown."""
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        raise ValueError("Cannot calculate performance for empty returns.")
    nav = (1 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS
    cagr = float(nav.iloc[-1] ** (1 / years) - 1)
    volatility = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((daily_returns.mean() * TRADING_DAYS - risk_free_rate) / volatility)
    drawdown = nav / nav.cummax() - 1
    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": float(drawdown.min()),
    }


def run_backtest(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    split: str = SPLIT,
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float | None = MAX_WEIGHT,
    transaction_cost_bps: float = 10,
) -> BacktestResult:
    """Optimize on the training window and evaluate only on the test window."""
    returns = prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    train = returns.loc[returns.index < split]
    test = returns.loc[returns.index >= split]
    if train.empty or test.empty:
        raise ValueError("Both train and test windows must contain observations.")
    allocation, train_stats = optimize_weights(train, risk_free_rate, max_weight)
    portfolio_test = test @ allocation.reindex(test.columns).to_numpy()
    test_index = portfolio_test.index.intersection(benchmark_returns.index)
    portfolio_test = portfolio_test.loc[test_index]
    benchmark_test = benchmark_returns.loc[test_index]
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    if not portfolio_test.empty:
        initial_cost = transaction_cost_bps / 10_000
        portfolio_test.iloc[0] = (1 + portfolio_test.iloc[0]) * (1 - initial_cost) - 1
    nav = pd.DataFrame({
        "Optimized": (1 + portfolio_test).cumprod(),
        "S&P 500": (1 + benchmark_test).cumprod(),
    })
    stats = pd.DataFrame({
        "Optimized": performance_stats(portfolio_test, risk_free_rate),
        "S&P 500": performance_stats(benchmark_test, risk_free_rate),
    }).T
    return BacktestResult(allocation, train_stats, stats, nav)


def save_chart(nav: pd.DataFrame, path: str | Path = "backtest.png") -> None:
    """Save cumulative out-of-sample growth of $1."""
    figure, axis = plt.subplots(figsize=(10, 5))
    nav.plot(ax=axis, linewidth=2)
    axis.set_title("Out-of-sample backtest (2024-2025)")
    axis.set_ylabel("Growth of $1")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("backtest.png"))
    args = parser.parse_args()

    prices, benchmark = download_prices(DEFAULT_TICKERS)
    result = run_backtest(prices, benchmark)
    print(f"Downloaded {len(prices)} complete price observations.")
    print("\nTrain-period maximum-Sharpe allocation:")
    print(result.allocation.round(3).to_string())
    expected_return, volatility, sharpe = result.train_stats
    print(f"\nTrain expected return {expected_return:.1%} | volatility {volatility:.1%} | Sharpe {sharpe:.2f}")
    print("\nOut-of-sample test statistics:")
    print(result.test_stats.to_string(float_format=lambda value: f"{value:.2%}"))
    save_chart(result.nav, args.output)
    print(f"\nChart saved to {args.output}")


if __name__ == "__main__":
    main()
