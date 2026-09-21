import numpy as np
import pandas as pd

from portfolio_backtest.optimize import optimize_weights, performance_stats, run_backtest


def make_prices(periods=520):
    dates = pd.bdate_range("2022-01-03", periods=periods)
    rng = np.random.default_rng(7)
    asset_returns = pd.DataFrame(
        rng.normal([0.0005, 0.0003, 0.0002, 0.0004], [0.008, 0.006, 0.005, 0.007], (periods, 4)),
        index=dates,
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    benchmark_returns = pd.Series(
        rng.normal(0.00035, 0.007, periods), index=dates, name="^GSPC"
    )
    return (1 + asset_returns).cumprod(), (1 + benchmark_returns).cumprod()


def test_optimizer_returns_fully_invested_capped_weights():
    prices, _ = make_prices()
    weights, stats = optimize_weights(prices.pct_change().dropna())

    assert np.isclose(weights.sum(), 1)
    assert (weights >= 0).all()
    assert (weights <= 0.30 + 1e-8).all()
    assert len(stats) == 3


def test_performance_stats_has_expected_fields():
    returns = pd.Series([0.01, -0.005, 0.002, 0.003])
    stats = performance_stats(returns, risk_free_rate=0)

    assert set(stats) == {"CAGR", "Volatility", "Sharpe", "Max Drawdown"}
    assert stats["Max Drawdown"] < 0


def test_backtest_optimizes_before_test_window():
    prices, benchmark = make_prices()
    result = run_backtest(prices, benchmark, split="2023-01-01")

    assert result.nav.index.min() >= pd.Timestamp("2023-01-01")
    assert list(result.test_stats.index) == ["Optimized", "S&P 500"]
    assert np.isclose(result.allocation.sum(), 1)


def test_backtest_applies_initial_transaction_cost():
    prices, benchmark = make_prices()
    without_cost = run_backtest(prices, benchmark, split="2023-01-01", transaction_cost_bps=0)
    with_cost = run_backtest(prices, benchmark, split="2023-01-01", transaction_cost_bps=100)

    assert with_cost.nav.iloc[0, 0] < without_cost.nav.iloc[0, 0]