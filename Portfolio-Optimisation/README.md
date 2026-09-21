# Portfolio Optimization & Backtesting

A small Python project that downloads adjusted prices, estimates a long-only maximum-Sharpe allocation, and evaluates it out of sample against the S&P 500.

## Method

- Assets: `AAPL`, `MSFT`, `NVDA`, `JPM`, `XOM`, `JNJ`, and `AMZN`
- Data: Yahoo Finance adjusted close prices from 2020-01-01 through 2025-12-31
- Training window: 2020-01-01 through 2023-12-31
- Test window: 2024-01-01 through 2025-12-31
- Optimization: annualized daily means and covariance, no shorting, fully invested, and a 30% maximum position size
- Risk-free rate: 4.5%
- Costs: 10 bps are charged on the initial portfolio purchase; production use should also model turnover and periodic rebalancing

The 30% cap is intentional. An uncapped optimizer can put nearly all capital in one or two assets because it is optimizing noisy historical estimates. The cap makes the example more diversified and is easier to defend in a real investment process.

## Run

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the backtest:

```powershell
python -m portfolio_backtest.optimize
```

This prints the train-period allocation and test-period CAGR, volatility, Sharpe ratio, and maximum drawdown. It also writes `backtest.png` with growth of $1 for the optimized portfolio and the S&P 500.

Run the deterministic tests:

```powershell
pytest
```

## Interpreting the result

The printed comparison answers whether the optimized portfolio beat the S&P 500 during 2024-2025 after being fit only on 2020-2023. That answer can change as Yahoo Finance data changes. A win is not proof of skill: the sample is short, parameters are estimated with noise, and this simple backtest does not model ongoing rebalancing drift.