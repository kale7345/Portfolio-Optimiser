# Portfolio-Optimiser
This optimizer is a constrained mean variance Sharpe maximization: it estimates asset returns and covariance, then solves for the best long only, fully invested portfolio under a per asset cap that maximizes risk adjusted return.

The project is designed around a long-only, fully invested portfolio with a 30% cap, which is enforced in optimize.py. That means your custom asset list must be compatible with that constraint. For instance, if you use only 3 assets, a 30% cap cannot fully invest the portfolio.

If you want to use your own assets replace the default ticker list:
from portfolio_backtest.optimize import download_prices, run_backtest

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
prices, benchmark = download_prices(tickers)
result = run_backtest(prices, benchmark)
print(result.allocation)

How this optimizer works
The optimizer in optimize.py is a classic long-only maximum Sharpe portfolio construction.

1. It starts from return data
The function optimize_weights() expects a DataFrame of daily asset returns.

It does:

mu, cov = annualized_inputs(returns)
computes:
mu: annualized expected return for each asset
cov: annualized covariance matrix of asset returns
This is done by converting daily returns into annualized values using:

mean return × 252
covariance × 252
So the optimization is based on annualized expected returns and risk, not raw daily values.

2. It enforces realistic portfolio constraints
The optimizer is intentionally constrained:

no shorting
fully invested
max position size per asset is 30%
In code:

bounds are set to (0, upper_bound) for each asset
the sum of weights must equal 1
So each weight must satisfy:

𝑤
𝑖
≥
0
w 
i
​
 ≥0
𝑤
𝑖
≤
0.30
w 
i
​
 ≤0.30
∑
𝑖
𝑤
𝑖
=
1
∑ 
i
​
 w 
i
​
 =1
This is why the validation check exists:

if there are too few assets, a 30% cap cannot sum to 100%
for example, 3 assets at 30% each only totals 90%
That is the exact reason the earlier failure happened.

3. It maximizes Sharpe ratio
The optimization objective is:

minimize negative Sharpe ratio
The code uses scipy.optimize.minimize with SLSQP:


The function portfolio_stats() computes:

expected return
volatility

The optimizer is therefore trying to find the portfolio with the highest risk-adjusted return subject to the constraints.

4. It returns the allocation
When the optimization succeeds, it returns:

a pandas Series of weights sorted descending
a tuple of:
expected return
volatility
Sharpe ratio
This lets you see the final trade-off in a simple summary.

5. It is used inside the backtest
In optimize.py, run_backtest():

splits returns into:
train period
test period
fits the optimizer on the train data
applies the resulting weights to the test-period returns
evaluates performance versus the S&P 500
So the process is:

estimate portfolio using historical training data
hold that allocation on out-of-sample data
compare realized portfolio returns with benchmark performance
6. The cost and performance logic
The same file also computes metrics like:

CAGR
volatility
Sharpe
max drawdown
And it applies an initial transaction cost:


This models the idea that buying into the portfolio at the beginning incurs a small cost.
