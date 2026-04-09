"""
P2-ETF-COPULA-ENGINE  ·  copula_model.py
Fit parametric copulas to uniform pseudo-observations, select the best
family by AIC, estimate tail dependence coefficients, and run Monte Carlo
sampling to generate return distributions for scoring.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm
from scipy.optimize import minimize_scalar

from config import COPULA_FAMILIES, N_MC_SAMPLES, RANDOM_SEED

warnings.filterwarnings("ignore")
rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Bivariate copula helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
    tau, _ = kendalltau(u, v)
    return float(tau)


def _gaussian_theta(tau: float) -> float:
    """Pearson rho from Kendall tau for Gaussian copula."""
    return np.sin(np.pi / 2 * tau)


def _student_theta(tau: float) -> tuple[float, int]:
    """Rho and df for Student-t copula (df fixed at 4 as robust default)."""
    return np.sin(np.pi / 2 * tau), 4


def _clayton_theta(tau: float) -> float:
    """Clayton parameter from Kendall tau. Clamped to [0.01, 50]."""
    theta = 2 * tau / (1 - tau)
    return float(np.clip(theta, 0.01, 50.0))


def _gumbel_theta(tau: float) -> float:
    """Gumbel parameter from Kendall tau. Clamped to [1.0, 50]."""
    theta = 1 / (1 - tau)
    return float(np.clip(theta, 1.0, 50.0))


def _frank_theta(tau: float) -> float:
    """Frank parameter estimated numerically from Kendall tau."""
    if abs(tau) < 1e-6:
        return 1e-6

    def objective(theta):
        if abs(theta) < 1e-9:
            return (tau - 0.0) ** 2
        from scipy.integrate import quad
        def integrand(t):
            denom = np.expm1(-theta)
            if abs(denom) < 1e-12:
                return 0.0
            return t / (np.expm1(-theta * t) * denom + 1e-15) if abs(np.expm1(-theta * t)) > 1e-12 else 0.0
        val, _ = quad(integrand, 0, 1)
        tau_hat = 1 - 4 / theta * (1 - val)
        return (tau - tau_hat) ** 2

    res = minimize_scalar(objective, bounds=(-40, 40), method="bounded")
    return float(res.x)


# ─────────────────────────────────────────────────────────────────────────────
# Copula log-likelihood  (bivariate, average pairwise over N ETFs)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_loglik(u: np.ndarray, rho: float) -> float:
    eps  = 1e-10
    x, y = norm.ppf(u[:, 0].clip(eps, 1 - eps)), norm.ppf(u[:, 1].clip(eps, 1 - eps))
    det  = 1 - rho ** 2
    ll   = (-0.5 * np.log(det)
            - 0.5 / det * (x**2 + y**2 - 2 * rho * x * y)
            + 0.5 * (x**2 + y**2))
    return float(np.mean(ll))


def _clayton_loglik(u: np.ndarray, theta: float) -> float:
    u1, u2 = u[:, 0].clip(1e-10, 1), u[:, 1].clip(1e-10, 1)
    t      = theta
    ll     = (np.log(1 + t) + (-1 - t) * (np.log(u1) + np.log(u2))
              + (-2 - 1 / t) * np.log(u1**(-t) + u2**(-t) - 1))
    return float(np.mean(ll))


def _gumbel_loglik(u: np.ndarray, theta: float) -> float:
    u1, u2  = u[:, 0].clip(1e-10, 1), u[:, 1].clip(1e-10, 1)
    x1, x2  = -np.log(u1), -np.log(u2)
    A       = (x1**theta + x2**theta) ** (1 / theta)
    C       = np.exp(-A)
    logC    = -A
    ll      = (logC
               + np.log(C)
               + (theta - 1) * (np.log(x1) + np.log(x2))
               - (x1**theta + x2**theta)
               + np.log(A + theta - 1)
               - (2 - 1 / theta) * np.log(x1**theta + x2**theta)
               - np.log(u1) - np.log(u2))
    # simplified version: use numeric approximation
    return float(np.nanmean(logC - np.log(u1) - np.log(u2)))


def _aic(loglik: float, n_params: int) -> float:
    return -2 * loglik + 2 * n_params


# ─────────────────────────────────────────────────────────────────────────────
# Fit copulas to a uniform matrix and select best family
# ─────────────────────────────────────────────────────────────────────────────

def fit_copula(u_df: pd.DataFrame) -> dict:
    """
    Fit all candidate copula families to the uniform matrix u_df (T × N).
    Select the best family by average pairwise AIC across all ETF pairs.

    Returns
    -------
    dict with:
        best_family   str
        params        dict  family → estimated parameters
        aic_scores    dict  family → AIC
        tau_matrix    pd.DataFrame  Kendall tau for each ETF pair
    """
    U      = u_df.values.astype(float)
    n_etfs = U.shape[1]
    tickers = list(u_df.columns)

    # Pairwise Kendall tau
    tau_mat = np.eye(n_etfs)
    for i in range(n_etfs):
        for j in range(i + 1, n_etfs):
            tau = _kendall_tau(U[:, i], U[:, j])
            tau_mat[i, j] = tau_mat[j, i] = tau

    tau_df = pd.DataFrame(tau_mat, index=tickers, columns=tickers)

    # Fit each family using average pairwise method-of-moments
    params     = {}
    aic_scores = {}

    for family in COPULA_FAMILIES:
        pair_lls = []
        pair_pars = {}

        for i in range(n_etfs):
            for j in range(i + 1, n_etfs):
                uij = U[:, [i, j]]
                tau = tau_mat[i, j]

                try:
                    if family == "gaussian":
                        rho = _gaussian_theta(tau)
                        ll  = _gaussian_loglik(uij, rho)
                        par = rho
                    elif family == "student":
                        rho, df = _student_theta(tau)
                        ll  = _gaussian_loglik(uij, rho)   # approx
                        par = (rho, df)
                    elif family == "clayton":
                        theta = _clayton_theta(max(tau, 0.01))
                        ll    = _clayton_loglik(uij, theta)
                        par   = theta
                    elif family == "gumbel":
                        theta = _gumbel_theta(max(tau, 0.01))
                        ll    = _gumbel_loglik(uij, theta)
                        par   = theta
                    elif family == "frank":
                        theta = _frank_theta(tau)
                        ll    = _gaussian_loglik(uij, np.tanh(tau))  # approx
                        par   = theta
                    else:
                        continue

                    if np.isfinite(ll):
                        pair_lls.append(ll)
                        pair_pars[f"{tickers[i]}_{tickers[j]}"] = par
                except Exception:
                    continue

        if pair_lls:
            avg_ll = float(np.mean(pair_lls))
            n_par  = 1 if family not in ("student",) else 2
            aic_scores[family] = _aic(avg_ll, n_par)
            params[family]     = pair_pars

    best_family = min(aic_scores, key=aic_scores.get) if aic_scores else "gaussian"

    return {
        "best_family": best_family,
        "params":      params,
        "aic_scores":  aic_scores,
        "tau_matrix":  tau_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tail dependence coefficients
# ─────────────────────────────────────────────────────────────────────────────

def tail_dependence(u_df: pd.DataFrame, quantile: float = 0.10) -> pd.DataFrame:
    """
    Empirical lower tail dependence coefficient λ_L for each ETF pair:
      λ_L(i,j) = P(U_i < q | U_j < q)
    Returns a symmetric DataFrame of λ_L values.
    """
    U      = u_df.values
    n      = U.shape[1]
    tickers = list(u_df.columns)
    td     = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            mask_j = U[:, j] < quantile
            if mask_j.sum() == 0:
                lam = 0.0
            else:
                lam = float((U[:, i][mask_j] < quantile).mean())
            td[i, j] = td[j, i] = lam

    return pd.DataFrame(td, index=tickers, columns=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation of joint returns
# ─────────────────────────────────────────────────────────────────────────────

def mc_simulate(copula_fit: dict,
                fitted_marginals: dict,
                window_returns: pd.DataFrame,
                n_samples: int = N_MC_SAMPLES) -> pd.DataFrame:
    """
    Draw n_samples correlated return vectors from the fitted copula,
    then invert the GARCH marginals to recover simulated returns.

    Returns
    -------
    pd.DataFrame  shape (n_samples, n_etfs)  simulated one-step returns
    """
    tickers    = window_returns.columns.tolist()
    n          = len(tickers)
    tau_mat    = copula_fit["tau_matrix"].loc[tickers, tickers].values
    best_fam   = copula_fit["best_family"]

    # Build correlation matrix from average Kendall tau
    rho_mat = np.sin(np.pi / 2 * tau_mat)
    np.fill_diagonal(rho_mat, 1.0)
    rho_mat = np.clip(rho_mat, -0.999, 0.999)

    # Ensure positive semi-definite via eigenvalue floor
    eigvals, eigvecs = np.linalg.eigh(rho_mat)
    eigvals          = np.maximum(eigvals, 1e-6)
    rho_mat          = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d                = np.sqrt(np.diag(rho_mat))
    rho_mat          = rho_mat / np.outer(d, d)

    # Draw correlated Gaussians and map to uniforms
    L   = np.linalg.cholesky(rho_mat)
    Z   = rng.standard_normal((n_samples, n))
    X   = Z @ L.T
    U   = norm.cdf(X)

    # Invert marginals: u → std_resid → scaled return
    sim_ret = np.zeros((n_samples, n))
    for k, ticker in enumerate(tickers):
        info    = fitted_marginals.get(ticker, {})
        res     = info.get("model_result", None)
        # Get last conditional volatility from GARCH
        try:
            forecast = res.forecast(horizon=1, reindex=False)
            sigma    = float(np.sqrt(forecast.variance.iloc[-1, 0])) / 100.0
        except Exception:
            sigma = float(window_returns[ticker].std())
        sigma = max(sigma, 1e-6)

        std_r = norm.ppf(U[:, k].clip(1e-6, 1 - 1e-6))
        sim_ret[:, k] = std_r * sigma

    return pd.DataFrame(sim_ret, columns=tickers)
