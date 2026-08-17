"""
Money-in-Utility New Keynesian Fiscal Sandbox
============================================

A compact Python implementation of the linearized MIU-NK model discussed with José.

Core objects
------------
1. Standard Galí-style NK private block:
       x_t = E_t x_{t+1} - (1/sigma)(i_t - E_t pi_{t+1} - r_t^n)
       pi_t = beta E_t pi_{t+1} + kappa x_t

2. Separable money-in-utility demand:
       muhat_t = (sigma/nu) yhat_t - eta_i i_t

3. Consolidated government budget constraint with one-period nominal debt:
       beta * btilde_t = btilde_{t-1} + mutilde_{t-1} - mutilde_t
                       + xi_t - (bbar + mubar) pi_t + beta*bbar*i_t

4. Non-Ricardian fiscal regime: primary surplus does not react to debt.

The model solves for linear policy functions in states:
       s_t = (ell_{t-1}, a_t, xi_t)
where ell_t = btilde_t + mutilde_t is total real nominal liabilities in deviations.

The monetary rule is flexible:
       i_t = phi_pi*pi_t + phi_x*x_t + phi_l*ell_{t-1} - phi_xi*xi_t

Notes
-----
- phi_l is a reaction to lagged total nominal public liabilities, not only bonds.
  This is the clean state variable in the model. If desired, it can later be replaced
  by a reaction to lagged bonds, at the cost of adding lagged money as a separate state.
- The code intentionally reports failure or unstable roots rather than forcing a solution.
  Some monetary/fiscal combinations genuinely do not deliver the desired bounded equilibrium.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import root

# %% Parameters


@dataclass
class ModelParams:
    """Primitive and policy parameters for the MIU-NK fiscal sandbox."""

    # Household / NK block
    beta: float = 0.99
    sigma: float = 1.0
    varphi: float = 1.0
    alpha: float = 0.0
    kappa: float = 0.08

    # Money-in-utility
    nu: float = 2.0
    mubar: float = 0.10  # steady real money balances, M/P

    # Fiscal steady state
    bbar: float = 0.60  # steady real one-period nominal bond liabilities

    # Monetary rule: i_t = phi_pi*pi_t + phi_x*x_t + phi_l*ell_{t-1} - phi_xi*xi_t
    phi_pi: float = 0.70  # passive by default, consistent with fiscal-dominance experiments
    phi_x: float = 0.0    # reaction to gdp gap
    phi_l: float = 0.0    # reaction to total liabilities
    phi_xi: float = 0.0   # reaction to deficits

    # Exogenous processes
    rho_a: float = 0.90   # persistence in technology
    rho_xi: float = 0.00  # persistence in fiscal shock

    # Numerical tolerances
    root_tol: float = 1e-10
    uniqueness_tol: float = 1e-7

    def validate(self) -> None:
        """Basic parameter checks."""
        if not (0 < self.beta < 1):
            raise ValueError("beta must lie in (0,1).")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.nu <= 0:
            raise ValueError("nu must be positive.")
        if self.mubar < 0 or self.bbar < 0:
            raise ValueError("mubar and bbar must be non-negative.")
        if not (-1 < self.rho_a < 1):
            raise ValueError("rho_a should lie in (-1,1) for stationarity.")
        if not (-1 < self.rho_xi < 1):
            raise ValueError("rho_xi should lie in (-1,1) for stationarity.")


@dataclass
class Solution:
    """Container for model solution coefficients."""

    params: ModelParams
    success: bool
    status: str
    stable: bool
    selected_root: Optional[Dict[str, float]]
    all_roots: List[Dict[str, float]]
    coefficients: Optional[pd.DataFrame]
    diagnostics: Dict[str, float]

    def require_success(self) -> None:
        if not self.success or self.coefficients is None:
            raise RuntimeError(f"Model has no usable solution. Status: {self.status}")


# -----------------------------------------------------------------------------
# Helper functions: derived primitives and reduced-form coefficients
# -----------------------------------------------------------------------------


def derived_objects(p: ModelParams) -> Dict[str, float]:
    """Compute derived steady-state and natural-rate objects."""
    p.validate()
    Ibar = 1.0 / p.beta
    qbar = p.beta
    Lbar = p.bbar + p.mubar

    # Natural-output elasticity wrt technology in Galí-style no-capital NK model.
    psi = (1.0 + p.varphi) / (p.sigma * (1.0 - p.alpha) + p.varphi + p.alpha)

    # Money-demand semi-elasticity wrt log gross nominal interest rate.
    eta_i = 1.0 / (p.nu * (Ibar - 1.0))

    return {
        "Ibar": Ibar,
        "qbar": qbar,
        "Lbar": Lbar,
        "psi": psi,
        "eta_i": eta_i,
    }


def money_demand_coefficients(p: ModelParams) -> Dict[str, float]:
    """
    Coefficients in the linear money-demand equation:

        mutilde_t = d_x*x_t + d_pi*pi_t + d_l*ell_{t-1} + d_a*a_t + d_xi*xi_t.

    The monetary rule is:

        i_t = phi_pi*pi_t + phi_x*x_t + phi_l*ell_{t-1} - phi_xi*xi_t.
    """
    d = derived_objects(p)
    eta_i = d["eta_i"]
    psi = d["psi"]

    return {
        "d_x": p.mubar * (p.sigma / p.nu - eta_i * p.phi_x),
        "d_pi": -p.mubar * eta_i * p.phi_pi,
        "d_l": -p.mubar * eta_i * p.phi_l,
        "d_a": p.mubar * (p.sigma / p.nu) * psi,
        "d_xi": p.mubar * eta_i * p.phi_xi,
    }


def fiscal_coefficients(p: ModelParams) -> Dict[str, float]:
    """
    Coefficients in the total-liability fiscal equation:

        beta*ell_t = F_l*ell_{t-1} + F_x*x_t + F_pi*pi_t + F_a*a_t + F_xi*xi_t.

    where ell_t = btilde_t + mutilde_t.
    """
    d = derived_objects(p)
    md = money_demand_coefficients(p)

    F_x = p.beta * p.bbar * p.phi_x - (1.0 - p.beta) * md["d_x"]
    F_pi = -d["Lbar"] + p.beta * p.bbar * p.phi_pi - (1.0 - p.beta) * md["d_pi"]
    F_l = 1.0 + p.beta * p.bbar * p.phi_l - (1.0 - p.beta) * md["d_l"]
    F_a = -(1.0 - p.beta) * md["d_a"]
    F_xi = 1.0 - p.beta * p.bbar * p.phi_xi - (1.0 - p.beta) * md["d_xi"]

    return {"F_x": F_x, "F_pi": F_pi, "F_l": F_l, "F_a": F_a, "F_xi": F_xi}


# -----------------------------------------------------------------------------
# Solving the inherited-liability block
# -----------------------------------------------------------------------------


def _liability_equations(vars_: np.ndarray, p: ModelParams) -> np.ndarray:
    """
    Nonlinear equations for response to inherited total nominal liabilities.

    Unknowns are (X_l, Pi_l, L_l), where:
        x_t     = X_l  * ell_{t-1}
        pi_t    = Pi_l * ell_{t-1}
        ell_t   = L_l  * ell_{t-1}

    Equations:
    1. IS, including direct monetary reaction phi_l*ell_{t-1}
    2. NKPC
    3. Government budget constraint
    """
    X_l, Pi_l, L_l = vars_
    F = fiscal_coefficients(p)

    eq_is = (
        (p.sigma * (1.0 - L_l) + p.phi_x) * X_l
        + (p.phi_pi - L_l) * Pi_l
        + p.phi_l
    )
    eq_nkpc = (1.0 - p.beta * L_l) * Pi_l - p.kappa * X_l
    eq_gbc = p.beta * L_l - (F["F_l"] + F["F_x"] * X_l + F["F_pi"] * Pi_l)

    return np.array([eq_is, eq_nkpc, eq_gbc], dtype=float)


def _unique_roots(roots: List[Dict[str, float]], tol: float) -> List[Dict[str, float]]:
    """Drop numerically duplicate roots."""
    unique: List[Dict[str, float]] = []
    for r in roots:
        vec = np.array([r["X_l"], r["Pi_l"], r["L_l"]])
        if not any(np.linalg.norm(vec - np.array([u["X_l"], u["Pi_l"], u["L_l"]])) < tol for u in unique):
            unique.append(r)
    return unique


def solve_liability_block(p: ModelParams, verbose: bool = False) -> List[Dict[str, float]]:
    """Find candidate roots for the inherited-liability block."""
    p.validate()

    # If phi_l == 0, the characteristic equation for L_l can be solved analytically.
    # We still compute X_l and Pi_l using the GBC, which pins down the scale.
    if abs(p.phi_l) < 1e-14:
        F = fiscal_coefficients(p)
        A = p.sigma + p.phi_x
        B = p.beta * A + p.sigma + p.kappa
        C = A + p.kappa * p.phi_pi
        discr = B**2 - 4.0 * p.sigma * p.beta * C

        roots: List[Dict[str, float]] = []
        if discr < -1e-12:
            return []
        discr = max(discr, 0.0)
        for sign in (-1.0, 1.0):
            L_l = (B + sign * np.sqrt(discr)) / (2.0 * p.sigma * p.beta)
            denom = F["F_x"] * (1.0 - p.beta * L_l) / p.kappa + F["F_pi"]
            if abs(denom) < 1e-12:
                continue
            Pi_l = (p.beta * L_l - F["F_l"]) / denom
            X_l = (1.0 - p.beta * L_l) * Pi_l / p.kappa
            residual = np.linalg.norm(_liability_equations(np.array([X_l, Pi_l, L_l]), p))
            roots.append(
                {
                    "X_l": float(X_l),
                    "Pi_l": float(Pi_l),
                    "L_l": float(L_l),
                    "residual": float(residual),
                    "stable": bool(abs(L_l) < 1.0 - 1e-8),
                }
            )
        return _unique_roots(roots, p.uniqueness_tol)

    # With a debt/liability term in the monetary rule, solve the full nonlinear system.
    # Try a broad but modest set of initial guesses.
    guesses: List[Tuple[float, float, float]] = []
    for X0 in (-5, -1, -0.1, 0, 0.1, 1, 5):
        for P0 in (-5, -1, -0.1, 0.1, 1, 5):
            for L0 in (-0.8, -0.2, 0.2, 0.8, 1.2):
                guesses.append((X0, P0, L0))

    roots = []
    for g in guesses:
        sol = root(lambda v: _liability_equations(v, p), np.array(g, dtype=float), tol=p.root_tol)
        if not sol.success:
            continue
        X_l, Pi_l, L_l = sol.x
        residual = np.linalg.norm(_liability_equations(sol.x, p))
        if residual < 1e-7 and np.all(np.isfinite(sol.x)):
            roots.append(
                {
                    "X_l": float(X_l),
                    "Pi_l": float(Pi_l),
                    "L_l": float(L_l),
                    "residual": float(residual),
                    "stable": bool(abs(L_l) < 1.0 - 1e-8),
                }
            )

    roots = _unique_roots(roots, p.uniqueness_tol)
    roots.sort(key=lambda r: (not r["stable"], abs(r["L_l"]), r["residual"]))

    if verbose:
        print(pd.DataFrame(roots))
    return roots


# -----------------------------------------------------------------------------
# Shock coefficients and full solution
# -----------------------------------------------------------------------------


def _solve_shock_coefficients(
    p: ModelParams,
    X_l: float,
    Pi_l: float,
    rho_z: float,
    r_z: float,
    p_z: float,
    F_z: float,
) -> Tuple[float, float, float]:
    """
    Solve coefficients (X_z, Pi_z, L_z) for an AR(1) shock z_t.

    z_t affects:
      - natural rate with coefficient r_z
      - monetary rule directly with coefficient p_z
      - fiscal equation directly with coefficient F_z

    The linear system is:

        [sigma(1-rho)+phi_x, phi_pi-rho, -(sigma*X_l+Pi_l)] [X_z ] = [r_z-p_z]
        [-kappa,              1-beta*rho, -beta*Pi_l       ] [Pi_z]   [0      ]
        [-F_x,                -F_pi,       beta             ] [L_z ]   [F_z    ]
    """
    F = fiscal_coefficients(p)
    A = np.array(
        [
            [p.sigma * (1.0 - rho_z) + p.phi_x, p.phi_pi - rho_z, -(p.sigma * X_l + Pi_l)],
            [-p.kappa, 1.0 - p.beta * rho_z, -p.beta * Pi_l],
            [-F["F_x"], -F["F_pi"], p.beta],
        ],
        dtype=float,
    )
    b = np.array([r_z - p_z, 0.0, F_z], dtype=float)
    out = np.linalg.solve(A, b)
    return float(out[0]), float(out[1]), float(out[2])


def solve_model(p: ModelParams, root_selection: str = "stable_min_abs", verbose: bool = False) -> Solution:
    """
    Solve the MIU-NK fiscal sandbox.

    Parameters
    ----------
    p : ModelParams
        Model parameters.
    root_selection : str
        How to select among candidate stable roots.
        Current options:
        - 'stable_min_abs': stable root with the smallest |L_l|.
        - 'stable_positive_pi': stable root with Pi_l > 0 and smallest |L_l|.
    verbose : bool
        If True, print candidate liability roots.
    """
    p.validate()
    d = derived_objects(p)
    md = money_demand_coefficients(p)
    F = fiscal_coefficients(p)
    roots = solve_liability_block(p, verbose=verbose)

    if not roots:
        return Solution(p, False, "No real liability-block roots found.", False, None, [], None, {})

    stable_roots = [r for r in roots if r["stable"]]
    if root_selection == "stable_positive_pi":
        stable_roots = [r for r in stable_roots if r["Pi_l"] > 0]

    if not stable_roots:
        return Solution(
            p,
            False,
            "Roots found, but none satisfy the requested stability/sign criteria.",
            False,
            None,
            roots,
            None,
            {"n_roots": len(roots), "n_stable_roots": 0},
        )

    selected = min(stable_roots, key=lambda r: abs(r["L_l"]))
    X_l, Pi_l, L_l = selected["X_l"], selected["Pi_l"], selected["L_l"]

    # Shock coefficients: technology and uncompensated deficit.
    psi = d["psi"]
    r_a = p.sigma * psi * (p.rho_a - 1.0)
    p_a = 0.0
    X_a, Pi_a, L_a = _solve_shock_coefficients(p, X_l, Pi_l, p.rho_a, r_a, p_a, F["F_a"])

    r_xi = 0.0
    p_xi = -p.phi_xi
    X_xi, Pi_xi, L_xi = _solve_shock_coefficients(p, X_l, Pi_l, p.rho_xi, r_xi, p_xi, F["F_xi"])

    # Implied policy-rate coefficients.
    I_l = p.phi_x * X_l + p.phi_pi * Pi_l + p.phi_l
    I_a = p.phi_x * X_a + p.phi_pi * Pi_a
    I_xi = p.phi_x * X_xi + p.phi_pi * Pi_xi - p.phi_xi

    # Implied money coefficients.
    M_l = md["d_x"] * X_l + md["d_pi"] * Pi_l + md["d_l"]
    M_a = md["d_x"] * X_a + md["d_pi"] * Pi_a + md["d_a"]
    M_xi = md["d_x"] * X_xi + md["d_pi"] * Pi_xi + md["d_xi"]

    # Bond-debt coefficients: btilde = ell - mutilde.
    B_l = L_l - M_l
    B_a = L_a - M_a
    B_xi = L_xi - M_xi

    coeffs = pd.DataFrame(
        {
            "ell_lag": [X_l, Pi_l, L_l, I_l, M_l, B_l],
            "a": [X_a, Pi_a, L_a, I_a, M_a, B_a],
            "xi": [X_xi, Pi_xi, L_xi, I_xi, M_xi, B_xi],
        },
        index=["x", "pi", "ell", "i", "mu", "b"],
    )

    # Fiscal value of inflation: useful sign diagnostic.
    H_l = -(F["F_x"] * (1.0 - p.beta * L_l) / p.kappa + F["F_pi"])

    diagnostics = {
        **d,
        **md,
        **F,
        "n_roots": float(len(roots)),
        "n_stable_roots": float(len([r for r in roots if r["stable"]])),
        "fiscal_value_of_inflation_H_l": float(H_l),
        "liability_root_abs": float(abs(L_l)),
        "Pi_l_positive": float(Pi_l > 0),
    }

    status = "Solved."
    if Pi_l < 0:
        status += " Warning: Pi_l < 0; inherited liabilities lower inflation under this selected root."
    if H_l <= 0:
        status += " Warning: fiscal value of inflation is non-positive; monetary rule may be too active."

    return Solution(p, True, status, True, selected, roots, coeffs, diagnostics)


# -----------------------------------------------------------------------------
# Simulation / IRFs
# -----------------------------------------------------------------------------


def simulate_irf(
    sol: Solution,
    T: int = 40,
    shock: str = "xi",
    size: float = 1.0,
    initial_ell_lag: float = 0.0,
    initial_mu_lag: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate impulse responses from the solved linear model.

    Parameters
    ----------
    sol : Solution
        Solved model.
    T : int
        Number of periods.
    shock : {'xi', 'a'}
        Shock type at t=0.
    size : float
        Size of the innovation at t=0.
    initial_ell_lag : float
        Initial inherited total nominal liabilities in deviation form.
    initial_mu_lag : float
        Initial inherited real money balances in deviation form.

    Returns
    -------
    pd.DataFrame
        IRF dataframe with macro variables and fiscal-financing components.
    """
    sol.require_success()
    p = sol.params
    c = sol.coefficients
    assert c is not None

    if shock not in {"xi", "a"}:
        raise ValueError("shock must be either 'xi' or 'a'.")

    rows = []
    ell_lag = float(initial_ell_lag)
    mu_lag = float(initial_mu_lag)
    b_lag = ell_lag - mu_lag

    for t in range(T):
        a_t = size * (p.rho_a**t) if shock == "a" else 0.0
        xi_t = size * (p.rho_xi**t) if shock == "xi" else 0.0

        # Current outcomes.
        x_t = c.loc["x", "ell_lag"] * ell_lag + c.loc["x", "a"] * a_t + c.loc["x", "xi"] * xi_t
        pi_t = c.loc["pi", "ell_lag"] * ell_lag + c.loc["pi", "a"] * a_t + c.loc["pi", "xi"] * xi_t
        ell_t = c.loc["ell", "ell_lag"] * ell_lag + c.loc["ell", "a"] * a_t + c.loc["ell", "xi"] * xi_t
        i_t = c.loc["i", "ell_lag"] * ell_lag + c.loc["i", "a"] * a_t + c.loc["i", "xi"] * xi_t
        mu_t = c.loc["mu", "ell_lag"] * ell_lag + c.loc["mu", "a"] * a_t + c.loc["mu", "xi"] * xi_t
        b_t = ell_t - mu_t

        # Fiscal-financing decomposition, linearized around zero inflation.
        debt_dilution = p.bbar * pi_t
        money_inflation_tax = p.mubar * pi_t
        real_money_creation = mu_t - mu_lag
        seigniorage = money_inflation_tax + real_money_creation
        total_inflation_erosion = (p.bbar + p.mubar) * pi_t
        net_bond_financing = p.beta * b_t - b_lag - p.beta * p.bbar * i_t
        financing_total = net_bond_financing + debt_dilution + money_inflation_tax + real_money_creation
        accounting_error = financing_total - xi_t

        # Natural output and actual output.
        psi = derived_objects(p)["psi"]
        y_n_t = psi * a_t
        y_t = x_t + y_n_t

        rows.append(
            {
                "t": t,
                "a": a_t,
                "xi": xi_t,
                "ell_lag": ell_lag,
                "x": x_t,
                "y_n": y_n_t,
                "y": y_t,
                "pi": pi_t,
                "i": i_t,
                "mu": mu_t,
                "b": b_t,
                "ell": ell_t,
                "debt_dilution": debt_dilution,
                "money_inflation_tax": money_inflation_tax,
                "real_money_creation": real_money_creation,
                "seigniorage": seigniorage,
                "total_inflation_erosion": total_inflation_erosion,
                "net_bond_financing": net_bond_financing,
                "financing_total": financing_total,
                "accounting_error": accounting_error,
            }
        )

        ell_lag = ell_t
        mu_lag = mu_t
        b_lag = b_t

    return pd.DataFrame(rows)


def financing_summary(irf: pd.DataFrame, beta: float = 0.99) -> pd.Series:
    """Discounted and undiscounted sums of financing components over an IRF horizon."""
    cols = [
        "xi",
        "net_bond_financing",
        "debt_dilution",
        "money_inflation_tax",
        "real_money_creation",
        "seigniorage",
        "financing_total",
        "accounting_error",
    ]
    out = {}
    weights = beta ** irf["t"].to_numpy()
    for col in cols:
        out[f"sum_{col}"] = irf[col].sum()
        out[f"pv_{col}"] = float(np.sum(weights * irf[col].to_numpy()))
    if abs(out["pv_xi"]) > 1e-12:
        out["pv_share_debt_dilution"] = out["pv_debt_dilution"] / out["pv_xi"]
        out["pv_share_money_inflation_tax"] = out["pv_money_inflation_tax"] / out["pv_xi"]
        out["pv_share_real_money_creation"] = out["pv_real_money_creation"] / out["pv_xi"]
        out["pv_share_seigniorage"] = out["pv_seigniorage"] / out["pv_xi"]
        out["pv_share_net_bond_financing"] = out["pv_net_bond_financing"] / out["pv_xi"]
    return pd.Series(out)


# -----------------------------------------------------------------------------
# Laffer-like inflation-financing curves
# -----------------------------------------------------------------------------


def exact_money_demand_from_gross_rate(
    I: np.ndarray,
    p: ModelParams,
    C: float = 1.0,
    chi: Optional[float] = None,
) -> np.ndarray:
    """
    Exact MIU money demand as a function of the gross nominal interest rate I.

    The exact condition is:
        chi * mu^{-nu} * C^sigma = (I-1)/I.

    If chi is None, choose chi so that mu(Ibar)=mubar at C=1.
    """
    I = np.asarray(I, dtype=float)
    if np.any(I <= 1.0):
        raise ValueError("Gross nominal rates I must be greater than 1 for finite money demand.")

    Ibar = 1.0 / p.beta
    if chi is None:
        chi = (p.mubar ** p.nu) * ((Ibar - 1.0) / Ibar) / (C ** p.sigma)
        # This inversion comes from mubar = [chi*C^sigma*Ibar/(Ibar-1)]^{1/nu}.
        # Therefore chi = mubar^nu * (Ibar-1)/(Ibar*C^sigma).
    return (chi * (C ** p.sigma) * I / (I - 1.0)) ** (1.0 / p.nu)


def laffer_curve(
    p: ModelParams,
    I_grid: Optional[np.ndarray] = None,
    C: float = 1.0,
    real_gross_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build Laffer-like curves for inflation financing.

    We map gross nominal interest rates I into gross inflation using Fisher:
        Pi = I / R,
    where R is the gross real rate. By default R = Ibar = 1/beta, so the baseline
    I=Ibar corresponds to Pi=1.

    The function reports two kinds of objects:

    A. One-period transition decomposition from the baseline stock of liabilities:
       - debt_dilution_transition      = bbar  * (1 - 1/Pi)
       - money_tax_transition          = mubar * (1 - 1/Pi)
       - real_money_creation_transition= mu(I) - mubar
       - seigniorage_transition        = money_tax_transition + real_money_creation_transition
       - total_financing_transition    = debt_dilution_transition + seigniorage_transition

       This mirrors the local decomposition used in the linear IRFs.

    B. Steady-flow money revenue:
       - steady_seigniorage_flow       = mu(I) * (1 - 1/Pi)
       - opportunity_cost_revenue      = ((I - 1)/I) * mu(I)

       The second is the Werning/Sargent-Wallace object i/(1+i) L(i), with I=1+i.
    """
    d = derived_objects(p)
    Ibar = d["Ibar"]
    R = Ibar if real_gross_rate is None else real_gross_rate

    if I_grid is None:
        # Start at the zero-inflation baseline Ibar and move to higher rates.
        # This keeps the plot focused on the financing tradeoff from progressively
        # higher nominal rates/inflation, rather than on deflationary points near I=1.
        low = Ibar
        high = 1.3
        I_grid = np.linspace(low, high, 400)
    else:
        I_grid = np.asarray(I_grid, dtype=float)

    if np.any(I_grid <= 1.0):
        raise ValueError("I_grid must contain only gross nominal rates above 1.")

    Pi = I_grid / R
    mu = exact_money_demand_from_gross_rate(I_grid, p, C=C)

    debt_dilution_transition = p.bbar * (1.0 - 1.0 / Pi)
    money_tax_transition = p.mubar * (1.0 - 1.0 / Pi)
    real_money_creation_transition = mu - p.mubar
    seigniorage_transition = money_tax_transition + real_money_creation_transition
    total_financing_transition = debt_dilution_transition + seigniorage_transition

    steady_seigniorage_flow = mu * (1.0 - 1.0 / Pi)
    opportunity_cost_revenue = ((I_grid - 1.0) / I_grid) * mu

    return pd.DataFrame(
        {
            "gross_nominal_rate_I": I_grid,
            "net_nominal_rate": I_grid - 1.0,
            "gross_inflation_Pi": Pi,
            "inflation_rate": Pi - 1.0,
            "real_money_demand": mu,
            "debt_dilution_transition": debt_dilution_transition,
            "money_tax_transition": money_tax_transition,
            "real_money_creation_transition": real_money_creation_transition,
            "seigniorage_transition": seigniorage_transition,
            "total_financing_transition": total_financing_transition,
            "steady_seigniorage_flow": steady_seigniorage_flow,
            "opportunity_cost_revenue": opportunity_cost_revenue,
        }
    )


def laffer_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Return grid maxima for the main Laffer-like financing objects."""
    objects = [
        "debt_dilution_transition",
        "money_tax_transition",
        "real_money_creation_transition",
        "seigniorage_transition",
        "total_financing_transition",
        "steady_seigniorage_flow",
        "opportunity_cost_revenue",
    ]
    rows = []
    for obj in objects:
        idx = df[obj].idxmax()
        rows.append(
            {
                "object": obj,
                "max_value": df.loc[idx, obj],
                "gross_nominal_rate_I_at_max": df.loc[idx, "gross_nominal_rate_I"],
                "net_nominal_rate_at_max": df.loc[idx, "net_nominal_rate"],
                "gross_inflation_Pi_at_max": df.loc[idx, "gross_inflation_Pi"],
                "inflation_rate_at_max": df.loc[idx, "inflation_rate"],
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Lightweight plotting helpers
# -----------------------------------------------------------------------------


def plot_irf(irf: pd.DataFrame, variables: Iterable[str], title: Optional[str] = None):
    """Plot selected IRF variables. Returns the matplotlib axis."""

    ax = None
    for var in variables:
        ax = irf.plot(x="t", y=var, ax=ax, label=var)
    if title:
        ax.set_title(title)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("t")
    return ax


def plot_laffer(df: pd.DataFrame, variables: Iterable[str], x: str = "net_nominal_rate", title: Optional[str] = None):
    """Plot selected Laffer-curve variables. Returns the matplotlib axis."""

    ax = None
    for var in variables:
        ax = df.plot(x=x, y=var, ax=ax, label=var)
    if title:
        ax.set_title(title)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(x)
    return ax

# %% Example run

params = ModelParams(phi_pi=0.2, bbar=0.2, mubar=0.2)
sol = solve_model(params, root_selection="stable_positive_pi", verbose=False)
print(sol.status)

irf = simulate_irf(sol, T=40, shock="xi", size=0.1)
financing_plot = plot_irf(
    irf,
    ['b', 'mu']
    )
plt.show()
# The higher the ss debt, the higher the debt dilution effect for financing
# and thus, lower segniorage (both money tax and money creation) are lower
# Lower debt means lower debt dilution effect, which neccessitates higher emission
# and thus, higher segniorage (both money tax and money creation)
# higher ss money balances --> higher money inflation tax and money creation!
# money holdings and debt holdings are 1:1 substitutes here
# If SS holdings of bond and money increase in the same proportion the net 
# effect on the IRF is exactly zero!
# IRL we may suppose bbar to be high and mubar to be low
# this means segniorage is of necessity low. There isn't a lot of money balances
# in the economy!
# So the mechanism must be debt dilution
# which necessitates HIGHER inflation!
# %% running the big comparison across factors
bbar_grid = np.linspace(0, 2, 101)  # zero to 200% GDP debt
mubar_grid = np.linspace(0, 1, 101) # cashless to full M2
phi_pi_grid = np.linspace(0, 0.99, 101)
triple_params = list(product(bbar_grid , mubar_grid, phi_pi_grid))
financing_objs = [
    'debt_dilution', 'money_inflation_tax', 'real_money_creation',
    'net_bond_financing'
    ]
macro_objs = ['x', 'pi', 'i', 'mu', 'b']
df_results = {}
dict_results = {}
_id=0
for bbar, mubar, phi_pi in triple_params:
    params = ModelParams(phi_pi=phi_pi, bbar=bbar, mubar=mubar)
    sol = solve_model(params, root_selection="stable_positive_pi", verbose=False)
    if not sol.success:
        print(f'No results for ({bbar, mubar, phi_pi})')
        continue
    _id += 1
    irf = simulate_irf(sol, T=40, shock="xi", size=0.1)
    # get percentual of total financing done by each of debt dilution, money infl tax, real mon creation

    deficit_financing = irf[financing_objs].iloc[0]/irf['financing_total'][0]
    debt_financing = -irf[financing_objs].apply(lambda x: x/irf['net_bond_financing']).iloc[1:]
    mean_debt_df = debt_financing.mean()
    macro_evol = irf[macro_objs]
    mean_macro_evol = macro_evol.mean()

    dict_results[_id] = {
        'bbar': bbar,
        'mubar': mubar,
        'phi_pi': phi_pi,
        'deficit_fin_debt_dilution': deficit_financing['debt_dilution'],
        'deficit_fin_pi_tax': deficit_financing['money_inflation_tax'],
        'deficit_fin_money_creation': deficit_financing['real_money_creation'],
        'deficit_fin_bond': deficit_financing['net_bond_financing'],
        'debt_fin_debt_dilution': mean_debt_df['debt_dilution'],
        'debt_fin_pi_tax': mean_debt_df['money_inflation_tax'],
        'debt_fin_money_creation': mean_debt_df['real_money_creation'],
        'mean_y_gap': mean_macro_evol['x'],
        'mean_pi_gap': mean_macro_evol['pi'],
        'mean_i_gap': mean_macro_evol['i'],
        'mean_mu_gap': mean_macro_evol['mu'],
        'mean_b_gap': mean_macro_evol['b']
        }
    df_results[_id] = {
        'deficit_fin_df': deficit_financing,
        'debt_fin_df': debt_financing,
        'macro': macro_evol
        }


df_results = pd.DataFrame(dict_results).T

df_br = df_results.query("0.35 < mubar < 0.65 and 0.7 < bbar < 1.2")
df_br[['debt_fin_debt_dilution', 'debt_fin_pi_tax', 'debt_fin_money_creation']]
# For financing variables mubar/bbar is a sufficient statistic
# the financing mix changes as the ration of money to debt in the economy
# changes; it is constant if that is constant
# so, a heatmap with mubar/bbar in one ax and phi_pi on the other is enough
# to illustrate changing financing mixes

# for macro outcomes however, the absolute values of mubar and bbar
# are relevant. Lower mubar / bbar --> higher impacts
# the ratio, however, is irrelevant! All that matters is consolidate public
# liabilities. So for that we can plot this against changing phi_pi
# how does the financing mix change, and how do the macro variables react

df_results['ellbar'] = df_results['mubar'] + df_results['bbar']
df_results['debt_money_ratio'] = df_results['bbar']/df_results['mubar']

df_results.to_parquet(r'D:\Users\c337191\Documents\unpleasant_nk_arithmetic\Data\sims_FTPL.parquet')
