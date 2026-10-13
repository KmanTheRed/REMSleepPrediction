"""
Continuous 2-segment ("hockey-stick") change-point model for one REM boundary,
plus a moving-block bootstrap confidence interval on the change-point time.

Model (see protocol.md Section 5, Group A):
  onset  ("rising_after"): y(t) = c0                  for t <  tau
                            y(t) = c0 + s*(t - tau)    for t >= tau
  offset ("flat_after"):   y(t) = c0 + s*(tau - t)     for t <  tau
                            y(t) = c0                  for t >= tau

Both are linear in (c0, s) for a fixed tau (a single hinge basis column), so the
grid search over tau is done by closed-form OLS on 2 parameters, vectorized
across the whole tau grid at once -- this is what makes a 200-resample
bootstrap per episode/boundary affordable.
"""
import numpy as np


def _hinge_matrix(t, tau_grid, kind):
    """h[i, j] = hinge basis at time t[j] for change-point tau_grid[i]."""
    t = np.asarray(t, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if kind == "rising_after":
        h = np.maximum(0.0, t[None, :] - tau_grid[:, None])
    elif kind == "flat_after":
        h = np.maximum(0.0, tau_grid[:, None] - t[None, :])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return h


def grid_search_tau(t, y, kind, tau_grid):
    """Closed-form OLS of y ~ 1 + hinge(t, tau) for every tau in tau_grid at
    once; returns the SSE-minimizing tau plus its fit.

    Returns: tau_hat, (c0, s), fitted_values, residuals, sse
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    h = _hinge_matrix(t, tau_grid, kind)  # (n_tau, n)

    sum_h = h.sum(axis=1)
    sum_h2 = (h * h).sum(axis=1)
    sum_y = y.sum()
    sum_hy = h @ y

    denom = n * sum_h2 - sum_h ** 2
    # Guard degenerate taus (all-zero hinge column, e.g. tau at/after the
    # window edge) -- these can never be optimal and are just masked out.
    valid = denom > 1e-9
    s = np.zeros_like(denom)
    c0 = np.full_like(denom, np.nan)
    s[valid] = (n * sum_hy[valid] - sum_h[valid] * sum_y) / denom[valid]
    c0[valid] = (sum_y - s[valid] * sum_h[valid]) / n

    fitted_all = c0[:, None] * np.ones((1, n)) + s[:, None] * h  # (n_tau, n)
    resid_all = y[None, :] - fitted_all
    sse = (resid_all ** 2).sum(axis=1)
    sse[~valid] = np.inf

    best = int(np.argmin(sse))
    tau_hat = float(np.asarray(tau_grid)[best])
    beta = (float(c0[best]), float(s[best]))
    fitted = fitted_all[best]
    resid = resid_all[best]
    return tau_hat, beta, fitted, resid, float(sse[best])


def _sloped_floor_columns(t, tau_grid, kind, w0):
    """Two hinge basis columns (n_tau, n) for the 'sloped floor' model:
    ramp at slope s1 for w0 seconds after departure, then continue at a
    (possibly different, non-zero) slope s2 -- i.e. genuine continued
    decline rather than a hard flat plateau or an unboundedly steepening
    ramp. kind='rising_after' -> onset (departure moves forward in time
    from tau); kind='flat_after' -> offset (departure approaches tau from
    before, mirrored via u = tau - t)."""
    t = np.asarray(t, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if kind == "rising_after":
        u = t[None, :] - tau_grid[:, None]
    elif kind == "flat_after":
        u = tau_grid[:, None] - t[None, :]
    else:
        raise ValueError(f"unknown kind {kind!r}")
    u = np.maximum(0.0, u)
    b1 = np.minimum(u, w0)
    b2 = np.maximum(u - w0, 0.0)
    return b1, b2


def grid_search_tau_slopedfloor(t, y, kind, tau_grid, w0=15.0, motion=None):
    """3-parameter (c0, s1, s2) sloped-floor model, optionally with a 4th
    fixed (tau-independent) covariate column `motion`. Fit via batched
    normal-equation solves across the whole tau grid (np.linalg.solve on
    stacked matrices) -- no Python loop over tau, so this stays fast enough
    for a full bootstrap.

    Returns: tau_hat, beta, fitted, resid, sse (same shape as grid_search_tau).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    b1, b2 = _sloped_floor_columns(t, tau_grid, kind, w0)
    n_tau = len(tau_grid)
    ones = np.ones((n_tau, n))
    cols = [ones, b1, b2]
    if motion is not None:
        cols.append(np.broadcast_to(np.asarray(motion, dtype=float), (n_tau, n)))
    X = np.stack(cols, axis=-1)  # (n_tau, n, k)

    XtX = np.einsum("tnk,tnj->tkj", X, X)
    Xty = np.einsum("tnk,n->tk", X, y)

    # Ridge-stabilize in case a tau makes b1/b2 degenerate (e.g. w0 window
    # too small for the data available at the grid edge).
    k = X.shape[-1]
    XtX = XtX + 1e-10 * np.eye(k)[None, :, :]
    beta_all = np.linalg.solve(XtX, Xty[..., None])[..., 0]  # (n_tau, k)

    fitted_all = np.einsum("tnk,tk->tn", X, beta_all)
    resid_all = y[None, :] - fitted_all
    sse = (resid_all ** 2).sum(axis=1)

    best = int(np.argmin(sse))
    tau_hat = float(tau_grid[best])
    beta = tuple(beta_all[best].tolist())
    fitted = fitted_all[best]
    resid = resid_all[best]
    return tau_hat, beta, fitted, resid, float(sse[best])


def changepoint_ci_slopedfloor(t, y, kind, tau_grid, w0=15.0, motion=None,
                                block_len=30, n_boot=200, seed=0, ci=0.95):
    """Bootstrap CI for the sloped-floor model, mirroring changepoint_ci."""
    tau_hat, beta, fitted, resid, sse = grid_search_tau_slopedfloor(
        t, y, kind, tau_grid, w0=w0, motion=motion)
    resid_c = resid - resid.mean()

    rng = np.random.default_rng(seed)
    n = len(t)
    taus_boot = np.empty(n_boot)
    for b in range(n_boot):
        boot_resid = moving_block_bootstrap_resid(resid_c, block_len, n, rng)
        y_boot = fitted + boot_resid
        tau_b, *_ = grid_search_tau_slopedfloor(t, y_boot, kind, tau_grid, w0=w0, motion=motion)
        taus_boot[b] = tau_b

    lo_p = (1 - ci) / 2 * 100
    hi_p = 100 - lo_p
    lo, hi = np.percentile(taus_boot, [lo_p, hi_p])
    return dict(
        tau=tau_hat, ci_lo=float(lo), ci_hi=float(hi), width=float(hi - lo),
        beta=beta, sse=sse, taus_boot=taus_boot,
    )


def moving_block_bootstrap_resid(resid, block_len, n, rng):
    """Resample residuals in contiguous blocks (respects short-range
    autocorrelation left by the moving-median smoother) to build a synthetic
    residual series of length n."""
    n_resid = len(resid)
    block_len = max(1, min(block_len, n_resid))
    n_blocks = int(np.ceil(n / block_len))
    max_start = max(1, n_resid - block_len + 1)
    starts = rng.integers(0, max_start, size=n_blocks)
    pieces = [resid[s:s + block_len] for s in starts]
    return np.concatenate(pieces)[:n]


def changepoint_ci(t, y, kind, tau_grid, block_len=15, n_boot=200, seed=0, ci=0.95):
    """Fit the change-point and return its bootstrap confidence interval.

    Returns a dict: tau, ci_lo, ci_hi, width, beta, sse, taus_boot.
    """
    tau_hat, beta, fitted, resid, sse = grid_search_tau(t, y, kind, tau_grid)
    resid_c = resid - resid.mean()

    rng = np.random.default_rng(seed)
    n = len(t)
    taus_boot = np.empty(n_boot)
    for b in range(n_boot):
        boot_resid = moving_block_bootstrap_resid(resid_c, block_len, n, rng)
        y_boot = fitted + boot_resid
        tau_b, _, _, _, _ = grid_search_tau(t, y_boot, kind, tau_grid)
        taus_boot[b] = tau_b

    lo_p = (1 - ci) / 2 * 100
    hi_p = 100 - lo_p
    lo, hi = np.percentile(taus_boot, [lo_p, hi_p])
    return dict(
        tau=tau_hat, ci_lo=float(lo), ci_hi=float(hi), width=float(hi - lo),
        beta=beta, sse=sse, taus_boot=taus_boot,
    )
