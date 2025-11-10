from scipy.interpolate import CubicSpline, interp1d
import numpy as np

def cubic_hermite_interp(x, y):
    """
    Creates a cubic Hermite interpolator for yield curves using finite differences.

    Parameters
    ----------
    x : array-like
        Time grid (year fractions, sorted ascending)
    y : array-like
        Corresponding rate values

    Returns
    -------
    callable
        Interpolation function that takes a time value and returns interpolated rate
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    def interpolate(s):
        """
        Interpolates the rate at target time s using cubic Hermite method.

        Parameters
        ----------
        s : float or array-like
            Target time(s) for interpolation

        Returns
        -------
        float or np.ndarray
            Interpolated rate(s) at s
        """
        # Handle array input
        if isinstance(s, (list, np.ndarray)):
            return np.array([interpolate(float(si)) for si in s])

        s = float(s)
        n = len(x)

        # Clamp if outside bounds (flat extrapolation)
        if s <= x[0]:
            return float(y[0])
        if s >= x[-1]:
            return float(y[-1])

        # Find surrounding interval
        i = np.searchsorted(x, s) - 1
        i = np.clip(i, 0, n - 2)

        # Get four neighboring points if possible
        x0 = x[max(i - 1, 0)]
        x1 = x[i]
        x2 = x[i + 1]
        x3 = x[min(i + 2, n - 1)]

        y0 = y[max(i - 1, 0)]
        y1 = y[i]
        y2 = y[i + 1]
        y3 = y[min(i + 2, n - 1)]

        # Calculate finite-difference slopes
        # Handle edge cases where denominators might be zero
        dx01 = x1 - x0
        dx12 = x2 - x1
        dx23 = x3 - x2

        if dx01 == 0:
            d0 = 0.0
        else:
            d0 = (y1 - y0) / dx01

        if dx12 == 0:
            d1 = 0.0
        else:
            dy12 = (y2 - y1) / dx12
            dx02 = x2 - x0
            if dx02 == 0:
                d1 = 0.0
            else:
                d1 = (dy12 - d0) / dx02

        if dx23 == 0 or dx12 == 0:
            d2 = 0.0
        else:
            dy23 = (y3 - y2) / dx23
            dy12 = (y2 - y1) / dx12
            dx13 = x3 - x1
            dx03 = x3 - x0
            if dx03 == 0:
                d2 = 0.0
            else:
                d2 = (dy23 - dy12 - d1 * dx13) / dx03

        # Piecewise cubic polynomial
        if x0 <= s <= x2:
            v = (
                y0
                + d0 * (s - x0)
                + d1 * (s - x0) * (s - x1)
                + d2 * (s - x0) * (s - x1) * (s - x2)
            )
        else:
            v = (
                y1
                + d0 * (s - x1)
                + d1 * (s - x1) * (s - x2)
                + d2 * (s - x1) * (s - x2) * (s - x3)
            )
        return float(v)

    return interpolate

def linear_interpolator(
    x: float,
    x_values: list[float],
    y_values: list[float],
    val_date: float
) -> float:
    """
    Performs linear interpolation on the cumulative (time-weighted)
    hazard rates, matching the VBA logic.

    Parameters:
        x (float): Target time (in years from valuation date)
        x_values (list[float]): Time points (in years from valuation date)
        y_values (list[float]): Hazard rates corresponding to x_values
        val_date (float): Valuation time (usually 0.0)

    Returns:
        float: Interpolated hazard rate at x
    """

    # Handle out-of-bound cases (flat extrapolation)
    if x <= x_values[0]:
        return y_values[0]
    if x >= x_values[-1]:
        return y_values[-1]

    # Loop through intervals to find where x lies
    for i in range(1, len(x_values)):
        if x <= x_values[i]:
            t0, t1 = x_values[i - 1], x_values[i]
            r0, r1 = y_values[i - 1], y_values[i]

            # Convert to cumulative hazard (λ * t)
            H0 = r0 * (t0 - val_date)
            H1 = r1 * (t1 - val_date)

            # Linear interpolation on cumulative hazard
            Hx = H0 + (H1 - H0) * (x - t0) / (t1 - t0)

            # Convert back to spot hazard rate
            interpolated_rate = Hx / (x - val_date)

            return interpolated_rate

    # Safety fallback (should not reach here)
    return y_values[-1]

def hermite_rt_interp(x, y, val_date=0.0):
    """
    Hermite RT interpolation — interpolates in (r * t) space as described in
    the provided formulas (Hermite RT method used in yield curve construction).

    Parameters
    ----------
    x : array-like
        Time grid (year fractions from valuation date, sorted ascending)
    y : array-like
        Spot rates corresponding to x
    val_date : float, optional
        Valuation time (default 0.0)

    Returns
    -------
    callable
        Interpolator function f(t) returning interpolated rates at target t
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Convert spot rates to r*t (Hermite RT works on this)
    rt = y * (x - val_date)
    n = len(x)

    # Compute slopes r'_i (derivatives of r*t w.r.t. t)
    r_prime = np.zeros_like(rt)

    # Interior slopes
    for i in range(1, n - 1):
        dx1 = x[i] - x[i - 1]
        dx2 = x[i + 1] - x[i]
        if dx1 == 0 or dx2 == 0:
            r_prime[i] = 0.0
        else:
            r_prime[i] = (
                (rt[i + 1] - rt[i]) * (x[i] - x[i - 1]) / dx2
                + (rt[i] - rt[i - 1]) * (x[i + 1] - x[i]) / dx1
            ) / (x[i + 1] - x[i - 1])

    # Boundary conditions
    if n >= 3:
        # Left boundary (r'_1)
        t1, t2, t3 = x[0], x[1], x[2]
        r1, r2, r3 = rt[0], rt[1], rt[2]
        r_prime[0] = (
            (r2 - r1) * (t3 + t2 - 2 * t1) / (t2 - t1)
            - (r3 - r2) * (t2 - t1) / (t3 - t2)
        ) / (t3 - t1)

        # Right boundary (r'_n)
        tn, tn1, tn2 = x[-1], x[-2], x[-3]
        rn, rn1, rn2 = rt[-1], rt[-2], rt[-3]
        r_prime[-1] = -(
            (rn1 - rn2) * (tn - tn1) / (tn1 - tn2)
            - (rn - rn1) * (2 * tn - tn1 - tn2) / (tn - tn1)
        ) / (tn - tn2)
    else:
        r_prime[0] = r_prime[-1] = 0.0

    def interpolate(t):
        """
        Interpolates the rate at time t using Hermite RT method.
        """
        if isinstance(t, (list, np.ndarray)):
            return np.array([interpolate(float(tt)) for tt in t])

        # Flat extrapolation
        if t <= x[0]:
            return y[0]
        if t >= x[-1]:
            return y[-1]

        # Find interval
        i = np.searchsorted(x, t) - 1
        i = np.clip(i, 0, n - 2)

        t1, t2 = x[i], x[i + 1]
        r1, r2 = rt[i], rt[i + 1]
        r1p, r2p = r_prime[i], r_prime[i + 1]
        h = t2 - t1
        m = (t - t1) / h  # m_t

        # Hermite polynomial on r*t
        rt_interp = (
            (1 - m) ** 2 * (1 + 2 * m) * r1
            + m**2 * (3 - 2 * m) * r2
            + (1 - m) ** 2 * m * h * r1p
            + m**2 * (m - 1) * h * r2p
        )

        # Convert back to rate
        if t - val_date == 0:
            return y[0]
        return rt_interp / (t - val_date)

    return interpolate

def cubic_spline(x, y):
    """Cubic spline interpolator with natural boundary conditions."""
    return CubicSpline(x, y, bc_type="natural")