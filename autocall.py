import numpy as np
import pandas as pd
from typing import Callable, List
from datetime import date


def price_autocall(
    paths_df: pd.DataFrame,
    initial_spot: float,
    observation_dates: List[date],
    barrier: float,  # e.g. 0.6 for 60%
    discount_factor: Callable[[date], float],  # DF to each obs/maturity
    notional: float = 1.0,
) -> float:
    """
    Price an equity autocallable with:
      - annual observations
      - autocall if S_t > initial_spot
      - coupon = number_of_years until call
      - maturity payoff:
          * if S_T > S0:   notional*(1+years)
          * if barrier < S_T < S0: notional*(S_T/S0)
          * if S_T < barrier*S0:   0

    Parameters
    ----------
    paths_df : pd.DataFrame
        DataFrame: index = dates, columns = Sim_1 .. Sim_n, values = S_t.
    initial_spot : float
        Initial index level S0.
    observation_dates : list of date
        Same order as DataFrame index.
    barrier : float
        Barrier as FRACTION of S0 (e.g. 0.6 = 60%).
    discount_factor : Callable[[date], float]
        Function returning DF(today, date).
    notional : float
        Investment amount (default 1.0).

    Returns
    -------
    float
        Fair price (discounted expected payoff).
    """

    S0 = float(initial_spot)
    paths = paths_df.values       # shape (num_dates, num_sims)
    dates = list(paths_df.index)
    n_dates, n_sims = paths.shape

    if n_dates != len(observation_dates):
        raise ValueError("observation_dates must align with DataFrame index")

    barrier_level = barrier * S0

    payoffs = np.zeros(n_sims)

    # Loop over each simulation path
    for sim in range(n_sims):

        autocalled = False

        for j in range(n_dates - 1):
            S_t = paths[j, sim]
            if S_t > S0:
                # autocall triggered at observation j
                years = j + 1  # zero-based index: j=0 => year 1
                payoff = notional * (1 + years)
                df = discount_factor(observation_dates[j])
                payoffs[sim] = payoff * df
                autocalled = True
                break

        if autocalled:
            continue

        # --- maturity payoff ---
        S_T = paths[-1, sim]

        if S_T > S0:
            years = n_dates
            payoff = notional * (1 + years)

        elif S_T > barrier_level:
            payoff = notional * (S_T / S0)

        else:
            payoff = 0.0

        df_T = discount_factor(observation_dates[-1])
        payoffs[sim] = payoff * df_T

    # Monte Carlo price = expectation
    return float(np.mean(payoffs))
