def price_autocall(
    paths_df: pd.DataFrame,
    initial_spot: float,
    barrier: float,  # expressed as fraction of S0 (e.g. 0.6)
    discount_factor: Callable[[date], float],
    notional: float = 1.0,
) -> float:
    """
    Price an equity autocallable using GBM paths.

    Autocall rule:
        If S(t) > S0 at an observation date:
            payoff = notional * (1 + elapsed_years)
            discounted back to today
            terminate

    Maturity payoff (if no autocall):
        If S_T > S0:
            payoff = notional * (1 + elapsed_years_total)
        elif S_T > barrier*S0:
            payoff = notional * (S_T / S0)
        else:
            payoff = 0

    Parameters
    ----------
    paths_df : pd.DataFrame
        DataFrame with index = dates, columns = Sim_i.
    initial_spot : float
        Initial index level S0.
    barrier : float
        Barrier fraction (e.g. 0.6 = 60%).
    discount_factor : Callable[[date], float]
        DF(today, date).
    notional : float
        Notional invested.

    Returns
    -------
    float
        Fair price (discounted expected payoff).
    """

    S0 = float(initial_spot)
    dates = list(paths_df.index)
    dates_arr = np.array(dates)
    n_dates = len(dates)
    paths = paths_df.values  # shape (dates, sims)
    n_sims = paths.shape[1]

    barrier_level = barrier * S0

    payoffs = np.zeros(n_sims)

    # Helper to get ACT/365 year-fraction
    def yf(d0, d1):
        return (d1 - d0).days / 365.0

    for sim in range(n_sims):

        autocalled = False

        # Loop through observation dates EXCLUDING maturity
        for j in range(n_dates - 1):

            S_t = paths[j, sim]
            obs_date = dates[j]

            if S_t > S0:  # autocall trigger
                elapsed_years = yf(dates[0], obs_date)
                payoff = notional * (1 + elapsed_years)
                payoffs[sim] = payoff * discount_factor(obs_date)
                autocalled = True
                break

        if autocalled:
            continue

        # ---------- maturity payoff ----------
        S_T = paths[-1, sim]
        maturity_date = dates[-1]
        total_years = yf(dates[0], maturity_date)

        if S_T > S0:
            payoff = notional * (1 + total_years)

        elif S_T > barrier_level:
            payoff = notional * (S_T / S0)

        else:
            payoff = 0.0

        payoffs[sim] = payoff * discount_factor(maturity_date)

    return float(np.mean(payoffs))
