def simulate_path_matrix_df(
    self,
    reset_dates: List[date],
    n: int,
    seed: Optional[int] = None,
    antithetic: bool = False,
    spot0: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulate continuous GBM paths across the provided reset_dates and 
    return a DataFrame with:
        - index   = reset_dates
        - columns = Sim_1, Sim_2, ..., Sim_n
        - values  = simulated paths
    """
    # --- run your existing matrix simulation ---
    paths = self.simulate_path_matrix(
        reset_dates=reset_dates,
        n=n,
        seed=seed,
        antithetic=antithetic,
        spot0=spot0,
    )

    # --- build DataFrame ---
    col_names = [f"Sim_{i+1}" for i in range(n)]
    df = pd.DataFrame(paths.T, index=reset_dates, columns=col_names)

    return df
