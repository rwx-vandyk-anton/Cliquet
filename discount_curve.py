import pandas as pd
import numpy as np
from datetime import date
from math import log,exp


def act365(start_date: date, end_date: date) -> float:
    """ACT/365 day count convention."""
    return (end_date - start_date).days / 365.0


class DiscountYieldCurveHandler:
    def __init__(self, valuation_date: date, df: pd.DataFrame, interpolator):
        """
        Initializes the DiscountYieldCurveHandler.

        Parameters:
            valuation_date (date): The base valuation date.
            df (pd.DataFrame): DataFrame containing:
                - 'Date' (datetime/date): pillar dates of the yield curve.
                - 'Rate' (float): NACA Act/365 annualized rates.
            interpolator (callable): A function that takes (x, y) and returns
                a callable f(t) that interpolates y-values for given t-values.
        """
        self.valuation_date = valuation_date
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        # Convert dates to year fractions relative to valuation date
        self.year_frac = np.array([act365(valuation_date, d) for d in df["Date"]])
        self.rates = np.array(df["Rate"])

        # Build interpolation function using the provided interpolator
        self.interpolator = interpolator(self.year_frac, self.rates)

    def get_rate(self, t: float) -> float:
        """Returns the interpolated NACA Act/365 rate for a given year fraction t."""
        return float(self.interpolator(t))
    
    def get_rate_for_date(self, target_date: date) -> float:
        """Returns the interpolated NACA Act/365 rate for a given year fraction t."""
        t = act365(self.valuation_date, target_date)
        return float(self.interpolator(t))

    def discount_factor(self, t: float) -> float:
        """Returns the discount factor for a given year fraction t using NACA/365."""
        r_t = self.get_rate(t)
        return np.exp(-r_t * t)

    def get_discount_factor_for_date(self, target_date: date) -> float:
        """Returns the discount factor for a given target date."""
        t = act365(self.valuation_date, target_date)
        return self.discount_factor(t)
    
    def flat_forward_cont_rate(self, t1: float, t2: float) -> float:
        """
        Computes the flat forward continuous rate between two year fractions.

        Parameters:
            t1 (float): Start time (in years from valuation date).
            t2 (float): End time (in years from valuation date).

        Returns:
            float: Continuous-compounded forward rate between t1 and t2.
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")

        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return (log(df1) - log(df2)) / (t2 - t1)

    def flat_forward_cont_rate_between_dates(self, start_date: date, end_date: date) -> float:
        """
        Computes the flat forward continuous rate between two calendar dates.

        Parameters:
            start_date (date): Start date.
            end_date (date): End date.

        Returns:
            float: Continuous-compounded forward rate between the two dates.
        """
        t1 = act365(self.valuation_date, start_date)
        t2 = act365(self.valuation_date, end_date)
        return self.flat_forward_cont_rate(t1, t2)
