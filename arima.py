import sys
from statsmodels.tsa.arima_model import ARIMA
from scipy.optimize import brute

def akaike_inf(order, endog):
    try:
        fit = ARIMA(endog, order).fit(full_outpu=False, disp=False)
        return fit.aic
    except Exception as e:
        return sys.maxsize

def auto(ts, maxorder):
    grid  = (slice(0, maxorder + 1), slice(0, 3), slice(0, maxorder + 1))
    order = brute(akaike_inf, grid, args=(ts,), finish=None)

    return ARIMA(ts, [int(i) for i in order]).fit()

def r_auto(ts):
    import readline
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    forecast = importr('forecast')
    return forecast.auto_arima(ts)
