import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def load_datasets(path_pred = './data/waterlevel_timeseries/prediction', path_meas = './data/waterlevel_timeseries/measured'):

    years = np.arange(2014,2024,1)
    for i in range(years.size):

        df_pred = pd.read_parquet(f'{path_pred}/Vlissingen_prediction_{years[i]}.parquet')
        w_pred = np.array(df_pred['Waterhoogte berekend Oppervlaktewater t.o.v. NAP']) / 100
        timestamp_pred = np.array(df_pred['Datum'] + ' ' + df_pred['Tijd (NL tijd)'] + ':00')
        t_pred = pd.to_datetime(timestamp_pred, dayfirst='True')
        t_pred = t_pred.tz_localize('Europe/Amsterdam', ambiguous='NaT', nonexistent='shift_forward')
        t_pred = t_pred.astype('int64') // 10**9

        df_meas = pd.read_parquet(f'{path_meas}/Vlissingen_measured_{years[i]}.parquet')
        w_meas  = np.array(df_meas['waterlevel']) / 100
        t_meas = df_meas['timestamp'].astype('int64') // 10**9

        w_pred = np.interp(x = t_meas,  xp = t_pred, fp = w_pred)
        t_pred = np.copy(t_meas)

        if i == 0:
            time  = np.copy(t_meas)
            w_measured = np.copy(w_meas)
            w_predicted = np.copy(w_pred)
        else:
            time = np.append(time, t_meas)
            w_measured = np.append(w_measured, w_meas)
            w_predicted = np.append(w_predicted, w_pred)

        w_stochastic = w_measured - w_predicted

    return time, w_measured, w_predicted, w_stochastic

def estimate_stochastic_parameters(x, dt, xmean):
    '''dx = sigma * np.sqrt(dt) * N(0,1) + rho * (xmean - x) * dt'''

    def linear_fit(x, a, b):
        return a + x*b

    dx = np.diff(x)
    residuals = dx - xmean

    # Predict rho
    par, _ = curve_fit(linear_fit, xdata = (xmean - x[:-1]), ydata = dx, p0=(0, 1))
    rho_est = par[1] / dt  # (hr^-1)

    # Predict sigma
    sigma_est = np.std(residuals) / np.sqrt(dt) # (m hr^-0.5)

    return rho_est, sigma_est

def generate_oun(nt, xmean = 0, sigma = 1, rho = 0.1, dt = 1):

    r = np.random.normal(size = nt)
    x = np.array([np.nan]*nt)
    x[0] = xmean
    for t in range(1, nt):
        x[t] = x[t - 1] + sigma*np.sqrt(dt)*r[t] + rho*(xmean - x[t - 1])*dt

    return x

t, w_measured, w_predicted, w_stochastic = load_datasets()

def find_tidal_amplitude(w_predicted, dt = 10/60):

    # Interpolate to 5 min measuring interval so 745 min tidal interval can be divided evenly
    t = np.arange(0, w_predicted.size, 1) * dt    
    tp = np.arange(0, t.max(), 5/60)
    w_predicted_interp = np.interp(x = tp, xp = t, fp = w_predicted)

    # Reshape water level time series into stacked array of n tidal cycles (each 149 measurements long at dt = 5/60)
    n_measurements_per_tide = int((745/60) / (5/60))
    n_tides = int(np.floor(w_predicted_interp.size/n_measurements_per_tide))
    w_predicted_interp = w_predicted_interp[:int(n_tides*n_measurements_per_tide)]
    w_predicted_interp = w_predicted_interp.reshape(n_tides, n_measurements_per_tide)

    # Calcualte high & low water level of each tidal cycle, and from taht calculate the avg. tidal amplitude
    high_water_level = w_predicted_interp.max(axis = 1)
    low_water_level = w_predicted_interp.min(axis = 1)
    tidal_amplitude = (high_water_level - low_water_level)/2

    return tidal_amplitude.mean()

dt = 10/60 # hours
rho_est, sigma_est = estimate_stochastic_parameters(x = w_stochastic, dt = dt, xmean = 0)

avg_tidal_amplitude = find_tidal_amplitude(w_predicted, dt = 10/60)

# Rescale sigma by tidal amplitude
sigma_est_rescaled = sigma_est / avg_tidal_amplitude
print(f'Rho = {rho_est}, Sigma = {sigma_est_rescaled}')

w_stochastic_est = generate_oun(nt = t.size, xmean = 0, sigma = sigma_est, rho = rho_est, dt = dt)
t = (t - t.min()) / (60*60*24*365.25) # years
plt.plot(t, w_stochastic,     c = 'black')
plt.plot(t, w_stochastic_est, c = 'red', alpha = 0.5)
plt.show()

plt.plot(t, w_measured)
plt.plot(t, w_predicted, c = 'red', alpha = 0.5)
plt.show()