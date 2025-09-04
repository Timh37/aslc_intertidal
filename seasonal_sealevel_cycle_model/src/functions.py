import numpy as np
import matplotlib.pyplot as plt

def create_waterlevel_timeseries(tidal_amplitude = 1, amplitude_annual_cycle = 0, amplitude_M2 = 1, amplitude_S2 = 0.4, sigma = 0.025, rho = 0.021, mean_waterlevel = 0, n_years = 1, dt = 5/60, plot = False):

    def generate_oun(nt, sigma, rho, dt, xmean = 0):

        r = np.random.normal(size = nt)
        x = np.array([np.nan]*nt)
        x[0] = xmean
        for t in range(1, nt):
            x[t] = x[t - 1] + sigma*np.sqrt(dt)*r[t] + rho*(xmean - x[t - 1])*dt

        return x

    tf    = 365.25 * 24 * n_years # 1 year (in hours)
    time = np.arange(0, tf + dt, dt)
    nt = time.size

    # Periodic parameters
    period_M2       = 12.42 # M2 period (h)
    period_S2       = 12    # S2 period (h)
    period_annual   = 365.25 * 24 # (h)

    # # Stochastic parameters
    # sigma           = 0.025 * tidal_amplitude # (m hr^-0.5)
    # rho             = 0.021 # (  hr^-1  )

    # Re-scale sigma when rho is manipulated to maintain constant std(w)
    rho_default = 0.021
    if rho != rho_default:
        std_w = sigma * tidal_amplitude / np.sqrt(2 * rho_default)
        sigma = std_w * np.sqrt(2 * rho) / tidal_amplitude

    waterlevel_periodic = mean_waterlevel + tidal_amplitude/np.sqrt(amplitude_M2**2 + amplitude_S2**2)*(amplitude_M2*np.cos(time*2*np.pi/period_M2) + amplitude_S2*np.cos(time*2*np.pi/period_S2)) + (amplitude_annual_cycle*np.cos(time*2*np.pi/period_annual))    
    waterlevel_stochastic = generate_oun(nt, sigma * tidal_amplitude, rho, dt, xmean = 0)
    waterlevel = waterlevel_periodic + waterlevel_stochastic
    mean_waterlevel = mean_waterlevel + (amplitude_annual_cycle*np.cos(time*2*np.pi/period_annual))   

    if plot:
        plt.plot(time/24, waterlevel, c = 'blue', alpha = 0.3)
        plt.plot(time/24, waterlevel_periodic, alpha = 0.3, c = 'blue')
        plt.plot(time/24, waterlevel_stochastic, alpha = 0.5, c = 'red')
        plt.ylabel('Water level (m MSL)', fontsize = 12)
        plt.xlabel('Time (days)', fontsize = 12)
        plt.show()

    return time, waterlevel, mean_waterlevel

def calculate_highlow_water(waterlevel, dt = 5/60, plot = False):
    '''
    Calculates the maximum and minimum waterlevel of each tidal cycle from an input waterlevel timeseries
    
    Input
    (1) waterlevel: Water level time series data                            (1D float array)
    (2) dt: Interval between timeseries measurements (in minutes!!)         (integer value)

    Output
    (1) low water level:  Series of low water levels (one per tidal cycle)  (1D float array)
    (2) high water level: Series of high water levels (one per tidal cycle) (1D float array)
    '''

    def reshape_timeseries_into_cycles(waterlevel, dt):

        # Define mesaurement frequency parameters
        tidal_interval_hr = 745/60 # (minutes)
        measurements_per_cycle = np.round(tidal_interval_hr/dt).astype(int)

        # Reshape waterlevel timeseries into array of tidal cycles
        n_measurements = np.floor(waterlevel.size/measurements_per_cycle).astype(int) * measurements_per_cycle

        waterlevel = waterlevel[:n_measurements]
        n_cycles = int(n_measurements/measurements_per_cycle)
        waterlevel_cycles = waterlevel.reshape(n_cycles, measurements_per_cycle).T

        return waterlevel_cycles

    waterlevel_cycles = reshape_timeseries_into_cycles(waterlevel, dt)
    high_waterlevel = waterlevel_cycles.max(axis = 0)
    low_waterlevel = waterlevel_cycles.min(axis = 0)
    tidal_cycle = np.arange(0, waterlevel_cycles.shape[1], 1)

    if plot:
        # print(f'Mean high water level: {np.round(np.nanmean(high_waterlevel), 2)}')
        # print(f'Mean low water level: {np.round(np.nanmean(low_waterlevel), 2)}')

        plt.plot(low_waterlevel, label = 'Low water')
        plt.plot(high_waterlevel, label = 'High water')
        plt.ylabel('High/low water level (m MSL)', fontsize = 12)
        plt.xlabel('Tidal cycles', fontsize = 12)
        plt.legend(frameon = False, fontsize = 11)
        plt.show()

    return tidal_cycle, low_waterlevel, high_waterlevel

def calculate_inundation_metrics(low_waterlevel, high_waterlevel, dz = 0.001, plot = False):

    tidal_amplitude = high_waterlevel.mean() - low_waterlevel.mean()
    elevation_bins = np.arange((-2.8 * tidal_amplitude) - dz/2, (2.8 * tidal_amplitude) + dz/2 + dz, dz)
    elevation = (elevation_bins[1:] + elevation_bins[:-1])/2

    counts, _ = np.histogram(high_waterlevel, bins = elevation_bins)
    inundation_freq = 1 - np.cumsum(counts/counts.sum())

    counts, _ = np.histogram(low_waterlevel, bins = elevation_bins)
    emergence_freq = np.cumsum(counts/counts.sum())

    if plot:
        plt.plot(elevation, emergence_freq, label = 'Frequency of emergence')
        plt.plot(elevation, inundation_freq, label = 'Frequency of inundation')
        plt.ylabel('Frequency of inundation/emergence', fontsize = 12)
        plt.xlabel('Elevation (m MSL)', fontsize = 12)
        plt.legend(frameon = False, fontsize = 11)
        plt.show()

    return elevation, emergence_freq, inundation_freq

def plot_figure(elevation, emergence_freq, inundation_freq):

    def identify_intertidal_zones(elevation, emergence_freq, inundation_freq):

        supratidal_boundary = elevation[(inundation_freq >= 0.01).argmin()]
        upper_intertidal_boundary = elevation[(inundation_freq <= 0.99).argmax()]
        lower_intertidal_boundary = elevation[(emergence_freq <= 0.99).argmin()]
        subtidal_boundary = elevation[(emergence_freq >= 0.01).argmax()]
        print(supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary)
        return supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary

    supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = identify_intertidal_zones(elevation, emergence_freq, inundation_freq)

    # intermediate_intertidal = (emergence_freq >= 0.99) & (inundation_freq >= 0.99)
    plt.plot([lower_intertidal_boundary, upper_intertidal_boundary], [-0.02,-0.02], alpha = 1, c = 'black')

    upper_intertidal = ((inundation_freq < 0.99) & (inundation_freq > 0.01))
    plt.plot([supratidal_boundary, upper_intertidal_boundary], [1.035,1.035], alpha = 1, c = 'orange')
    plt.fill_between(elevation, upper_intertidal, color = 'orange', alpha = 0.05)

    lower_intertidal = ((emergence_freq < 0.99) & (emergence_freq > 0.01))
    plt.plot([lower_intertidal_boundary, subtidal_boundary], [1.02,1.02], alpha = 1, c = 'blue')
    plt.fill_between(elevation, lower_intertidal, color = 'blue', alpha = 0.05)

    # plt.text( 3.33, -0.050, rotation = 0, s = 'Inundation', c = 'orange', fontsize = 11, horizontalalignment = 'right',  verticalalignment = 'center')
    # plt.text(-3.33, -0.050, rotation = 0, s = 'Emergence',  c = 'blue',   fontsize = 11, horizontalalignment = 'left',   verticalalignment = 'center')

    plt.plot(elevation, emergence_freq, c = 'blue', alpha = 1, label = 'Emergence')
    plt.plot(elevation, inundation_freq, c = 'orange', alpha = 1, label = 'Inundation')
    plt.ylabel('Frequency of inundation/emergence [-]', fontsize = 12)
    plt.xlabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.legend(frameon = False, fontsize = 11)
    # plt.ylim(-0.17,1.17)
    # plt.yticks(np.arange(0,1 + 0.2,0.2))
    # plt.xlim(-3.4, 3.4)
    plt.show()

def plot_figure_2(elevation, emergence_freq, inundation_freq):

    def identify_intertidal_zones(elevation, emergence_freq, inundation_freq):

        supratidal_boundary = elevation[(inundation_freq >= 0.01).argmin()]
        upper_intertidal_boundary = elevation[(inundation_freq <= 0.99).argmax()]
        lower_intertidal_boundary = elevation[(emergence_freq <= 0.99).argmin()]
        subtidal_boundary = elevation[(emergence_freq >= 0.01).argmax()]
        print(supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary)
        return supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary

    for i in range(emergence_freq.shape[0]):

        supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = identify_intertidal_zones(elevation, emergence_freq[i,:], inundation_freq[i,:])
        upper_intertidal = ((inundation_freq[i,:] < 0.99) & (inundation_freq[i,:] > 0.01))
        lower_intertidal = ((emergence_freq[i,:] < 0.99) & (emergence_freq[i,:] > 0.01))

        plt.fill_between(elevation, upper_intertidal, color = 'orange', alpha = 0.05)
        plt.fill_between(elevation, lower_intertidal, color = 'blue', alpha = 0.05)

        if i == 0:
            plt.plot(elevation, emergence_freq[i,:], c = 'blue', alpha = 1, label = 'Emergence')
            plt.plot(elevation, inundation_freq[i,:], c = 'orange', alpha = 1, label = 'Inundation')

            plt.plot([lower_intertidal_boundary, upper_intertidal_boundary], [-0.02,-0.02], alpha = 1, c = 'black')
            plt.plot([supratidal_boundary, upper_intertidal_boundary], [1.035,1.035], alpha = 1, c = 'orange')
            plt.plot([lower_intertidal_boundary, subtidal_boundary], [1.02,1.02], alpha = 1, c = 'blue')
        else:
            plt.plot(elevation, emergence_freq[i,:], c = 'blue', alpha = 0.5, linestyle = 'dashed')
            plt.plot(elevation, inundation_freq[i,:], c = 'orange', alpha = 0.5, linestyle = 'dashed')

            plt.plot([lower_intertidal_boundary, upper_intertidal_boundary], [-0.02,-0.02], c = 'black', alpha = 0.5, linestyle = 'dashed')
            plt.plot([supratidal_boundary, upper_intertidal_boundary], [1.035,1.035], c = 'orange', alpha = 0.5, linestyle = 'dashed')
            plt.plot([lower_intertidal_boundary, subtidal_boundary], [1.02,1.02], c = 'blue', alpha = 0.5, linestyle = 'dashed')

    plt.ylabel('Frequency of inundation/emergence [-]', fontsize = 12)
    plt.xlabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.legend(frameon = False, fontsize = 11)
    # plt.ylim(-0.17,1.17)
    # plt.yticks(np.arange(0,1 + 0.2,0.2))
    # plt.xlim(-3.4, 3.4)
    plt.show()
