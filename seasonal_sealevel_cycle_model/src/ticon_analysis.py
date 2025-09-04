import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functions as f
from matplotlib.patches import Rectangle

def ticon_wlts(path_csv = './supplementary_materials/ticon_analysis/data/csv/TICON.txt'):

    tidal_constants = np.array([
        'M2', 'K1', 'N2', 'O1', 'P1', 'Q1', 'K2', 'S2', 'S1', 'SA', 'T2', 'MF', 'MM',
        '2N2', 'M4', 'J1', 'SSA', 'MSF', 'MSQ', 'EP2', 'L2', 'M3', 'R2', 'MI2', 'MTM',
        'NI2', 'LM2', 'MN4', 'MS4', 'MKS', 'N4', 'M6', 'M8', 'S4', '2Q1', 'OO1', 'S3',
        'MA2', 'MB2', 'M1'
    ])
    period_hr = np.array([
        12.4206, 23.9345, 12.6583, 25.8193, 24.0659, 26.8684, 11.9672, 12.0000, 24.0000,
        8766.0, 12.0160, 327.85, 661.31, 12.9054, 6.2103, 23.0984, 4383.0, 354.37, 6.1328,
        12.3283, 12.1916, 8.2804, 11.8700, 12.8729, 6.1033, 12.8864, 12.6321, 6.2692,
        6.1033, 6.0769, 6.3293, 4.1402, 3.1052, 6.0000, 27.8454, 22.3061, 8.0000, 12.8714,
        12.5660, 23.9340
    ])

    df = pd.read_csv(path_csv, delim_whitespace=True)
    df.columns = ['latitude', 'longitude', 'tidal_constituent', 'amplitude_cm', 'phase_degrees', 'amplitude-std_cm', 'phase-std_degrees', 'missing_data_percent', 'total_observations', 'time_gap_max', 'start_date', 'end_date', 'data_source']

    df['site_id'] = df.groupby(['latitude', 'longitude', 'data_source', 'start_date', 'end_date']).ngroup()
    site_list = np.unique(df['site_id'])
    n_sites = site_list.size

    i = np.random.choice(n_sites)
    i = 57#392
    print(i)
    site = df['site_id'] == i
    df_site = df[site]
    amplitude_m  = df_site['amplitude_cm'].to_numpy() / 100
    phase_rad  = np.deg2rad(df_site['phase_degrees'].to_numpy())

    # subset = amplitude_m > 0.1
    # amplitude_m = amplitude_m[subset]
    # phase_rad = phase_rad[subset]
    # period_hr = period_hr[subset]
    # print(tidal_constants[subset])

    n_years = 10
    dt = 5/60
    tf    = 365.25 * 24 * n_years # 1 year (in hours)
    t = np.arange(0, tf + dt, dt)
    
    w = (amplitude_m[np.newaxis, :] * np.cos(t[:, np.newaxis]*2*np.pi/period_hr[np.newaxis, :] + phase_rad[np.newaxis, :])).sum(axis = 1)

    amplitude_annual_t0_m = 0#0.25
    amplitude_annual_t1_m = 1#0.5
    period_annual_hr = 365.25 * 24

    annual_cycle_0 = amplitude_annual_t0_m * np.cos(t*2*np.pi/period_annual_hr)
    annual_cycle_1 = amplitude_annual_t1_m * np.cos(t*2*np.pi/period_annual_hr)
    w0 = w + annual_cycle_0
    w1 = w + annual_cycle_1

    # plt.plot(t, w)
    # plt.xlabel('Time (hours)', fontsize = 12)
    # plt.ylabel('Water level (cm)', fontsize = 12)
    # plt.show()

    _, low_waterlevel_0, high_waterlevel_0 = f.calculate_highlow_water(w0, dt = dt, plot = False)
    _, low_waterlevel_1, high_waterlevel_1 = f.calculate_highlow_water(w1, dt = dt, plot = False)
    elevation, emergence_freq_0, inundation_freq_0 = f.calculate_inundation_metrics(low_waterlevel_0, high_waterlevel_0, dz = 0.01, plot = False)
    elevation, emergence_freq_1, inundation_freq_1 = f.calculate_inundation_metrics(low_waterlevel_1, high_waterlevel_1, dz = 0.01, plot = False)

    dz = np.round(np.diff(elevation).mean(), 3)
    
    emergence_freq = np.vstack((emergence_freq_0, emergence_freq_1))
    inundation_freq = np.vstack((inundation_freq_0, inundation_freq_1))

    f.plot_figure_2(elevation, emergence_freq, inundation_freq)

    fig, ax = plt.subplots()
    for i in range(emergence_freq.shape[0]):
        upper_intertidal         = (emergence_freq [i,:] > 0.99) & ((inundation_freq[i,:] < 0.99) & (inundation_freq[i,:] > 0.01))
        intermediate_intertidal  = (inundation_freq[i,:] > 0.99) & (emergence_freq [i,:] > 0.99)
        lower_intertidal         = ((emergence_freq [i,:] < 0.99) & (emergence_freq [i,:] > 0.01)) & (inundation_freq[i,:] > 0.99)
        transitioning_intertidal = ((inundation_freq[i,:] < 0.99) & (inundation_freq[i,:] > 0.01)) & ((emergence_freq [i,:] < 0.99) & (emergence_freq [i,:] > 0.01))

        y = np.cumsum([0, lower_intertidal.sum(), intermediate_intertidal.sum(), transitioning_intertidal.sum(), upper_intertidal.sum()])*dz
        y = y - y.max()/2
        lower = Rectangle((i*0.125 - 0.1, y[0]), 0.1, y[1] - y[0], edgecolor = (0.0, 0.0, 1.0, 1),   facecolor = (0.0, 0.0, 1.0, 0.1), lw = 2)
        ax.add_patch(lower)
        upper = Rectangle((i*0.125 - 0.1, y[3]), 0.1, y[4] - y[3], edgecolor = (1.0, 0.647, 0.0, 1), facecolor = (1.0, 0.647, 0.0, 0.1), lw = 2)
        ax.add_patch(upper)
        if (y[2] - y[1]) > 0:
            inter = Rectangle((i*0.125 - 0.1, y[1]), 0.1, y[2] - y[1], edgecolor = (0.5, 0.5, 0.5, 1),   facecolor = (0.5, 0.5, 0.5, 0.1), lw = 2)
            ax.add_patch(inter)
        if (y[3] - y[2]) > 0:
            trans = Rectangle((i*0.125 - 0.1, y[1]), 0.1, y[3] - y[2], edgecolor = (0.1, 0.1, 0.1, 1),   facecolor = (0.1, 0.1, 0.1, 0.4), lw = 2)
            ax.add_patch(trans)
    # plt.ylim(-y[4]*1.55, y[4]*1.55)
    plt.ylim(elevation.min(), elevation.max())
    plt.xlim(-1, 1)
    plt.xticks(ticks = [0], labels = ['Mediterranean Sea'])
    plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.xlabel('Location', fontsize = 12)
    plt.show()


ticon_wlts()