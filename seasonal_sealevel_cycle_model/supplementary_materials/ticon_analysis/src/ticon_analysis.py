import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functions as f

def ticon_wlts(path_csv = './supplementary_materials/supplementary_note_1/data/csv/TICON.txt'):

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
    # i = 392
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
    # plt.plot(t, w)
    # plt.xlabel('Time (hours)', fontsize = 12)
    # plt.ylabel('Water level (cm)', fontsize = 12)
    # plt.show()

    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(w, dt = dt, plot = True)
    elevation, emergence_freq, inundation_freq = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel, dz = 0.01, plot = False)
    f.plot_figure(elevation, emergence_freq, inundation_freq)

ticon_wlts()