import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import src.functions as f

def figure_2a(path_bin = './bin/figure_2/figure_2a.npy', path_png = None, run = False):

    def run_simulation_figure_2(path_bin):

        def identify_intertidal_zones(amplitude_annual_cycle, n_years, dz = 0.01):

            def closest_to(x, seq):
                return np.argmin(np.abs(seq - x))

            _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = amplitude_annual_cycle, n_years = n_years)
            water_level = water_level/2 # Convert from amplitude to tidal range
            _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
            elevation, emergence_freq, inundation_freq = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel, dz)

            supratidal_boundary = elevation[closest_to(0.01, inundation_freq)]
            upper_intertidal_boundary = elevation[closest_to(0.99, inundation_freq)]
            lower_intertidal_boundary = elevation[closest_to(0.99, emergence_freq)]
            subtidal_boundary = elevation[closest_to(0.01, emergence_freq)]
            
            total_intertidal = ((emergence_freq > 0.01) & (inundation_freq > 0.01)).sum()*dz

            return total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary

        n_years = 50
        n = 100
        amplitude_annual_cycle = np.linspace(0, 1.5, n)
        total_intertidal          = np.array([np.nan]*n)
        supratidal_boundary       = np.array([np.nan]*n)
        upper_intertidal_boundary = np.array([np.nan]*n)
        lower_intertidal_boundary = np.array([np.nan]*n)
        subtidal_boundary         = np.array([np.nan]*n)
        for i in range(n):
            total_intertidal[i], supratidal_boundary[i], upper_intertidal_boundary[i], lower_intertidal_boundary[i], subtidal_boundary[i] = identify_intertidal_zones(amplitude_annual_cycle[i], n_years)
            print(i)

        np.save(path_bin, (amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary))

    def plot_figure_2(path_bin, path_png):

        amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = np.load(path_bin)

        # Figure 2b
        plt.plot(amplitude_annual_cycle, supratidal_boundary, c = 'orange', alpha = 0.5)
        plt.plot(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange')
        plt.plot(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue')
        plt.plot(amplitude_annual_cycle, subtidal_boundary, c = 'blue', alpha = 0.5)
        plt.scatter(amplitude_annual_cycle, supratidal_boundary, c = 'orange', s = 1, alpha = 0.5)
        plt.scatter(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange', s = 1, alpha = 1.0)
        plt.scatter(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue', s = 1, alpha = 1.0)
        plt.scatter(amplitude_annual_cycle, subtidal_boundary, c = 'blue', s = 1, alpha = 0.5)

        def closest_to(x, seq):
            return np.argmin(np.abs(seq - x))
        
        transition_point = amplitude_annual_cycle[closest_to(upper_intertidal_boundary, 0)]

        plt.fill_between(x = amplitude_annual_cycle[amplitude_annual_cycle <= transition_point], 
                        y1 = (upper_intertidal_boundary)[amplitude_annual_cycle <= transition_point], 
                        y2 = (lower_intertidal_boundary)[amplitude_annual_cycle <= transition_point], color = 'grey', alpha = 0.2)
        plt.fill_between(x = amplitude_annual_cycle[amplitude_annual_cycle > transition_point], 
                        y1 = (upper_intertidal_boundary)[amplitude_annual_cycle > transition_point], 
                        y2 = (lower_intertidal_boundary)[amplitude_annual_cycle > transition_point], color = 'grey', alpha = 0.7)
        plt.fill_between(x = amplitude_annual_cycle, 
                        y1 = (supratidal_boundary), 
                        y2 = np.maximum(lower_intertidal_boundary, upper_intertidal_boundary), color = 'orange', alpha = 0.2)
        plt.fill_between(x = amplitude_annual_cycle, 
                        y1 = (subtidal_boundary), 
                        y2 = np.minimum(lower_intertidal_boundary, upper_intertidal_boundary), color = 'blue', alpha = 0.2)

        plt.text(x = amplitude_annual_cycle.max()*0.5, y =  1.25, s = 'Supratidal zone',       c = 'black',  rotation =  7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max()*0.5, y =  0.90, s = 'Upper intertidal zone', c = 'orange', rotation =  7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max()*0.5, y = -0.95, s = 'Lower intertidal zone', c = 'blue',   rotation = -7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max()*0.5, y = -1.30, s = 'Subtidal zone',         c = 'black',  rotation = -7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = 0.02, y = -0.03, s = 'Stable zone',     c = 'black',  rotation =   3, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max() - 0.02, y = -0.15, s = 'Seasonally-transitioning zone', c = 'black',  rotation = -7, fontsize = 10, horizontalalignment = 'right', verticalalignment = 'center')
        plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        plt.xlabel('ASLC range / Tidal range [-]', fontsize = 12)
        plt.ylim(-2.1, 2.1)
        if path_png is not None:
            plt.savefig(path_png, dpi = 300)
        plt.show()

    if run:
        run_simulation_figure_2(path_bin)
    plot_figure_2(path_bin, path_png)

def figure_2b(path_csv = 'data/ticon/TICON.txt', path_png = None):

    def create_ticon_timeseries(path_csv):

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

        n_years = 10
        dt = 5/60
        tf    = 365.25 * 24 * n_years # 1 year (in hours)
        t = np.arange(0, tf + dt, dt)
        
        w = (amplitude_m[np.newaxis, :] * np.cos(t[:, np.newaxis]*2*np.pi/period_hr[np.newaxis, :] + phase_rad[np.newaxis, :])).sum(axis = 1)

        return t, w

    def add_aslc(time, waterlevel, amplitude_annual_m):

        period_annual_hr = 365.25 * 24
        waterlevel_new = waterlevel + amplitude_annual_m * np.cos(time*2*np.pi/period_annual_hr)

        return waterlevel_new

    def plot_test_figure(elevation, emergence_freq, inundation_freq):

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

    def plot_boxplot_figure(elevation, emergence_freq, inundation_freq, path_png):
    
        dz = np.round(np.diff(elevation).mean(), 3)    

        fig, ax = plt.subplots()
        for i in range(emergence_freq.shape[0]):
            upper_intertidal         = (emergence_freq [i,:] > 0.99) & ((inundation_freq[i,:] < 0.99) & (inundation_freq[i,:] > 0.01))
            intermediate_intertidal  = (inundation_freq[i,:] > 0.99) & (emergence_freq [i,:] > 0.99)
            lower_intertidal         = ((emergence_freq [i,:] < 0.99) & (emergence_freq [i,:] > 0.01)) & (inundation_freq[i,:] > 0.99)
            transitioning_intertidal = ((inundation_freq[i,:] < 0.99) & (inundation_freq[i,:] > 0.01)) & ((emergence_freq [i,:] < 0.99) & (emergence_freq [i,:] > 0.01))

            y = np.cumsum([0, lower_intertidal.sum(), intermediate_intertidal.sum(), transitioning_intertidal.sum(), upper_intertidal.sum()])*dz
            y -= y.max()/2
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
        plt.ylim(elevation.min(), elevation.max())
        plt.xlim(-1, 1)
        plt.xticks(ticks = [0], labels = ['Mediterranean Sea'])
        plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        plt.xlabel('Location', fontsize = 12)
        if path_png is not None:
            plt.savefig(path_png, dpi = 300)
        plt.show()

    time, waterlevel = create_ticon_timeseries(path_csv)
    waterlevel_t0 = add_aslc(time, waterlevel, amplitude_annual_m = 0)
    waterlevel_t1 = add_aslc(time, waterlevel, amplitude_annual_m = 1)

    dt = np.diff(time).mean()
    _, low_waterlevel_0, high_waterlevel_0 = f.calculate_highlow_water(waterlevel_t0, dt = dt, plot = False)
    _, low_waterlevel_1, high_waterlevel_1 = f.calculate_highlow_water(waterlevel_t1, dt = dt, plot = False)
    elevation, emergence_freq_0, inundation_freq_0 = f.calculate_inundation_metrics(low_waterlevel_0, high_waterlevel_0, dz = 0.01, plot = False)
    _,         emergence_freq_1, inundation_freq_1 = f.calculate_inundation_metrics(low_waterlevel_1, high_waterlevel_1, dz = 0.01, plot = False)
    emergence_freq = np.vstack((emergence_freq_0, emergence_freq_1))
    inundation_freq = np.vstack((inundation_freq_0, inundation_freq_1))

    plot_test_figure(elevation, emergence_freq, inundation_freq)
    plot_boxplot_figure(elevation, emergence_freq, inundation_freq, path_png)

figure_2a(path_bin = './bin/figure_2/figure_2a.npy', path_png = './figures/figure_2/figure_2a.png', run = False)
figure_2b(path_csv = './data/ticon/TICON.txt', path_png = './figures/figure_2/figure_2b.png')