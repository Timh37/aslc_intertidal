import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import src.functions as f

def figure_2a(ax, path_bin = './bin/figure_2/figure_2a.npy', path_png = None, run = False):

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

    def plot_figure_2(ax,path_bin, path_png):

        amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = np.load(path_bin)

        # Figure 2b
        ax.plot(amplitude_annual_cycle, supratidal_boundary, c = 'orange', alpha = 0.5)
        ax.plot(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange')
        ax.plot(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue')
        ax.plot(amplitude_annual_cycle, subtidal_boundary, c = 'blue', alpha = 0.5)
        ax.scatter(amplitude_annual_cycle, supratidal_boundary, c = 'orange', s = 1, alpha = 0.5)
        ax.scatter(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange', s = 1, alpha = 1.0)
        ax.scatter(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue', s = 1, alpha = 1.0)
        ax.scatter(amplitude_annual_cycle, subtidal_boundary, c = 'blue', s = 1, alpha = 0.5)

        def closest_to(x, seq):
            return np.argmin(np.abs(seq - x))
        
        transition_point = amplitude_annual_cycle[closest_to(upper_intertidal_boundary, 0)]

        ax.fill_between(x = amplitude_annual_cycle[amplitude_annual_cycle <= transition_point], 
                        y1 = (upper_intertidal_boundary)[amplitude_annual_cycle <= transition_point], 
                        y2 = (lower_intertidal_boundary)[amplitude_annual_cycle <= transition_point], color = 'grey', alpha = 0.2)
        ax.fill_between(x = amplitude_annual_cycle[amplitude_annual_cycle > transition_point], 
                        y1 = (upper_intertidal_boundary)[amplitude_annual_cycle > transition_point], 
                        y2 = (lower_intertidal_boundary)[amplitude_annual_cycle > transition_point], color = 'grey', alpha = 0.7)
        ax.fill_between(x = amplitude_annual_cycle, 
                        y1 = (supratidal_boundary), 
                        y2 = np.maximum(lower_intertidal_boundary, upper_intertidal_boundary), color = 'orange', alpha = 0.2)
        ax.fill_between(x = amplitude_annual_cycle, 
                        y1 = (subtidal_boundary), 
                        y2 = np.minimum(lower_intertidal_boundary, upper_intertidal_boundary), color = 'blue', alpha = 0.2)

        ax.text(x = amplitude_annual_cycle.max()*0.5, y =  1.25, s = 'Supratidal zone',       c = 'black',  rotation =  7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        ax.text(x = amplitude_annual_cycle.max()*0.5, y =  0.90, s = 'Upper intertidal zone', c = 'orange', rotation =  7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        ax.text(x = amplitude_annual_cycle.max()*0.5, y = -0.95, s = 'Lower intertidal zone', c = 'blue',   rotation = -7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        ax.text(x = amplitude_annual_cycle.max()*0.5, y = -1.30, s = 'Subtidal zone',         c = 'black',  rotation = -7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        ax.text(x = 0.02, y = -0.03, s = 'Stable zone',     c = 'black',  rotation =   3, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center')
        ax.text(x = amplitude_annual_cycle.max() - 0.02, y = -0.15, s = 'Seasonally-transitioning zone', c = 'black',  rotation = -7, fontsize = 10, horizontalalignment = 'right', verticalalignment = 'center')
        ax.set_ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        ax.set_xlabel('ASLC range / Tidal range [-]', fontsize = 12)
        ax.set_ylim(-2.1, 2.1)
        #if path_png is not None:
        #    plt.savefig(path_png, dpi = 300)
        #plt.show()

    if run:
        run_simulation_figure_2(path_bin)
    plot_figure_2(ax,path_bin, path_png)

def figure_2c(ax, path_csv = 'data/ticon/TICON.txt', path_png = None):

    def create_ticon_timeseries(path_csv, lat, lon):

        def calculate_tidal_range(w):

            nx = 149
            ny = int(np.floor(w.size/nx))
            w = w[:(nx*ny)].reshape(nx,ny)
            lwl = w.min(axis = 0)
            hwl = w.max(axis = 0)
            tidal_amplitude = np.mean(hwl - lwl)
            print(tidal_amplitude)

            return tidal_amplitude

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

        df = pd.read_csv(path_csv, sep='\s+')
        df.columns = ['latitude', 'longitude', 'tidal_constituent', 'amplitude_cm', 'phase_degrees', 'amplitude-std_cm', 'phase-std_degrees', 'missing_data_percent', 'total_observations', 'time_gap_max', 'start_date', 'end_date', 'data_source']

        df['site_id'] = df.groupby(['latitude', 'longitude', 'data_source', 'start_date', 'end_date']).ngroup()
        site_list = np.unique(df['site_id'])
        n_sites = site_list.size

        def closest_to(x0, y0, x1, y1):
            return np.argmin(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
        idx = closest_to(df['latitude'], df['longitude'],  lat, lon)

        site = df['site_id'] == df['site_id'][idx]
        df_site = df[site]
        amplitude_m  = df_site['amplitude_cm'].to_numpy() / 100
        phase_rad  = np.deg2rad(df_site['phase_degrees'].to_numpy())

        n_years = 10
        dt = 5/60
        tf    = 365.25 * 24 * n_years # 1 year (in hours)
        t = np.arange(0, tf + dt, dt)
        
        w = (amplitude_m[np.newaxis, :] * np.cos(t[:, np.newaxis]*2*np.pi/period_hr[np.newaxis, :] + phase_rad[np.newaxis, :])).sum(axis = 1)
        tidal_range = calculate_tidal_range(w)
        w = w/tidal_range

        return t, w

    def add_aslc(time, waterlevel, amplitude_annual_m, phase):

        period_annual_hr = 365.25 * 24
        waterlevel_new = waterlevel + amplitude_annual_m * np.cos(time*2*np.pi/period_annual_hr + phase)

        return waterlevel_new

    def add_boxplot(ax, pos, elevation, emergence_freq, inundation_freq, path_png = None):
    
        dz = np.round(np.diff(elevation).mean(), 3)    

        for i in range(emergence_freq.shape[0]):
            upper_intertidal         = (emergence_freq [i,:] > 0.99) & ((inundation_freq[i,:] < 0.99) & (inundation_freq[i,:] > 0.01))
            intermediate_intertidal  = (inundation_freq[i,:] > 0.99) & (emergence_freq [i,:] > 0.99)
            lower_intertidal         = ((emergence_freq [i,:] < 0.99) & (emergence_freq [i,:] > 0.01)) & (inundation_freq[i,:] > 0.99)
            transitioning_intertidal = ((inundation_freq[i,:] < 0.99) & (inundation_freq[i,:] > 0.01)) & ((emergence_freq [i,:] < 0.99) & (emergence_freq [i,:] > 0.01))

            y = np.cumsum([0, lower_intertidal.sum(), intermediate_intertidal.sum(), transitioning_intertidal.sum(), upper_intertidal.sum()])*dz
            y -= y.max()/2
            lower = Rectangle((i*0.125 - 0.1 + pos*0.5, y[0]), 0.1, y[1] - y[0], edgecolor = (0.0, 0.0, 1.0, 1),   facecolor = (0.0, 0.0, 1.0, 0.1), lw = 2)
            ax.add_patch(lower)
            upper = Rectangle((i*0.125 - 0.1 + pos*0.5, y[3]), 0.1, y[4] - y[3], edgecolor = (1.0, 0.647, 0.0, 1), facecolor = (1.0, 0.647, 0.0, 0.1), lw = 2)
            ax.add_patch(upper)
            if (y[2] - y[1]) > 0:
                inter = Rectangle((i*0.125 - 0.1 + pos*0.5, y[1]), 0.1, y[2] - y[1], edgecolor = (0.5, 0.5, 0.5, 1),   facecolor = (0.5, 0.5, 0.5, 0.1), lw = 2)
                ax.add_patch(inter)
            if (y[3] - y[2]) > 0:
                trans = Rectangle((i*0.125 - 0.1 + pos*0.5, y[1]), 0.1, y[3] - y[2], edgecolor = (0.1, 0.1, 0.1, 1),   facecolor = (0.1, 0.1, 0.1, 0.4), lw = 2)
                ax.add_patch(trans)

    rows, cols = 1, 1
    fig = plt.figure(figsize=((6.4 + 0.3)*cols,(4.8 + 0.3)*rows))
    gs1 = fig.add_gridspec(rows,cols,hspace=.15, wspace=.15) #define subplot grid
    ax = fig.add_subplot(gs1[0,0])

    for i in range(3):

        if i == 0:
            lat, lon, future_aslc_change, phase = -17.533, -149.573, 0.0724, 0.25*2*np.pi
        elif i == 1:
            lat, lon, future_aslc_change, phase =  41.342,    2.166, 0.1192, 0.7*2*np.pi
        elif i == 2:
            lat, lon, future_aslc_change, phase =  35.548,  133.243, 0.0603, 0.65*2*np.pi

        time, waterlevel_t0 = create_ticon_timeseries(path_csv, lat, lon)
        waterlevel_t1 = add_aslc(time, waterlevel_t0, amplitude_annual_m = future_aslc_change, phase = phase)

        dt = np.diff(time).mean()
        _, low_waterlevel_0, high_waterlevel_0 = f.calculate_highlow_water(waterlevel_t0, dt = dt, plot = False)
        _, low_waterlevel_1, high_waterlevel_1 = f.calculate_highlow_water(waterlevel_t1, dt = dt, plot = False)
        elevation, emergence_freq_0, inundation_freq_0 = f.calculate_inundation_metrics(low_waterlevel_0, high_waterlevel_0, dz = 0.01, plot = False)
        _,         emergence_freq_1, inundation_freq_1 = f.calculate_inundation_metrics(low_waterlevel_1, high_waterlevel_1, dz = 0.01, plot = False)
        emergence_freq = np.vstack((emergence_freq_0, emergence_freq_1))
        inundation_freq = np.vstack((inundation_freq_0, inundation_freq_1))

        add_boxplot(ax, i, elevation, emergence_freq, inundation_freq, path_png)

    # ax.set_ylim(elevation.min(), elevation.max())
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(ticks = [0*0.5,1*0.5,2*0.5], labels = ['Pacific', 'Mediterranean Sea', 'Japan Sea'])
    ax.set_ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    ax.set_xlabel('Location', fontsize = 12)
    ax.plot([-0.5, 1.5], [0.5,0.5])
    ax.plot([-0.5, 1.5], [-0.5,-0.5])
    if path_png is not None:
        plt.savefig(path_png, dpi = 300)
    plt.show()

ax = None
#figure_2a(ax, path_bin = './bin/figure_2/figure_2a.npy', path_png = './figures/figure_2/figure_2a.png', run = False)
figure_2c(ax, path_csv = './data/ticon/TICON.txt', path_png = './figures/figure_2/figure_2b.png')