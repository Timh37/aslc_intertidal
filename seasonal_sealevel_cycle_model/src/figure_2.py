import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functions as f

def figure_2(path_bin = './bin/figure_2.npy', run = False, save = True):

    def closest_to(x, seq, axis = None):
        if axis is not None:
            return np.argmin(np.abs(seq[np.newaxis,:] - x[:,np.newaxis]), axis = axis)
        else:
            return np.argmin(np.abs(seq - x))

    def run_figure_2_simulation(path_bin):

        def identify_intertidal_zones(amplitude_annual_cycle, n_years, dz = 0.01):

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
        total_intertidal = np.array([np.nan]*n)
        supratidal_boundary       = np.array([np.nan]*n)
        upper_intertidal_boundary = np.array([np.nan]*n)
        lower_intertidal_boundary = np.array([np.nan]*n)
        subtidal_boundary         = np.array([np.nan]*n)
        for i in range(n):
            total_intertidal[i], supratidal_boundary[i], upper_intertidal_boundary[i], lower_intertidal_boundary[i], subtidal_boundary[i] = identify_intertidal_zones(amplitude_annual_cycle[i], n_years)
            print(i)

        np.save(path_bin, (amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary))

    def plot_figure_2(path_bin, save):

        amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = np.load(path_bin)

        # Load ASLC global sites dataset
        df = pd.read_csv('./csv/sites_table.csv', nrows = 3)
        historical = np.array(df['historical_ASLC_range/tidal_range'])
        change = np.array(df['change_ASLC_range/tidal_range(SSP3-7.0)'])
        future = historical + change

        # Figure 2b
        plt.plot(amplitude_annual_cycle, supratidal_boundary, c = 'orange', alpha = 0.5)
        plt.plot(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange')
        plt.plot(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue')
        plt.plot(amplitude_annual_cycle, subtidal_boundary, c = 'blue', alpha = 0.5)
        plt.scatter(amplitude_annual_cycle, supratidal_boundary, c = 'orange', s = 1, alpha = 0.5)
        plt.scatter(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange', s = 1, alpha = 1.0)
        plt.scatter(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue', s = 1, alpha = 1.0)
        plt.scatter(amplitude_annual_cycle, subtidal_boundary, c = 'blue', s = 1, alpha = 0.5)

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

        # # Annotations
        # x0 = amplitude_annual_cycle[closest_to(historical, amplitude_annual_cycle, axis = 1)]
        # x1 = amplitude_annual_cycle[closest_to(future, amplitude_annual_cycle, axis = 1)]
        # y0_high = supratidal_boundary[closest_to(historical, amplitude_annual_cycle, axis = 1)]
        # y0_low = subtidal_boundary[closest_to(historical, amplitude_annual_cycle, axis = 1)]
        # y1_high = supratidal_boundary[closest_to(future, amplitude_annual_cycle, axis = 1)]
        # y1_low = subtidal_boundary[closest_to(future, amplitude_annual_cycle, axis = 1)]

 
        # plt.fill_between(x = [x0[0],x1[0]], y1 = [y0_high[0],y1_high[0]], y2 = [y0_low[0], y1_low[0]], color = 'white', alpha = 0.5)
        # plt.fill_between(x = [x0[1],x1[1]], y1 = [y0_high[1],y1_high[1]], y2 = [y0_low[1], y1_low[1]], color = 'white', alpha = 0.5)
        # plt.fill_between(x = [x0[2],x1[2]], y1 = [y0_high[2],y1_high[2]], y2 = [y0_low[2], y1_low[2]], color = 'white', alpha = 0.5)
        # plt.plot([x0,x0], [y0_low,y0_high], c = 'black', zorder = 1)
        # plt.plot([x1,x1], [y1_low,y1_high], c = 'black', zorder = 1)
        # plt.annotate('', xy = (x1[0],0), xytext = (x0[0],0),
        #             arrowprops=dict(arrowstyle='->', color='black', lw = 1))
        # plt.annotate('', xy = (x1[1],0), xytext = (x0[1],0),
        #             arrowprops=dict(arrowstyle='->', color='black', lw = 1))
        # plt.annotate('', xy = (x1[2],0), xytext = (x0[2],0),
        #             arrowprops=dict(arrowstyle='->', color='black', lw = 1))

        # xmean = (x0 + x1)/2
        # plt.text(x = xmean[0], y = 1.9, s = '$Mediterranean$ $Sea$', rotation = 0, verticalalignment = 'center', horizontalalignment = 'center')
        # plt.text(x = xmean[1], y = 1.9, s = '$Japanese/East$ $Sea$', rotation = 0, verticalalignment = 'center', horizontalalignment = 'center')
        # plt.text(x = xmean[2], y = 1.9, s = '$South$ $Pacific$',     rotation = 0, verticalalignment = 'center', horizontalalignment = 'center')

        # ystart = [1.8,1.8,1.8]
        # ymean_high = (y0_high + y1_high)/2
        # plt.plot([xmean, xmean], [ystart, ymean_high], c = 'black', linestyle = 'dotted')

        # plt.text(x = (x0[2] + x1[2])/2, y = 1.2, s = '$South$ $Pacific$',      rotation = 10, verticalalignment = 'center', horizontalalignment = 'center')
        # plt.text(x = (x0[0] + x1[0])/2, y = 1.5, s = '$Mediterranean$ $Sea$',  rotation = 10, verticalalignment = 'center', horizontalalignment = 'center')
        # plt.text(x = (x0[1] + x1[1])/2, y = 1.85, s = '$Japanese/East$ $Sea$', rotation = 10, verticalalignment = 'center', horizontalalignment = 'center')

        plt.text(x = amplitude_annual_cycle.max()*0.5, y =  1.25, s = 'Supratidal zone',       c = 'black',  rotation =  7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max()*0.5, y =  0.90, s = 'Upper intertidal zone', c = 'orange', rotation =  7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max()*0.5, y = -0.95, s = 'Lower intertidal zone', c = 'blue',   rotation = -7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max()*0.5, y = -1.30, s = 'Subtidal zone',         c = 'black',  rotation = -7, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(x = 0.02, y = -0.03, s = 'Stable zone',     c = 'black',  rotation =   3, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center')
        plt.text(x = amplitude_annual_cycle.max() - 0.02, y = -0.15, s = 'Seasonally-transitioning zone', c = 'black',  rotation = -7, fontsize = 10, horizontalalignment = 'right', verticalalignment = 'center')
        plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        plt.xlabel('ASLC range / Tidal range [-]', fontsize = 12)
        plt.ylim(-2.1, 2.1)
        if save is True:
            plt.savefig('./figures/figure_2-alt.png', dpi = 300)
        plt.show()

    if run:
        run_figure_2_simulation(path_bin)
    plot_figure_2(path_bin, save)

figure_2(path_bin = './bin/figure_2.npy', run = False, save = True)



