import numpy as np
import matplotlib.pyplot as plt
import functions as f
import os
from scipy.optimize import curve_fit

def figure_s1(plot = False, path_bin = None, path_png = None, amplitude_M2 = 1, amplitude_S2 = 0.4, n_years = 10):

    def closest_to(x, seq, axis = None):
        if axis is not None:
            return np.argmin(np.abs(seq[np.newaxis,:] - x[:,np.newaxis]), axis = axis)
        else:
            return np.argmin(np.abs(seq - x))

    def identify_intertidal_zones(amplitude_annual_cycle, amplitude_M2, amplitude_S2, n_years, dz = 0.01):

        _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = amplitude_annual_cycle, amplitude_M2 = amplitude_M2, amplitude_S2 = amplitude_S2, n_years = n_years)
        water_level = water_level/2 # Convert from amplitude to tidal range
        _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
        elevation, emergence_freq, inundation_freq = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel, dz)

        supratidal_boundary = elevation[closest_to(0.01, inundation_freq)]
        upper_intertidal_boundary = elevation[closest_to(0.99, inundation_freq)]
        lower_intertidal_boundary = elevation[closest_to(0.99, emergence_freq)]
        subtidal_boundary = elevation[closest_to(0.01, emergence_freq)]
        
        total_intertidal = ((emergence_freq > 0.01) & (inundation_freq > 0.01)).sum()*dz

        return total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary

    def plot_figure():

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

        # plt.text(x = 1.00, y = 1.3,  s = 'Supratidal zone',                    c = 'black',  rotation =  10, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        # plt.text(x = 1.00, y = 0.95,  s = 'Intermittent upper intertidal zone', c = 'orange', rotation =  10, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        # plt.text(x = 1.00, y = -0.95, s = 'Intermittent lower intertidal zone', c = 'blue',   rotation = -10, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        # plt.text(x = 1.00, y = -1.3, s = 'Subtidal zone',                      c = 'black',  rotation = -10, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')
        # plt.text(x = 0.00, y = -0.10, s = 'Daily intertidal zone',              c = 'black',  rotation =   9, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center')
        # plt.text(x = 1.50, y = -0.20, s = 'Fluctuating intertidal zone',        c = 'black',  rotation = -10, fontsize = 10, horizontalalignment = 'center', verticalalignment = 'center')

        plt.scatter(transition_point, 0, s = 30, c = 'black', zorder = 2, label = 'Transition point')
        plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        plt.xlabel('ASLC range / Tidal range [-]', fontsize = 12)
        plt.title(rf'$A_{{S2}}/A_{{M2}} = {np.round(amplitude_S2/amplitude_M2,3)}$')
        plt.legend(frameon = False, fontsize = 11)
        plt.ylim(-2.1, 2.1)

        plt.savefig(path_png, dpi = 300)
        plt.show()

    # Load simulation
    if os.path.exists(path_bin):
        amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = np.load(path_bin)
    # Or run simulation
    else:
        n = 100
        amplitude_annual_cycle = np.linspace(0, 1.5, n)
        total_intertidal = np.array([np.nan]*n)
        supratidal_boundary       = np.array([np.nan]*n)
        upper_intertidal_boundary = np.array([np.nan]*n)
        lower_intertidal_boundary = np.array([np.nan]*n)
        subtidal_boundary         = np.array([np.nan]*n)
        for i in range(n):
            total_intertidal[i], supratidal_boundary[i], upper_intertidal_boundary[i], lower_intertidal_boundary[i], subtidal_boundary[i] = identify_intertidal_zones(amplitude_annual_cycle[i],  amplitude_M2, amplitude_S2, n_years)
            print(i)

        np.save(path_bin, (amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary))

    transition_point = amplitude_annual_cycle[closest_to(upper_intertidal_boundary, 0)]

    # Plot figure
    if plot:
        plot_figure()

    return transition_point 

# Figure S1 panels
figure_s1(plot = True, path_bin = './bin/figure_s1/figure_s1-0.7.npy', path_png = './figures/figure_s1/figure_s1-0.7.png', amplitude_M2 = 1, amplitude_S2 = 0.70, n_years = 50)
figure_s1(plot = True, path_bin = './bin/figure_s1/figure_s1-0.4.npy', path_png = './figures/figure_s1/figure_s1-0.4.png', amplitude_M2 = 1, amplitude_S2 = 0.40, n_years = 50)
figure_s1(plot = True, path_bin = './bin/figure_s1/figure_s1-0.15.npy', path_png = './figures/figure_s1/figure_s1-0.15.png', amplitude_M2 = 1, amplitude_S2 = 0.15, n_years = 50)


# Figure S1b - linear regression
n = 11
amplitude_m2 = 1
amplitude_s2_seq = np.linspace(0.15, 0.70, n)
s2m2_ratio = np.round(amplitude_s2_seq/amplitude_m2, 3)
transition_point = np.array([np.nan]*n)
for i in range(amplitude_s2_seq.size):
    transition_point[i] = figure_s1(plot = False, 
                                    path_bin = f'./bin/figure_s1b/figure_s1-{s2m2_ratio[i]}.npy', 
                                    path_png = f'./figures/figure_s1b/figure_s1-{s2m2_ratio[i]}.png', 
                                    amplitude_M2 = 1, amplitude_S2 = amplitude_s2_seq[i], n_years = 20)

def linear_model(x, a):
    return a + -x
par, _ = curve_fit(linear_model, s2m2_ratio, transition_point, p0 = (1))
xp = np.linspace(s2m2_ratio.min(), s2m2_ratio.max(), 100)
yp = linear_model(xp, *par)

plt.plot(xp, yp, c = 'black')
plt.scatter(s2m2_ratio, transition_point, c = 'black', s = 20)
plt.xlabel('$A_{S2}$/$A_{M2}$', fontsize = 12)
plt.ylabel('Transition point (ASLC range / Tidal range) [-]', fontsize = 12)
plt.savefig('./figures/figure_s1/figure_s1b.png', dpi = 300)
plt.show()
print(par)