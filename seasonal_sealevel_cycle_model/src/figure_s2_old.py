import numpy as np
import matplotlib.pyplot as plt
import functions as f

def figure_s2(save = False, load = False, path_bin = None, path_png = None, sigma = 0.025, rho = 0.021, n_years = 10):

    def closest_to(x, seq, axis = None):
        if axis is not None:
            return np.argmin(np.abs(seq[np.newaxis,:] - x[:,np.newaxis]), axis = axis)
        else:
            return np.argmin(np.abs(seq - x))

    def identify_intertidal_zones(amplitude_annual_cycle, sigma, rho, n_years, dz = 0.01):

        _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = amplitude_annual_cycle, sigma = sigma, rho = rho, n_years = n_years)
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

        plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        plt.xlabel('ASLC range / Tidal range [-]', fontsize = 12)
        # plt.title(rf'$Sigma = {sigma}$')
        plt.title(rf'$σ_{{w}} = {np.round(sigma/np.sqrt(2*rho),3)}$')
        
        plt.ylim(-2.1, 2.1)

        if save is True:
            plt.savefig(path_png, dpi = 300)
        # plt.show()
        plt.clf()

    # Run simulation
    if load is False:
        n = 100
        amplitude_annual_cycle = np.linspace(0, 1.5, n)
        total_intertidal = np.array([np.nan]*n)
        supratidal_boundary       = np.array([np.nan]*n)
        upper_intertidal_boundary = np.array([np.nan]*n)
        lower_intertidal_boundary = np.array([np.nan]*n)
        subtidal_boundary         = np.array([np.nan]*n)
        for i in range(n):
            total_intertidal[i], supratidal_boundary[i], upper_intertidal_boundary[i], lower_intertidal_boundary[i], subtidal_boundary[i] = identify_intertidal_zones(amplitude_annual_cycle[i], sigma, rho, n_years)
            print(i)

        if save is True:
            np.save(path_bin, (amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary))

    # Or load simulation
    if load is True:
        amplitude_annual_cycle, total_intertidal, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = np.load(path_bin)
        
    # # Plot figure
    # plot_figure()

    transition_point = amplitude_annual_cycle[closest_to(upper_intertidal_boundary, 0)]
    return transition_point 

# figure_s2(save = True, load = False, path_bin = './bin/output_sigma-0.000.npy', path_png = './figures/suppl_figure_sigma-0.001.png', sigma = 0.001, rho = 0.021, n_years = 50)

n = 20
tidal_amplitude = 1
# sigma   = np.random.random(size = n)
# rho     = 10**np.linspace(-3,-1, n)
rho = 0.021
sigma_seq = np.linspace(0.000, 0.075, n)
sigma_w = np.round(sigma_seq * tidal_amplitude / np.sqrt(2 * rho),3)

transition_point = np.array([np.nan]*n)
for i in range(sigma_seq.size):

    transition_point[i] = figure_s2(save = True, load = True, path_bin = f'./bin/figure_s2/output_sigmaw-{sigma_w[i]}.npy', path_png = f'./figures/figure_s2/suppl_figure_sigmaw-{sigma_w[i]}.png', sigma = sigma_seq[i], rho = rho, n_years = 5)

plt.scatter(sigma_w, transition_point, c = 'black', s = 20)
plt.xlabel('$σ_{w}$', fontsize = 12)
plt.ylabel('Transition point (ASLC range / Tidal range) [-]', fontsize = 12)
plt.savefig('./figures/figure_s2/figure_s2b.png', dpi = 300)
plt.show()
