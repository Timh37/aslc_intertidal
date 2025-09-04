import numpy as np
import matplotlib.pyplot as plt
import functions as f
import os
from scipy.optimize import curve_fit

def run_sim(amplitude_annual_cycle, sigma, amplitude_S2, n_years):
    _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = amplitude_annual_cycle, amplitude_S2 = amplitude_S2, sigma = sigma, n_years = n_years)
    water_level = water_level/2 # Convert from amplitude to tidal range
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation, emergence_freq, inundation_freq = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel, dz = 0.01)

    return elevation, emergence_freq, inundation_freq

def identify_intertidal_widths(elevation, emergence_freq, inundation_freq):

    upper_intertidal = (emergence_freq >= 0.99) & (inundation_freq >= 0.01) & (inundation_freq < 0.99)
    intermediate_intertidal = (emergence_freq >= 0.99) & (inundation_freq >= 0.99)
    lower_intertidal = (emergence_freq >= 0.01) & (emergence_freq < 0.99) & (inundation_freq >= 0.99)
    transitioning_intertidal = (emergence_freq >= 0.01) & (emergence_freq < 0.99) & (inundation_freq >= 0.01) & (inundation_freq < 0.99)

    if (upper_intertidal.sum() > 0) & (lower_intertidal.sum() > 0):
        total_intertidal_width = elevation[upper_intertidal].max() - elevation[lower_intertidal].min()
    else: 
        total_intertidal_width = 0
    if upper_intertidal.sum() > 0:
        upper_intertidal_width = elevation[upper_intertidal].max() - elevation[upper_intertidal].min()
    else:
        upper_intertidal_width = 0
    if lower_intertidal.sum() > 0:
        lower_intertidal_width = elevation[lower_intertidal].max() - elevation[lower_intertidal].min()
    else:
        lower_intertidal_width = 0
    if intermediate_intertidal.sum() > 0:
        intermediate_intertidal_width = elevation[intermediate_intertidal].max() - elevation[intermediate_intertidal].min()
    else:
        intermediate_intertidal_width = 0
    if transitioning_intertidal.sum() > 0:
        transitioning_intertidal_width = elevation[transitioning_intertidal].max() - elevation[transitioning_intertidal].min()
    else:
        transitioning_intertidal_width = 0

    
    return total_intertidal_width, upper_intertidal_width, lower_intertidal_width, intermediate_intertidal_width, transitioning_intertidal_width

def identify_intertidal_zones(elevation, emergence_freq, inundation_freq):

    supratidal_boundary = elevation[(inundation_freq >= 0.01).argmin()]
    upper_intertidal_boundary = elevation[(inundation_freq <= 0.99).argmax()]
    lower_intertidal_boundary = elevation[(emergence_freq <= 0.99).argmin()]
    subtidal_boundary = elevation[(emergence_freq >= 0.01).argmax()]
    
    return supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary

def run_test(amplitude_annual_cycle = 0, sigma = 0.0, amplitude_S2 = 0.5, n_years = 1):

    def plot_figure_test(elevation, emergence_freq, inundation_freq):

        supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = identify_intertidal_zones(elevation, emergence_freq, inundation_freq)

        # intermediate_intertidal = (emergence_freq >= 0.99) & (inundation_freq >= 0.99)
        plt.plot([lower_intertidal_boundary, upper_intertidal_boundary], [-0.02,-0.02], alpha = 1, c = 'black')

        upper_intertidal = ((inundation_freq < 0.99) & (inundation_freq > 0.01))
        plt.plot([supratidal_boundary, upper_intertidal_boundary], [1.035,1.035], alpha = 1, c = 'orange')
        plt.fill_between(elevation, upper_intertidal, color = 'orange', alpha = 0.05)

        lower_intertidal = ((emergence_freq < 0.99) & (emergence_freq > 0.01))
        plt.plot([lower_intertidal_boundary, subtidal_boundary], [1.02,1.02], alpha = 1, c = 'blue')
        plt.fill_between(elevation, lower_intertidal, color = 'blue', alpha = 0.05)

        # plt.text(  end_upper + 0.07, 0.95, rotation = 90, s = 'Supratidal zone',           c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
        # plt.text(  end_upper - 0.07, 0.95, rotation = 90, s = 'Upper intertidal zone',     c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
        # plt.text(start_inter + 0.07, 0.95, rotation = 90, s = 'Stable intermeditate zone', c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
        # plt.text(start_lower + 0.07, 0.95, rotation = 90, s = 'Lower intertidal zone',     c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
        # plt.text(start_lower - 0.07, 0.95, rotation = 90, s = 'Subtidal zone',             c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')

        # plt.text(  end_upper + 0.07, 0.050, rotation = 90, s =  '<1%', c = 'orange', fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')
        # plt.text(  end_inter - 0.07, 0.050, rotation = 90, s = '>99%', c = 'orange', fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')
        # plt.text(start_inter + 0.07, 0.050, rotation = 90, s = '>99%', c = 'blue',   fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')
        # plt.text(start_lower - 0.07, 0.050, rotation = 90, s =  '<1%', c = 'blue',   fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')

        plt.text( 3.33, -0.050, rotation = 0, s = 'Inundation', c = 'orange', fontsize = 11, horizontalalignment = 'right',  verticalalignment = 'center')
        plt.text(-3.33, -0.050, rotation = 0, s = 'Emergence',  c = 'blue',   fontsize = 11, horizontalalignment = 'left',   verticalalignment = 'center')

        plt.plot(elevation, emergence_freq, c = 'blue', alpha = 1)
        plt.plot(elevation, inundation_freq, c = 'orange', alpha = 1)
        plt.ylabel('Frequency of inundation/emergence [-]', fontsize = 12)
        plt.xlabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        plt.ylim(-0.17,1.17)
        plt.yticks(np.arange(0,1 + 0.2,0.2))
        plt.xlim(-3.4, 3.4)
        plt.show()

    # Run simulation
    elevation, emergence_freq, inundation_freq = run_sim(amplitude_annual_cycle, sigma, amplitude_S2, n_years)
    plot_figure_test(elevation, emergence_freq, inundation_freq)

def run_batch_sim(path_bin):

    def plot_figure(path_png, amplitude_annual_cycle, sigma_w, transition_point, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary):

        # Figure S2 panels
        plt.plot(amplitude_annual_cycle, supratidal_boundary, c = 'orange', alpha = 0.5)
        plt.plot(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange')
        plt.plot(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue')
        plt.plot(amplitude_annual_cycle, subtidal_boundary, c = 'blue', alpha = 0.5)
        plt.scatter(amplitude_annual_cycle, supratidal_boundary, c = 'orange', s = 1, alpha = 0.5)
        plt.scatter(amplitude_annual_cycle, upper_intertidal_boundary, c = 'orange', s = 1, alpha = 1.0)
        plt.scatter(amplitude_annual_cycle, lower_intertidal_boundary, c = 'blue', s = 1, alpha = 1.0)
        plt.scatter(amplitude_annual_cycle, subtidal_boundary, c = 'blue', s = 1, alpha = 0.5)

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

        plt.scatter(transition_point, 0, s = 30, c = 'black', zorder = 2, label = 'Transition point')
        plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
        plt.xlabel('ASLC range / Tidal range [-]', fontsize = 12)
        plt.title(f'$σ_w$ = {np.round(sigma_w,3)}')
        plt.title(f'$σ_w$ = {np.round(sigma_w,3)}')
        plt.legend(frameon = False, fontsize = 11)
        plt.ylim(-2.1, 2.1)

        plt.savefig(path_png, dpi = 300)
        plt.show()

    if False:
    # if os.path.exists(path_bin):
        amplitude_annual_cycle, amplitude_S2, sigma_w, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary = np.load(path_bin)
        nx, ny = amplitude_annual_cycle.shape
    else:
        # Run simulation
        n_years = 50
        nx = 100
        ny = 20
        amplitude_annual_cycle = np.tile(np.linspace(0., 1.5, nx), reps = ny).reshape(ny,nx).T

        # Figure S1
        # amplitude_S2 = np.tile(np.linspace(0., 0.9, ny), reps = nx).reshape(nx,ny)
        # sigma = np.array([0.000]*nx*ny).reshape(nx,ny)  

        # Figure S2
        amplitude_S2 = np.array([0.0]*nx*ny).reshape(nx,ny)
        rho = 0.021
        sigma_w = np.tile(np.linspace(0.00, 0.5, ny), reps = nx).reshape(nx,ny)
        sigma = sigma_w*np.sqrt(2*rho)

        supratidal_boundary       = np.array([np.nan]*nx*ny).reshape(nx,ny)
        upper_intertidal_boundary = np.array([np.nan]*nx*ny).reshape(nx,ny)
        lower_intertidal_boundary = np.array([np.nan]*nx*ny).reshape(nx,ny)
        subtidal_boundary         = np.array([np.nan]*nx*ny).reshape(nx,ny)
        for iy in range(ny):
            for ix in range(nx):
                elevation, emergence_freq, inundation_freq = run_sim(amplitude_annual_cycle = amplitude_annual_cycle[ix,iy], sigma = sigma[ix,iy], amplitude_S2 = amplitude_S2[ix,iy], n_years = n_years)
                supratidal_boundary[ix,iy], upper_intertidal_boundary[ix,iy], lower_intertidal_boundary[ix,iy], subtidal_boundary[ix,iy] = identify_intertidal_zones(elevation, emergence_freq, inundation_freq)
                print(ix)
            print(iy)

        np.save(path_bin, (amplitude_annual_cycle, amplitude_S2, sigma_w, supratidal_boundary, upper_intertidal_boundary, lower_intertidal_boundary, subtidal_boundary))
        # np.save(path_bin, (amplitude_annual_cycle, amplitude_S2, sigma, total_intertidal_width, upper_intertidal_width, lower_intertidal_width, intermediate_intertidal_width, transitioning_intertidal_width))

    amplitude_annual_cycle = amplitude_annual_cycle.mean(axis = 1)
    amplitude_S2 = amplitude_S2.mean(axis = 0)
    sigma_w = sigma_w.mean(axis = 0)
    
    transition_point = amplitude_annual_cycle[np.argmin(np.abs(upper_intertidal_boundary - lower_intertidal_boundary), axis = 0)]

    for iy in range(ny):
        plot_figure(f'./figures/figure_s2/figure_s2_panel-{iy}.png', amplitude_annual_cycle, sigma_w[iy], transition_point[iy], supratidal_boundary[:,iy], upper_intertidal_boundary[:,iy], lower_intertidal_boundary[:,iy], subtidal_boundary[:,iy])

    # Figure S1
    # def linear_model(x, a):
    #     return a + -x
    # par, _ = curve_fit(linear_model, amplitude_S2, transition_point, p0 = (1))
    # xp = np.linspace(amplitude_S2.min(), amplitude_S2.max(), 100)
    # yp = linear_model(xp, *par)
    # plt.plot(xp, yp, c = 'black')
    # plt.scatter(amplitude_S2, transition_point, c = 'black', s = 20)
    # plt.xlabel('$A_{S2}$/$A_{M2}$', fontsize = 12)
    # plt.ylabel('Transition point (ASLC range / Tidal range) [-]', fontsize = 12)
    # plt.savefig('./figures/figure_s1/figure_s1_regression.png', dpi = 300)
    # plt.show()

    # Figure S2
    def linear_model(x, a, b):
        return a*x**b
    par, _ = curve_fit(linear_model, sigma_w, 1 - transition_point, p0 = (2.16, 1.3))
    print(par)
    xp = np.linspace(sigma_w.min(), sigma_w.max(), 100)
    yp = 1 - linear_model(xp, *par)
    plt.plot(xp, yp, c = 'black')
    plt.scatter(sigma_w, transition_point, c = 'black', s = 20)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('$σ_{w}$', fontsize = 12)
    plt.ylabel('Transition point (ASLC range / Tidal range) [-]', fontsize = 12)
    plt.savefig('./figures/figure_s2/figure_s2_regression.png', dpi = 300)
    plt.show()

run_batch_sim(path_bin = './bin/figure_s2/figure_s2_regression.npy')

# run_test()

