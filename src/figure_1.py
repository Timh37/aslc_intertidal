import numpy as np
import matplotlib.pyplot as plt
import src.functions as f

def figure_1a(path_png = None):

    time, waterlevel, mean_waterlevel = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0., mean_waterlevel = 0., n_years = 1)
 
    # Change to tidal range from tidal amplitude
    waterlevel = waterlevel/2
    mean_waterlevel = mean_waterlevel/2

    tidal_cycle, low_waterlevel, high_waterlevel = f.calculate_highlow_water(waterlevel, dt = 5/60, plot = False)        
    plt.plot(time/24, mean_waterlevel, zorder = 1, c = 'black', alpha = 0.6)
    plt.plot(365*tidal_cycle/705, low_waterlevel, c = 'blue', alpha = 0.6)
    plt.plot(365*tidal_cycle/705, high_waterlevel, c = 'orange', alpha = 0.6)

    plt.text(x = 0,     y =  0.85, s = 'High water level',  c = 'orange', fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = 0)
    plt.text(x = 0,     y =  0.1, s = 'Mean water level',  c = 'black',  fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = 0)
    plt.text(x = 0,     y = -0.85, s = 'Low water level',   c = 'blue',   fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = 0)
    plt.text(x = 365, y = 1.1, s = 'A',  c = 'black',  fontsize = 20, horizontalalignment = 'right',   verticalalignment = 'top', rotation = 0)

    plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.xlabel('Time [days]', fontsize = 12)
    plt.ylim(-1.2, 1.2)
    if path_png is not None:
        plt.savefig(path_png, dpi = 300)
    plt.show()

def figure_1b(path_png = None):

    _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0. , n_years = 10)
    water_level = water_level/2
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation, emergence_freq, inundation_freq = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel)

    intermediate_intertidal = (emergence_freq > 0.99) & (inundation_freq > 0.99)
    start_inter = elevation[np.where(intermediate_intertidal == 1)[0][0]]
    end_inter   = elevation[np.where(intermediate_intertidal == 1)[0][-1]]
    plt.plot([start_inter, end_inter], [-0.02,-0.02], alpha = 1, c = 'black')

    upper_intertidal = ((inundation_freq < 0.99) & (inundation_freq > 0.01))
    start_upper = elevation[np.where(upper_intertidal == 1)[0][0]]
    end_upper   = elevation[np.where(upper_intertidal == 1)[0][-1]]
    plt.plot([start_upper, end_upper], [1.02,1.02], alpha = 1, c = 'orange')
    plt.fill_between(elevation, upper_intertidal, color = 'orange', alpha = 0.05)

    lower_intertidal = ((emergence_freq < 0.99) & (emergence_freq > 0.01))
    start_lower = elevation[np.where(lower_intertidal == 1)[0][0]]
    end_lower   = elevation[np.where(lower_intertidal == 1)[0][-1]]
    plt.plot([start_lower, end_lower], [1.02,1.02], alpha = 1, c = 'blue')
    plt.fill_between(elevation, lower_intertidal, color = 'blue', alpha = 0.05)

    plt.text(  end_upper + 0.07, 0.95, rotation = 90, s = 'Supratidal zone',           c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
    plt.text(  end_upper - 0.07, 0.95, rotation = 90, s = 'Upper intertidal zone',     c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
    plt.text(start_inter + 0.07, 0.95, rotation = 90, s = 'Stable intermeditate zone', c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
    plt.text(start_lower + 0.07, 0.95, rotation = 90, s = 'Lower intertidal zone',     c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')
    plt.text(start_lower - 0.07, 0.95, rotation = 90, s = 'Subtidal zone',             c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'top')

    plt.text(  end_upper + 0.07, 0.050, rotation = 90, s =  '<1%', c = 'orange', fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')
    plt.text(  end_inter - 0.07, 0.050, rotation = 90, s = '>99%', c = 'orange', fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')
    plt.text(start_inter + 0.07, 0.050, rotation = 90, s = '>99%', c = 'blue',   fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')
    plt.text(start_lower - 0.07, 0.050, rotation = 90, s =  '<1%', c = 'blue',   fontsize = 10, horizontalalignment = 'center',  verticalalignment = 'bottom')

    plt.text( 2.33/2, -0.050, rotation = 0, s = 'Inundation', c = 'orange', fontsize = 11, horizontalalignment = 'right',  verticalalignment = 'center')
    plt.text(-2.33/2, -0.050, rotation = 0, s = 'Emergence',  c = 'blue',   fontsize = 11, horizontalalignment = 'left',   verticalalignment = 'center')
    plt.text(x = 1.125, y = 1.125, s = 'B',  c = 'black',  fontsize = 20, horizontalalignment = 'right',   verticalalignment = 'top', rotation = 0)

    plt.plot(elevation, emergence_freq, c = 'blue', alpha = 1)
    plt.plot(elevation, inundation_freq, c = 'orange', alpha = 1)
    plt.ylabel('Frequency of inundation/emergence [-]', fontsize = 12)
    plt.xlabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.ylim(-0.17,1.17)
    plt.yticks(np.arange(0,1 + 0.2,0.2))
    plt.xlim(-2.4/2, 2.4/2)
    if path_png is not None:
        plt.savefig(path_png, dpi = 300)
    plt.show()

def figure_1c(path_png = None):

    def closest_to(seq, x):
        return np.argmin(np.abs(x - seq))
    
    time, waterlevel, mean_waterlevel_0 = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0., mean_waterlevel = 0., n_years = 1)        
    waterlevel = waterlevel/2 # Change to tidal range from tidal amplitude
    mean_waterlevel_0 = mean_waterlevel_0/2 # Change to tidal range from tidal amplitude
    tidal_cycle, low_waterlevel_0, high_waterlevel_0 = f.calculate_highlow_water(waterlevel, dt = 5/60, plot = False)    
    plt.plot(365*tidal_cycle/705, high_waterlevel_0, c = 'orange', alpha = 0.33)
    plt.plot(time/24, mean_waterlevel_0, zorder = 1, c = 'black', alpha = 0.33)
    plt.plot(365*tidal_cycle/705, low_waterlevel_0, c = 'blue', alpha = 0.33)

    time, waterlevel, mean_waterlevel_1 = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0., mean_waterlevel = 0.3, n_years = 1)
    waterlevel = waterlevel/2 # Change to tidal range from tidal amplitude
    mean_waterlevel_1 = mean_waterlevel_1/2 # Change to tidal range from tidal amplitude
    tidal_cycle, low_waterlevel_1, high_waterlevel_1 = f.calculate_highlow_water(waterlevel, dt = 5/60, plot = False)
    plt.plot(365*tidal_cycle/705, high_waterlevel_1, c = 'orange', alpha = 0.66, label = 'High water level')
    plt.plot(time/24, mean_waterlevel_1, zorder = 1, c = 'black', alpha = 0.66, label = 'Mean water level')
    plt.plot(365*tidal_cycle/705, low_waterlevel_1, c = 'blue', alpha = 0.66, label = 'Low water level')

    plt.annotate('', 
                xy = (5, mean_waterlevel_1[closest_to(time/24, 5)]), 
                xytext = (5, mean_waterlevel_0[closest_to(time/24, 5)]),
                arrowprops=dict(arrowstyle='->', color='black', lw = 1, alpha = 0.5))

    plt.annotate('', 
                xy = (365/2, mean_waterlevel_1[closest_to(time/24, 365/2)]), 
                xytext = (365/2, mean_waterlevel_0[closest_to(time/24, 365/2)]),
                arrowprops=dict(arrowstyle='->', color='black', lw = 1, alpha = 0.5))

    plt.annotate('', 
                xy = (360, mean_waterlevel_1[closest_to(time/24, 360)]), 
                xytext = (360, mean_waterlevel_0[closest_to(time/24, 360)]),
                arrowprops=dict(arrowstyle='->', color='black', lw = 1, alpha = 0.5))

    plt.text(x = 365/2, y = -0.166/2, s = 'No sea level rise', c = 'black',  alpha = 0.3, fontsize = 11, horizontalalignment = 'center', verticalalignment = 'center')
    plt.text(x = 365/2, y =  0.500/2, s = 'Sea level rise',    c = 'black',  alpha = 0.6, fontsize = 11, horizontalalignment = 'center', verticalalignment = 'center')
    plt.text(x = 0,     y =  1.866/2, s = 'High water level',  c = 'orange', alpha = 0.6, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = 0)
    plt.text(x = 0,     y =  0.500/2, s = 'Mean water level',  c = 'black',  alpha = 0.6, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = 0)
    plt.text(x = 0,     y = -1.533/2, s = 'Low water level',   c = 'blue',   alpha = 0.6, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = 0)
    plt.text(x = 365, y = 1.1, s = 'C',  c = 'black',  fontsize = 20, horizontalalignment = 'right',   verticalalignment = 'top', rotation = 0)

    plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.xlabel('Time [days]', fontsize = 12)
    plt.ylim(-2.4/2, 2.4/2)
    if path_png is not None:
        plt.savefig(path_png, dpi = 300)
    plt.show()

def figure_1d(path_png = None):

    def closest_to(seq, x):
        return np.argmin(np.abs(x - seq))

    # Run tidal water level scenarios
    _, water_level, _ = f.create_waterlevel_timeseries(mean_waterlevel = 0., n_years = 10)
    water_level = water_level/2 # Change to tidal range from tidal amplitude
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation_0, emergence_freq_0, inundation_freq_0 = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel)

    _, water_level, _ = f.create_waterlevel_timeseries(mean_waterlevel = 0.25, n_years = 10)
    water_level = water_level/2 # Change to tidal range from tidal amplitude
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation_1, emergence_freq_1, inundation_freq_1 = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel)

    _, water_level, _ = f.create_waterlevel_timeseries(mean_waterlevel = 0.5, n_years = 10)
    water_level = water_level/2 # Change to tidal range from tidal amplitude
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation_2, emergence_freq_2, inundation_freq_2 = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel)

    # Modify output for tidal squeeze
    max_elevation = 1.55/2
    inundation_freq_1_squeeze   = np.where(elevation_1 >  max_elevation, 0,      inundation_freq_1)
    inundation_freq_1_nosqueeze = np.where(elevation_1 <= max_elevation, np.nan, inundation_freq_1)
    inundation_freq_2_squeeze   = np.where(elevation_2 >  max_elevation, 0,      inundation_freq_2)
    inundation_freq_2_nosqueeze = np.where(elevation_2 <= max_elevation, np.nan, inundation_freq_2)

    # Draw annotation bars of different intertidal zones
    def plot_daily_intertidal(elevation, emergence_freq, inundation_freq, alpha, ypos, linestyle = 'solid'):
        daily_intertidal = (emergence_freq > 0.99) & (inundation_freq > 0.99)
        start = elevation[np.where(daily_intertidal == 1)[0][0]]
        end   = elevation[np.where(daily_intertidal == 1)[0][-1]]
        plt.plot([start, end], [ypos,ypos], alpha = alpha, c = 'black', linestyle = linestyle)

    def plot_intermitent_dry_intertidal(elevation, inundation_freq, alpha, ypos, linestyle = 'solid', fill = True):
        intermitent_dry_intertidal = ((inundation_freq < 0.99) & (inundation_freq > 0.01))
        start = elevation[np.where(intermitent_dry_intertidal == 1)[0][0]]
        end   = elevation[np.where(intermitent_dry_intertidal == 1)[0][-1]]
        if fill:
            plt.fill_between(elevation, intermitent_dry_intertidal, color = 'orange', alpha = 0.05)
        plt.plot([start, end], [ypos,ypos], alpha = alpha, c = 'orange', linestyle = linestyle)

    def plot_intermitent_wet_intertidal(elevation, emergence_freq, alpha, ypos, linestyle = 'solid', fill = True):
        intermitent_dry_intertidal = ((emergence_freq < 0.99) & (emergence_freq > 0.01))
        start = elevation[np.where(intermitent_dry_intertidal == 1)[0][0]]
        end   = elevation[np.where(intermitent_dry_intertidal == 1)[0][-1]]
        if fill:
            plt.fill_between(elevation, intermitent_dry_intertidal, color = 'blue', alpha = 0.05)
        plt.plot([start, end], [ypos,ypos], alpha = alpha, c = 'blue', linestyle = linestyle)

    plot_daily_intertidal(elevation_0, emergence_freq_0, inundation_freq_0, alpha = 0.33, ypos = -0.02)
    plot_daily_intertidal(elevation_1, emergence_freq_1, inundation_freq_1, alpha = 0.66, ypos = -0.04)
    plot_daily_intertidal(elevation_2, emergence_freq_2, inundation_freq_2, alpha = 1.00, ypos = -0.06)

    plot_intermitent_wet_intertidal(elevation_0, emergence_freq_0, alpha = 0.33, ypos = 1.06)
    plot_intermitent_wet_intertidal(elevation_1, emergence_freq_1, alpha = 0.66, ypos = 1.04)
    plot_intermitent_wet_intertidal(elevation_2, emergence_freq_2, alpha = 1.00, ypos = 1.02)

    plot_intermitent_dry_intertidal(elevation_0, inundation_freq_0,           alpha = 0.33, ypos = 1.06)
    plot_intermitent_dry_intertidal(elevation_1, inundation_freq_1_squeeze,   alpha = 0.66, ypos = 1.04)
    plot_intermitent_dry_intertidal(elevation_1, inundation_freq_1_nosqueeze, alpha = 0.66, ypos = 1.04, linestyle = 'dotted', fill = False)
    plot_intermitent_dry_intertidal(elevation_2, inundation_freq_2_squeeze,   alpha = 1.00, ypos = 1.02)
    plot_intermitent_dry_intertidal(elevation_2, inundation_freq_2_nosqueeze, alpha = 1.00, ypos = 1.02, linestyle = 'dotted', fill = False)

    # Draw arrows            
    plt.annotate('', 
                    xy = (elevation_2[closest_to(inundation_freq_2_squeeze, 0.85)], 0.85), 
                    xytext = (elevation_0[closest_to(inundation_freq_0, 0.85)], 0.85),
                arrowprops=dict(arrowstyle='->', color='orange', lw = 1))
    plt.annotate('', 
                    xy = (elevation_2[closest_to(inundation_freq_2_squeeze, 0.15)], 0.15), 
                    xytext = (elevation_0[closest_to(inundation_freq_0, 0.15)], 0.15),
                arrowprops=dict(arrowstyle='->', color='orange', lw = 1))
    plt.annotate('', 
                    xy = (elevation_2[closest_to(emergence_freq_2, 0.85)], 0.85), 
                    xytext = (elevation_0[closest_to(emergence_freq_0, 0.85)], 0.85),
                arrowprops=dict(arrowstyle='->', color='blue', lw = 1))
    plt.annotate('', 
                    xy = (elevation_2[closest_to(emergence_freq_2, 0.15)], 0.15), 
                    xytext = (elevation_0[closest_to(emergence_freq_0, 0.15)], 0.15),
                arrowprops=dict(arrowstyle='->', color='blue', lw = 1))

    # Draw emergence/inundation frequency curves
    plt.plot(elevation_0, emergence_freq_0, c = 'blue', alpha = 0.33)
    plt.plot(elevation_0, inundation_freq_0, c = 'orange', alpha = 0.33)
    plt.plot(elevation_1, emergence_freq_1, c = 'blue', alpha = 0.66)
    plt.plot(elevation_1, inundation_freq_1_squeeze, c = 'orange', alpha = 0.66)
    plt.plot(elevation_1, inundation_freq_1_nosqueeze, c = 'orange', alpha = 0.66, linestyle = 'dotted')
    plt.plot(elevation_2, emergence_freq_2, label = 'Frequency of emergence', c = 'blue', alpha = 1)
    plt.plot(elevation_2, inundation_freq_2_squeeze, label = 'Frequency of inundation', c = 'orange', alpha = 1)
    plt.plot(elevation_2, inundation_freq_2_nosqueeze, label = 'Frequency of inundation', c = 'orange', alpha = 1, linestyle = 'dotted')

    # Draw text, axis labels, and define plot limits
    plt.plot([max_elevation, max_elevation], [0,1], c = 'grey', lw = 1.5)
    plt.text(x = max_elevation * 1.15, y = 0.77, s = 'Coastal squeeze', rotation = 90, c = 'grey', horizontalalignment = 'right', verticalalignment = 'center')
    plt.text(0,  1.12, rotation = 0, s = 'Loss of upper intertidal to coastal squeeze', c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'center')
    plt.text( 2.33/2, -0.05, rotation = 0, s = 'Inundation', c = 'orange', fontsize = 11, horizontalalignment = 'right', verticalalignment = 'center')
    plt.text(-2.33/2, -0.05, rotation = 0, s = 'Emergence',  c = 'blue',   fontsize = 11, horizontalalignment = 'left', verticalalignment = 'center')
    plt.text(x = 1.125, y = 1.125, s = 'D',  c = 'black',  fontsize = 20, horizontalalignment = 'right',   verticalalignment = 'top', rotation = 0)
    plt.ylabel('Frequency of inundation/emergence [-]',    fontsize = 12)
    plt.xlabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.ylim(-0.17,1.17)
    plt.yticks(np.arange(0,1 + 0.2,0.2))
    plt.xlim(-2.4/2, 2.4/2)
    if path_png is not None:
        plt.savefig(path_png, dpi = 300)
    plt.show()

def figure_1e(path_png = None):

    def closest_to(seq, x):
        return np.argmin(np.abs(x - seq))
    
    time, waterlevel, mean_waterlevel = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0., mean_waterlevel = 0., n_years = 1)
    # Change to tidal range from tidal amplitude
    waterlevel = waterlevel/2
    mean_waterlevel = mean_waterlevel/2

    tidal_cycle, low_waterlevel, high_waterlevel = f.calculate_highlow_water(waterlevel, dt = 5/60, plot = False)        
    plt.plot(time/24, mean_waterlevel, zorder = 1, c = 'black', alpha = 0.33)
    plt.plot(365*tidal_cycle/705, low_waterlevel, c = 'blue', alpha = 0.33)
    plt.plot(365*tidal_cycle/705, high_waterlevel, c = 'orange', alpha = 0.33)

    time, waterlevel, mean_waterlevel = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0.3, mean_waterlevel = 0., n_years = 1)
    # Change to tidal range from tidal amplitude
    waterlevel = waterlevel/2
    mean_waterlevel = mean_waterlevel/2

    tidal_cycle, low_waterlevel, high_waterlevel = f.calculate_highlow_water(waterlevel, dt = 5/60, plot = False)
    plt.plot(365*tidal_cycle/705, high_waterlevel, c = 'orange', alpha = 0.66, label = 'High water level')
    plt.plot(time/24, mean_waterlevel, zorder = 1, c = 'black', alpha = 0.66, label = 'Mean water level')
    plt.plot(365*tidal_cycle/705, low_waterlevel, c = 'blue', alpha = 0.66, label = 'Low water level')

    plt.annotate('', 
                xy = (365/2, mean_waterlevel[closest_to(time/24, 365/2)]), 
                xytext = (365/2, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw = 1, alpha = 0.5))

    plt.annotate('', 
                xy = (5, mean_waterlevel[closest_to(time/24, 5)]), 
                xytext = (5, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw = 1, alpha = 0.5))

    plt.annotate('', 
                xy = (360, mean_waterlevel[closest_to(time/24, 360)]), 
                xytext = (360, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw = 1, alpha = 0.5))

    plt.text(x = 365/2, y =  0.166/2, s = 'No annual cycle',  c = 'black',  alpha = 0.3, fontsize = 11, horizontalalignment = 'center', verticalalignment = 'center')
    plt.text(x = 365/2, y = -0.533/2, s = 'Annual cycle',     c = 'black',  alpha = 0.6, fontsize = 11, horizontalalignment = 'center', verticalalignment = 'center')
    plt.text(x = 0,     y =  1.833/2, s = 'High water level', c = 'orange', alpha = 0.6, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = -5)
    plt.text(x = 0,     y =  0.466/2, s = 'Mean water level', c = 'black',  alpha = 0.6, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = -5)
    plt.text(x = 0,     y = -1.566/2, s = 'Low water level',  c = 'blue',   alpha = 0.6, fontsize = 10, horizontalalignment = 'left',   verticalalignment = 'center', rotation = -5)

    plt.text(x = 365, y = 1.1, s = 'E',  c = 'black',  fontsize = 20, horizontalalignment = 'right',   verticalalignment = 'top', rotation = 0)
    plt.ylabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.xlabel('Time [days]', fontsize = 12)
    plt.ylim(-2.4/2, 2.4/2)
    if path_png is not None:
        plt.savefig(path_png, dpi = 300)
    plt.show()

def figure_1f(path_png = None):

    def closest_to(seq, x):
        return np.argmin(np.abs(x - seq))

    _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0., n_years = 10)
    water_level = water_level/2
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation_0, emergence_freq_0, inundation_freq_0 = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel)

    _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0.25, n_years = 10)
    water_level = water_level/2
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation_1, emergence_freq_1, inundation_freq_1 = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel)

    _, water_level, _ = f.create_waterlevel_timeseries(amplitude_annual_cycle = 0.5, n_years = 10)
    water_level = water_level/2
    _, low_waterlevel, high_waterlevel = f.calculate_highlow_water(water_level)
    elevation_2, emergence_freq_2, inundation_freq_2 = f.calculate_inundation_metrics(low_waterlevel, high_waterlevel)

    intermediate_intertidal_0 = (emergence_freq_0 > 0.99) & (inundation_freq_0 > 0.99)
    intermediate_intertidal_1 = (emergence_freq_1 > 0.99) & (inundation_freq_1 > 0.99)
    intermediate_intertidal_2 = (emergence_freq_2 > 0.99) & (inundation_freq_2 > 0.99)
    start_0 = elevation_0[np.where(intermediate_intertidal_0 == 1)[0][0]]
    end_0   = elevation_0[np.where(intermediate_intertidal_0 == 1)[0][-1]]
    start_1 = elevation_1[np.where(intermediate_intertidal_1 == 1)[0][0]]
    end_1   = elevation_1[np.where(intermediate_intertidal_1 == 1)[0][-1]]
    start_2 = elevation_2[np.where(intermediate_intertidal_2 == 1)[0][0]]
    end_2   = elevation_2[np.where(intermediate_intertidal_2 == 1)[0][-1]]
    plt.plot([start_0, end_0], [-0.02,-0.02], alpha = 0.33, c = 'black')
    plt.plot([start_1, end_1], [-0.04,-0.04], alpha = 0.66, c = 'black')
    plt.plot([start_2, end_2], [-0.06,-0.06], alpha = 1, c = 'black')

    upper_intertidal_0 = ((inundation_freq_0 < 0.99) & (inundation_freq_0 > 0.01))
    upper_intertidal_1 = ((inundation_freq_1 < 0.99) & (inundation_freq_1 > 0.01))
    upper_intertidal_2 = ((inundation_freq_2 < 0.99) & (inundation_freq_2 > 0.01))
    start_0 = elevation_0[np.where(upper_intertidal_0 == 1)[0][0]]
    end_0   = elevation_0[np.where(upper_intertidal_0 == 1)[0][-1]]
    start_1 = elevation_1[np.where(upper_intertidal_1 == 1)[0][0]]
    end_1   = elevation_1[np.where(upper_intertidal_1 == 1)[0][-1]]
    start_2 = elevation_2[np.where(upper_intertidal_2 == 1)[0][0]]
    end_2   = elevation_2[np.where(upper_intertidal_2 == 1)[0][-1]]
    plt.plot([start_0, end_0], [1.06,1.06], alpha = 0.33, c = 'orange')
    plt.plot([start_1, end_1], [1.04,1.04], alpha = 0.66, c = 'orange')
    plt.plot([start_2, end_2], [1.02,1.02], alpha = 1, c = 'orange')
    plt.fill_between(elevation_0, upper_intertidal_0, color = 'orange', alpha = 0.05)
    plt.fill_between(elevation_1, upper_intertidal_1, color = 'orange', alpha = 0.05)
    plt.fill_between(elevation_2, upper_intertidal_2, color = 'orange', alpha = 0.05)

    lower_intertidal_0 = ((emergence_freq_0 < 0.99) & (emergence_freq_0 > 0.01))
    lower_intertidal_1 = ((emergence_freq_1 < 0.99) & (emergence_freq_1 > 0.01))
    lower_intertidal_2 = ((emergence_freq_2 < 0.99) & (emergence_freq_2 > 0.01))
    start_0 = elevation_0[np.where(lower_intertidal_0 == 1)[0][0]]
    end_0   = elevation_0[np.where(lower_intertidal_0 == 1)[0][-1]]
    start_1 = elevation_1[np.where(lower_intertidal_1 == 1)[0][0]]
    end_1   = elevation_1[np.where(lower_intertidal_1 == 1)[0][-1]]
    start_2 = elevation_2[np.where(lower_intertidal_2 == 1)[0][0]]
    end_2   = elevation_2[np.where(lower_intertidal_2 == 1)[0][-1]]
    plt.plot([start_0, end_0], [1.06,1.06], alpha = 0.33, c = 'blue')
    plt.plot([start_1, end_1], [1.04,1.04], alpha = 0.66, c = 'blue')
    plt.plot([start_2, end_2], [1.02,1.02], alpha = 1, c = 'blue')
    plt.fill_between(elevation_0, lower_intertidal_0, color = 'blue', alpha = 0.05)
    plt.fill_between(elevation_1, lower_intertidal_1, color = 'blue', alpha = 0.05)
    plt.fill_between(elevation_2, lower_intertidal_2, color = 'blue', alpha = 0.05)

    plt.text(    0,   -0.125, rotation = 0, s = 'Shrinking zone of stability', c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'center')
    plt.text(    0,    1.120, rotation = 0, s = 'Growing zones of volitility', c = 'black',  fontsize = 11, horizontalalignment = 'center', verticalalignment = 'center')
    plt.text( 2.33/2, -0.050, rotation = 0, s = 'Inundation',                  c = 'orange', fontsize = 11, horizontalalignment = 'right',  verticalalignment = 'center')
    plt.text(-2.33/2, -0.050, rotation = 0, s = 'Emergence',                   c = 'blue',   fontsize = 11, horizontalalignment = 'left',   verticalalignment = 'center')
    plt.text(x = 1.125, y = 1.125, s = 'F',  c = 'black',  fontsize = 20, horizontalalignment = 'right',   verticalalignment = 'top', rotation = 0)

    # Draw arrows            
    plt.annotate('', 
                    xy = (elevation_2[closest_to(inundation_freq_2, 0.85)], 0.85), 
                    xytext = (elevation_0[closest_to(inundation_freq_0, 0.85)], 0.85),
                arrowprops=dict(arrowstyle='->', color='orange', lw = 1))
    plt.annotate('', 
                    xy = (elevation_2[closest_to(inundation_freq_2, 0.15)], 0.15), 
                    xytext = (elevation_0[closest_to(inundation_freq_0, 0.15)], 0.15),
                arrowprops=dict(arrowstyle='->', color='orange', lw = 1))
    plt.annotate('', 
                    xy = (elevation_2[closest_to(emergence_freq_2, 0.85)], 0.85), 
                    xytext = (elevation_0[closest_to(emergence_freq_0, 0.85)], 0.85),
                arrowprops=dict(arrowstyle='->', color='blue', lw = 1))
    plt.annotate('', 
                    xy = (elevation_2[closest_to(emergence_freq_2, 0.15)], 0.15), 
                    xytext = (elevation_0[closest_to(emergence_freq_0, 0.15)], 0.15),
                arrowprops=dict(arrowstyle='->', color='blue', lw = 1))

    plt.plot(elevation_0, emergence_freq_0, c = 'blue', alpha = 0.33)
    plt.plot(elevation_0, inundation_freq_0, c = 'orange', alpha = 0.33)
    plt.plot(elevation_1, emergence_freq_1, c = 'blue', alpha = 0.66)
    plt.plot(elevation_1, inundation_freq_1, c = 'orange', alpha = 0.66)
    plt.plot(elevation_2, emergence_freq_2, label = 'Frequency of emergence', c = 'blue', alpha = 1)
    plt.plot(elevation_2, inundation_freq_2, label = 'Frequency of inundation', c = 'orange', alpha = 1)
    plt.ylabel('Frequency of inundation/emergence [-]', fontsize = 12)
    plt.xlabel('Deviation from MSL / Tidal range [-]', fontsize = 12)
    plt.ylim(-0.17,1.17)
    plt.yticks(np.arange(0,1 + 0.2,0.2))
    plt.xlim(-2.4/2, 2.4/2)
    if path_png is not None:
        plt.savefig(path_png, dpi = 300)
    plt.show()

figure_1a(path_png = './figures/figure_1/figure_1a-2.png')
figure_1b(path_png = './figures/figure_1/figure_1b-2.png')
figure_1c(path_png = './figures/figure_1/figure_1c-2.png')
figure_1d(path_png = './figures/figure_1/figure_1d-2.png')
figure_1e(path_png = './figures/figure_1/figure_1e-2.png')
figure_1f(path_png = './figures/figure_1/figure_1f-2.png')
