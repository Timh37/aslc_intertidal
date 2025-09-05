import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def predict_stochastic_parameters():

    def generate_oun(nt, sigma = 1, rho = 0.1, dt = 1, xmean = 0):
        
        r = np.random.normal(size = nt)
        x = np.array([np.nan]*nt)
        x[0] = xmean
        for t in range(1, nt):
            x[t] = x[t - 1] + sigma*np.sqrt(dt)*r[t] + rho*(xmean - x[t - 1])*dt

        return x

    def estimate_stochastic_parameters(x, dt, xmean = 0):
        '''
        dx = sigma * np.sqrt(dt) * N(0,1) + rho * (xmean - x) * dt
        '''

        def linear_fit(x, a, b):
            return a + x*b

        dx = np.diff(x)
        residuals = dx - xmean

        # Predict rho
        par, _ = curve_fit(linear_fit, xdata = (xmean - x[:-1]), ydata = dx, p0=(0, 1))
        rho_est = par[1] / dt

        # Predict sigma
        sigma_est = np.std(residuals) / np.sqrt(dt)

        return rho_est, sigma_est

    n       = 100
    sigma   = np.random.random(size = n)
    rho     = 10**np.linspace(-3,-1, n)
    dt      = 10**np.linspace(-0.3,0.0, n)
    nt      = 100000

    rho_est   = np.array([np.nan]*n)
    sigma_est = np.array([np.nan]*n)
    w_std     = np.array([np.nan]*n)
    for i in range(n):

        w = generate_oun(nt, sigma = sigma[i], rho = rho[i], dt = dt[i])
        w_std[i] = w.std()
        rho_est[i], sigma_est[i] = estimate_stochastic_parameters(w, dt = dt[i])
        print(f'{i+1}/{n}')

    return rho, sigma, w_std, rho_est, sigma_est

def plot_figures(rho, sigma, w_std, rho_est, sigma_est):

    def scale_fit(x, a):
        return x*a

    par, _ = curve_fit(scale_fit, rho, rho_est, p0 = (1))
    print(par)

    xp = np.linspace(rho.min(), rho.max(), 100)
    yp = scale_fit(xp, *par)
    plt.plot(xp, yp, c = 'black')
    plt.scatter(rho, rho_est, c = 'black', s = 10, alpha = 0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rho', fontsize = 12)
    plt.ylabel('Rho prediction', fontsize = 12)
    plt.show()

    par, _ = curve_fit(scale_fit, sigma, sigma_est, p0 = (1))
    print(par)
    xp = np.linspace(sigma.min(), sigma.max(), 100)
    yp = scale_fit(xp, *par)
    plt.plot(xp, yp, c = 'black')
    plt.scatter(sigma, sigma_est, c = 'black', s = 10, alpha = 0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sigma', fontsize = 12)
    plt.ylabel('Sigma prediction', fontsize = 12)
    plt.show()

    y = w_std
    x = sigma/np.sqrt(2*rho)
    par, _ = curve_fit(scale_fit, x, y, p0 = (1))
    print(par)
    xp = np.linspace(x.min(), x.max(), 100)
    yp = scale_fit(xp, *par)
    plt.plot(xp, yp, c = 'black')
    plt.scatter(x, y, c = 'black', s = 10, alpha = 0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Standard deviation of stochastic tidal consitutent (m)', fontsize = 12)
    plt.ylabel('sigma/sqrt(2*rho)', fontsize = 12)
    plt.show()

rho, sigma, w_std, rho_est, sigma_est = predict_stochastic_parameters()
plot_figures(rho, sigma, w_std, rho_est, sigma_est)