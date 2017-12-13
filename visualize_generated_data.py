# Enable Matplotib to work for headless nodes
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()

import scipy.io
from scipy.signal import argrelextrema
from scipy.stats import probplot
from sklearn import mixture

import numpy as np
import random
import vectfit
import pickle

###############################################################################
#                  Model parameters and loading the data                      #
###############################################################################

# Measured resolved resonance parameters
average_spacing = 5     # eV
sigma_spacing   = 0.5   # eV
average_width   = 0.4   # eV
sigma_width     = 0.05  # eV
E0 = 100000

# Characteristic of generated data
Ng = 100000  # number of points used to find peaks, needs to be fine or widths appear binned
Nr = 10      # number of resonances
method = "em" # em or dg depending on what generated the data

plot_histograms = False
plot_all_profiles = False

# Load the data
if method == "dg":
    # FILE NAME is assumed to be hidden + hidden_layers
    hidden_layers = 11
    file_name = "deep_gen_data/hidden"+str(hidden_layers)+".mat"
    dict_gen = scipy.io.loadmat(file_name, mdict=None, appendmat=True,)
    results = dict_gen['results_sample']
    Np = results.shape[0]
elif method == "em":
    file_name = open("./em_gen_data/simple_generated_"+str(Nr)+"r_"+\
         str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width)+"-"+\
         str(sigma_width)+"w", "rb")
    [poles_i, residues] = pickle.load(file_name)
    Np = poles_i.shape[0]
    results = np.zeros([Np, 2, 2*Nr], dtype = np.complex_)
    results[:, 0, :] = poles_i
    results[:, 1, :] = residues

dE = np.max(np.real(results[:, 0, :])) - E0 + 0.1
print("\nArray size of generated results", results.shape)

###############################################################################
#                        Visualize generated data
###############################################################################

if plot_histograms:
    plt.figure()
    for i in range(Np):
        plt.scatter(np.real(results[i, 0, np.nonzero(results[i, 0])]), np.real(results[i, 1, np.nonzero(results[i, 1])]))
    plt.xlabel('Real(pole)')
    plt.ylabel('Real(residue)')
    plt.savefig('./visu_generated_simple/real_poles_vs_real_residues')

    # Plot real and imaginary parts of poles
    plt.figure()
    for i in range(Np):
        nz = np.count_nonzero(results[i][0])
        plt.scatter(np.real(results[i, 0, 0:nz:2]), np.imag(results[i, 0, 0:nz:2]))
    plt.title("Top half of poles")
    plt.xlabel('Real(pole)')
    plt.ylabel('Imag(pole)')
    plt.savefig('./visu_generated_simple/poles')


    kk = 0
    POLES = np.zeros(3 * Np * Nr, dtype = np.complex_)
    dPOLES = np.zeros(3 * Np * Nr, dtype = np.complex_)
    RESID = np.zeros(3 * Np * Nr, dtype = np.complex_)
    for i in range(Np):
        for j in range(np.count_nonzero(results[i, 0, :]) - 2):
            POLES[kk] = results[i, 0, j]
            dPOLES[kk] = results[i, 0, j+2] - results[i, 0, j]
            RESID[kk] = results[i, 1, j]
            kk += 1
    NZ = kk - 1

    plt.figure()
    plt.hist(np.real(POLES[0:NZ]), bins=100)
    plt.savefig('./visu_generated_simple/real_poles')

    plt.figure()
    plt.hist(np.real(dPOLES[0:NZ]), bins=100)
    plt.xlabel("Delta real(poles)")
    plt.ylabel("Occurences")
    plt.savefig('./visu_generated_simple/real_Dpoles')

    plt.figure()
    plt.hist(np.imag(POLES[0:NZ]), bins=100)
    plt.xlabel("Imag(poles)")
    plt.ylabel("Occurences")
    plt.savefig('./visu_generated_simple/imag_poles')

    # Plot histograms of real and imaginary parts of residues
    plt.figure()
    plt.hist(np.real(RESID[0:NZ]), bins=100)
    plt.xlabel("Real(residue)")
    plt.ylabel("Occurences")
    plt.savefig('./visu_generated_simple/real_resid')

    plt.figure()
    plt.hist(np.imag(RESID[0:NZ]), bins=100)
    plt.xlabel("Imag(residue)")
    plt.ylabel("Occurences")
    plt.savefig('./visu_generated_simple/imag_resid')

###############################################################################
#                              Reconstruct XS
###############################################################################
plt.figure()

grid = np.linspace(E0, E0 + dE, Ng)
peaks = np.zeros([Np, Nr])
dpeaks = np.zeros([Np, Nr-1])
widths = np.zeros([Np, Nr])
for i in range(Np):

    poles = results[i, 0, :]
    residues = results[i, 1, :]
    offset = 0
    slope = 0

    fitted = vectfit.model(grid, poles, residues, offset, slope)

    # Find extremas
    index_max = argrelextrema(fitted, np.greater)[0]

    # Delete extremas that are too close : they are due to poles not being rigorously the same
    deleted = 0
    for kk, spacing in enumerate(np.ediff1d(grid[index_max])):
        #print("Spacing between poles", spacing)
        if spacing < average_spacing / 100:

            print("Deleting duplicate pole: profile", i, "index", kk, "spacing", spacing)
            fitted[index_max[kk-deleted] : index_max[kk-deleted+1]] = (fitted[index_max[kk-deleted]] + fitted[index_max[kk-deleted+1]])/2
            index_max[kk+1-deleted] = (index_max[kk-deleted] + index_max[kk-deleted+1]) / 2
            index_max = np.delete(index_max, [kk-deleted])
            deleted += 1
            

    # Save peaks and deltas between peaks
    peaks[i, 0:len(index_max)] = grid[index_max]
    dpeaks[i, 0:len(index_max)-1] = np.ediff1d(grid[index_max])

    # Find width of each resonance
    for kk, ind in enumerate(index_max):

        start = max(0, ind - int(Ng / Nr / 1.5))
        end = ind + int(Ng / Nr / 1.5)

        tr_wid = 1. / np.abs(fitted[start:end] - fitted[ind]/2.)
        index_wid_m = argrelextrema(tr_wid, np.greater)[0]

        # Pick middle indexes
        ii = -1
        for ind2 in index_wid_m:
            ii += 1
            if start + ind2 > ind:
                bot = start + index_wid_m[ii-1]
                top = start + ind2
                break

        widths[i, kk] = grid[top]-grid[bot]

        if plot_all_profiles:
            plt.scatter(grid[bot], np.log(np.real(fitted[bot])), marker='+')
            plt.scatter(grid[top], np.log(np.real(fitted[top])), marker='+')

    if plot_all_profiles or i ==6810:
        plt.scatter(grid[index_max], np.log(np.real(fitted[index_max])), marker='o')
        plt.plot(grid, np.log(np.real(fitted)))
        #plt.show()
        #plt.plot(grid, np.log(np.imag(fitted)), label="log(+imag)")
        #plt.plot(grid, np.log(-np.imag(fitted)), label="log(-imag)")
        #plt.legend()
        #plt.show()


print(np.where(dpeaks == 0))
print(np.where(widths == 0))

# Delete the annoying zeros
widths = widths[widths>0]
print(dpeaks.size, np.min(dpeaks))
dpeaks = dpeaks[dpeaks!=0]
print(dpeaks.size, np.min(dpeaks))

###############################################################################
#                 Examine Gaussianity of dPoles and widths 
###############################################################################
# Visual test
plt.xlabel("Energy (eV)")
plt.ylabel("Cross section (b)")
plt.savefig('./visu_generated_simple/first_ten_profiles')

if True:
    plt.figure()
    plt.hist(widths.flatten(), bins=40)
    plt.title("Histogram of the widths")
    plt.savefig('./visu_generated_simple/hist_widths')
    plt.show()

    plt.figure()
    res = probplot(widths.flatten(), plot=plt)
    plt.title("Quantiles of the widths")
    plt.savefig('./visu_generated_simple/pplot_widths')
    plt.show()

    plt.figure()
    plt.hist(dpeaks.flatten(), bins=40)
    plt.title("Histogram of the energy spacings")
    plt.savefig('./visu_generated_simple/hist_spacings')
    plt.show()

    plt.figure()
    res = probplot(dpeaks.flatten(), plot=plt)
    plt.title("Quantiles of the delta peaks")
    plt.savefig('./visu_generated_simple/pplot_spacings')
    plt.show()


# Compute likelihood
def gauss_logL(xbar, mu, sigma, n):
    """Gaussian likelihood"""
    return (- n * np.log(sigma) - 0.5 * n * np.log(2*np.pi) - 0.5 * ((xbar - mu) ** 2) / sigma ** 2)
    
log_like_s = np.sum(gauss_logL(dpeaks.flatten(), average_spacing, sigma_spacing, Np * Nr), 0)
log_like_w = np.sum(gauss_logL(widths.flatten(), average_width, sigma_width, Np * Nr), 0)

print("Log likelihood of distribution of spacings generated according to initial law :", log_like_s)
print("Log likelihood of distribution of widths generated according to initial law :", log_like_w)
# why different signs??

# S-W test
print("Shapiro-Wilk test results for spacings", scipy.stats.shapiro(dpeaks.flatten()))
print("Shapiro-Wilk test results for widths", scipy.stats.shapiro(widths.flatten()))

# Fit a gaussian and see if parameters are the same as the ones for generation
dpeak_fit = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(dpeaks.reshape(-1, 1))
width_fit = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(widths.reshape(-1, 1))
print("\n\nEM gaussian for the d_peaks : parameters", dpeak_fit.means_, np.sqrt(dpeak_fit.covariances_))
print("log likelihood", dpeak_fit.lower_bound_)
print("Initial parameters used", average_spacing, sigma_spacing)

print("EM gaussian for the widths", width_fit.means_, np.sqrt(width_fit.covariances_))
print("log likelihood", width_fit.lower_bound_)
print("Initial parameters used", average_width, sigma_width)



