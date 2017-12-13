# Enable Matplotib to work for headless nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import openmc
from openmc.data.endf import *
from openmc.data.function import Tabulated1D, Sum, ResonancesWithBackground

from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import random
import vectfit
import time
import pickle

start = time.time()

# Plot parameters
font = {'size'   : 16}
matplotlib.rc('font', **font)

###############################################################################
#                           Helper functions                                  #
###############################################################################

# Function that does the multi-pole decomposition
def vector_fitting(sigma, widths, grid):

    # 2 conjugate poles per resonance, since the data is Real
    N_poles =  np.count_nonzero(widths)

    # f(s) = multipole decompo
    f = sigma
    s = grid

    # perform decomposition
    [poles, residues, offset, slope] = vectfit.vectfit_auto(f, s, n_poles=N_poles, \
         n_iter=100, show=False, inc_real=False, loss_ratio=1e-2, rcond=-1, track_poles=False)

    return [poles, residues, offset, slope]

###############################################################################
#                             Model parameters                                #
###############################################################################

# Measured resolved resonance parameters
average_spacing = 5     # eV
sigma_spacing   = 0.5   # eV
average_width_el   = 0.4   # eV
sigma_width_el     = 0.05  # eV
average_width_ab   = 0.1   # eV
sigma_width_ab     = 0.002  # eV

nuclide = 'Isaac/n-092_U_238.endf'

# Run parameters
Np    = 1000   # Number of different profiles wanted
extra = 1000   # extra profiles if vector fitting fails
E0    = 100000 # Minimum energy
Ng    = 400    # Number of points for vector fitting
Nt    = 1000   # Number of points for testing the vector fitting
random.seed(42)
print("Number of sample profiles :", Np)

Nr = 40
print("Number of resonances per profile :", Nr)

###############################################################################
#                     Generate resonance parameters                           #
###############################################################################

# Allocate arrays
spacings = np.zeros([Np + extra, Nr])
el_widths = np.zeros([Np + extra, Nr])
ab_widths = np.zeros([Np + extra, Nr])
peaks = np.zeros([Np + extra, Nr])

# Compute vector of mean spacings
mean_spacing = np.ones(Nr) * average_spacing
mean_width = np.ones(Nr) * average_width
Sigma_width = np.eye(Nr) * sigma_width ** 2

# Compute correlation between adjacent levels
cov = -0.27 * sigma_spacing**2

# Loop on different profiles
for p in range(Np + extra):

    # Starting energy
    # assuming that we start at the last known resonance
    E = E0 + average_spacing / 2

    # Construct level spacing covariance matrix
    Sigma = np.eye(Nr) * sigma_spacing ** 2
    Sigma += np.diag(cov*np.ones(Nr-1) , -1)
    Sigma += np.diag(cov*np.ones(Nr-1) , 1)

    # Draw distribution of level spacings
    spacings[p, :] = np.random.multivariate_normal(mean_spacing, Sigma)

    # Save the energy peak distributions
    for i in range(Nr):
        peaks[p, i] = E
        E += spacings[p, i]
        
    # Generate widths of resonances
    el_widths[p, :] = np.random.multivariate_normal(mean_width_el, Sigma_width_el)
    ab_widths[p, :] = np.random.multivariate_normal(mean_width_ab, Sigma_width_ab)

dE = np.max(peaks) - E0 + 0.1

###############################################################################
#                    Generate cross section profile                           #
###############################################################################

# Reich Moore formalism reconstructs XS based on peaks and widths
sigma_el = np.zeros([Np + extra, Ng])
sigma_ab = np.zeros([Np + extra, Ng])
sigma_fine = np.zeros([Np + extra, Nt])

# Reconstruct parameters and others
energies = np.logspace(np.log10(E0), np.log10(E0 + dE), Ng)
energies_fine = np.logspace(np.log10(E0), np.log10(E0 + dE), Nt)
nuc_endf = openmc.data.IncidentNeutron.from_endf(nuclide)

# Loop on samples
for p in range(Np + extra):
    for i in range(Nr):
        nuc_endf.resonances.ranges[0].parameters['energy'][i] = peaks[p, i]
        nuc_endf.resonances.ranges[0].parameters['neutronWidth'][i] = el_widths[p, i]
        nuc_endf.resonances.ranges[0].parameters['captureWidth'][i] = ab_widths[p, i]
    nuc_endf.resonances.ranges[0]._prepare_resonances()

    # Index gives MT channel
    reconstructed_XS = nuc_endf.resonances.ranges[0].reconstruct(energies)
    sigma_el[p, :] = reconstructed_XS[2]
    sigma_ab[p, :] = reconstructed_XS[102]

    # Keep a finer one
    sigma_fine[p, :] = nuc_endf.resonances.ranges[0].reconstruct(energies_fine)[102]

    if p % 50 == 0 and p>0:
        print(p, "profiles generated")

print(time.time() - start, "s : resonance profiles generated")
plt.figure(figsize=[7,7])
for p in range(1):
    plt.loglog(energies, sigma_el[p, :], label="Generated elastic XS")
    plt.loglog(energies, sigma_ab[p, :], label="Generated absorption XS")
plt.xlabel("Energy (eV)")
plt.ylabel("Cross section (b)")
plt.legend()
plt.savefig("./visu_complex/cross_section_profile")
#plt.show()

###############################################################################
#                          Use vector fitting                                 #
###############################################################################

# Fit using vector fitting algorithm
num_cores = multiprocessing.cpu_count()
results_el = []
results_ab = []
n_simu_jobs = 100
print("Using", num_cores, "processors")
kk = 0
jj = -1
while kk < Np // n_simu_jobs:
    jj += 1
    try:
        low_bound = jj * n_simu_jobs
        high_bound = min((jj+1) * n_simu_jobs, Np)

        buffered1 = Parallel(n_jobs=num_cores, timeout=1000, batch_size=1)(delayed(vector_fitting)(i,j,energies) for i,j in zip(sigma_el[low_bound:high_bound], el_widths[low_bound:high_bound]))
        print(time.time() - start, "s to run vector fitting for", n_simu_jobs, "elastic xs samples of ", Nr, "resonances")

        buffered2 = Parallel(n_jobs=num_cores, timeout=1000, batch_size=1)(delayed(vector_fitting)(i,j,energies) for i,j in zip(sigma_ab[low_bound:high_bound], ab_widths[low_bound:high_bound]))
        print(time.time() - start, "s to run vector fitting for", n_simu_jobs, "absorption xs samples of ", Nr, "resonances")
        
        if len(buffered1) == n_simu_jobs and len(buffered2) == n_simu_jobs:
            print("Appending data")
            results_el.append(buffered1)
            results_ab.append(buffered2)
            kk += 1
    except:
        print("Vector fitting failed at t=", time.time() - start, "s")

    time.sleep(1)


###############################################################################
#                          Examine and save data                              #
###############################################################################

# Re-construct XS using poles and residues
# Save data in numpy array
results_np = np.zeros([Np, 2, 2, 2*Nr], dtype=np.complex_)
offsets = np.zeros([2, Np])
slopes  = np.zeros([2, Np])
print(results_np.shape)
error = 0
kk = -1
for j in range(Np // n_simu_jobs):
  for p in range(n_simu_jobs):
    kk += 1
    print(j, p)
    poles = results_ab[j][p][0]
    residues = results_ab[j][p][1]
    offset = results_ab[j][p][2]
    slope = results_ab[j][p][3]
    fitted = vectfit.model(energies_fine, poles, residues, offset, slope)

    results_np[kk, 0, 0, 0:len(results_el[j][p][1])] = results_el[j][p][0]
    results_np[kk, 0, 1, 0:len(results_el[j][p][1])] = results_el[j][p][1]
    results_np[kk, 1, 0, 0:len(results_ab[j][p][1])] = results_ab[j][p][0]
    results_np[kk, 1, 1, 0:len(results_ab[j][p][1])] = results_ab[j][p][1]
    offsets[0, kk]          = results_el[j][p][2]
    slopes[0, kk]           = results_el[j][p][3]
    offsets[1, kk]          = results_ab[j][p][2]
    slopes[1, kk]           = results_ab[j][p][3]

    error += np.linalg.norm((fitted - sigma_fine[kk, :]) / fitted) / Np

print("Average L2 norm of fractional error", error)

# Save the results
resonance_data = open("./data/complex_res_data_"+str(Nr)+"r_"+str(Np)+"sa_"+str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width_el)+"-"+str(sigma_width_el)+str(average_width_ab)+"-"+str(sigma_width_ab)+"w", "wb")
pickle.dump([energies, results_np, offsets, slopes, peaks, el_widths, ab_widths], resonance_data)

# Save as other formats
import scipy.io
scipy.io.savemat("./data/complex_res_data_"+str(Nr)+"r_"+str(Np)+"sa_"+str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width_el)+"-"+str(sigma_width_el)+str(average_width_ab)+"-"+str(sigma_width_ab)+"w.mat",\
mdict={'results_np': results_np, 'offsets': offsets, 'slopes': slopes, 'peaks': peaks, 'elastic widths': el_widths, 'absorption widths': ab_widths})

# Plot the two profiles compared
plt.loglog(energies_fine, sigma_fine[Np-1, :], label="Generated XS")
plt.loglog(energies_fine, fitted, label="Fitted XS")
plt.xlabel("Energy (eV)")
plt.ylabel("Cross section (b)")
plt.legend()
plt.savefig("./visu_complex/fitted_vs_generated")
plt.show()

# Time cost with 10000 points (can do 100)
# 80 s for 10 samples of 20 res
# 700 s for 100 samples of 20 res
