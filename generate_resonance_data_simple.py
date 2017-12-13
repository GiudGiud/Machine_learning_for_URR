# Enable Matplotib to work for headless nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import random
import vectfit
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pickle
import time

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
    [poles, residues, offset, slope] = vectfit.vectfit_auto(f, s, n_poles=N_poles, n_iter=100, show=False, inc_real=False, loss_ratio=1e-2, rcond=-1, track_poles=False)

    return [poles, residues, offset, slope]


###############################################################################
#                             Model parameters                                #
###############################################################################

# Measured resolved resonance parameters
average_spacing = 5     # eV
sigma_spacing   = 0.5   # eV
average_width   = 0.4   # eV
sigma_width     = 0.05  # eV

# We need to add scattering as well, as it interferes with the absorption XS. Let's think about it more

# Run parameters
Np    = 100     # Number of different profiles wanted
extra = 200     # Extra profiles if some cant be vector fitted
E0    = 100000  # Minimum energy
Ng    = 500     # Number of points for vector fitting
Nt    = 10000   # Number of points for testing the vector fitting
#random.seed(47)
print("Number of sample profiles :", Np)

# Fix number of resonances for profile generation
Nr = 10
print("Number of resonances per profile :", Nr)

###############################################################################
#                     Generate resonance parameters                           #
###############################################################################

# Allocate arrays
spacings = np.zeros([Np + extra, Nr])
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
    ab_widths[p, :] = np.random.multivariate_normal(mean_width, Sigma_width)

###############################################################################
#                    Generate cross section profile                           #
###############################################################################

# Plot resonance profile
sigma_ab = np.zeros([Np + extra, Ng])
sigma_fine = np.zeros([Np + extra, Nt])

# Find max values of peaks
print("Maximum peak", np.max(peaks))
dE = np.max(peaks) - E0 + 0.1

# Reconstruct parameters and others
energies = np.logspace(np.log10(E0), np.log10(E0 + dE), Ng)
energies_fine = np.logspace(np.log10(E0), np.log10(E0 + dE), Nt)

r0 = 100
# Loop on samples
for p in range(Np + extra):
    Er = E0
    for i in range(np.count_nonzero(ab_widths[p, :])):

        # Scattering not treated yet
        Er = peaks[p, i]
        sigma_ab[p, :]   += r0*np.sqrt(Er / energies)      * 1./(1. + (2*(energies-Er)/ab_widths[p, i])**2 )
        sigma_fine[p, :] += r0*np.sqrt(Er / energies_fine) * 1./(1. + (2*(energies_fine-Er)/ab_widths[p, i])**2)


print(time.time() - start, "s : resonance profiles generated")
plt.figure(figsize=[7,7])
for p in range(1):
    plt.loglog(energies, sigma_ab[p, :], label="Generated absorption XS")
plt.xlabel("Energy (eV)")
plt.ylabel("Cross section (b)")
plt.legend()
plt.savefig("./visu_simple/cross_section_profile")
#plt.show()

###############################################################################
#                          Use vector fitting                                 #
###############################################################################

# Fit using vector fitting algorithm
num_cores = multiprocessing.cpu_count()
results_ab = []
n_simu_jobs = 40
print("Using", num_cores, "threads")
kk = 0
jj = -1
while kk < Np // n_simu_jobs:
    jj += 1
    print("Index", jj)
    try: 
        low_bound = jj * n_simu_jobs
        high_bound = min((jj+1) * n_simu_jobs, Np)
        
        buffered = Parallel(n_jobs=num_cores, timeout=300, batch_size=1)(delayed(vector_fitting)(i,j,energies) for i,j in zip(sigma_ab[low_bound:high_bound], ab_widths[low_bound:high_bound]))
        print(time.time() - start, "s to run vector fitting for", n_simu_jobs, "absorption xs samples of ", Nr, "resonances")

        print("lengths of array", len(buffered))
        if len(buffered) == n_simu_jobs:
            print("Appending data")
            results_ab.append(buffered)
            kk += 1

    except:
        print("Vector fitting failed at t=", time.time() - start, "s")

    time.sleep(1)


###############################################################################
#                          Examine and save data                              #
###############################################################################

# Re-construct XS using poles and residues
# Save data in numpy array
results_np = np.zeros([Np, 2, 2*np.shape(spacings)[1]], dtype=np.complex_)
offsets = np.zeros(Np)
slopes  = np.zeros(Np)
print(results_np.shape)
error = 0
kk = -1
for j in range(Np // n_simu_jobs):
  for p in range(n_simu_jobs):
    kk += 1
    poles = results_ab[j][p][0]
    residues = results_ab[j][p][1]
    offset = results_ab[j][p][2]
    slope = results_ab[j][p][3]
    fitted = vectfit.model(energies_fine, poles, residues, offset, slope)

    results_np[kk, 0, 0:len(results_ab[j][p][1])] = results_ab[j][p][0]
    results_np[kk, 1, 0:len(results_ab[j][p][1])] = results_ab[j][p][1]
    offsets[kk]          = results_ab[j][p][2]
    slopes[kk]           = results_ab[j][p][3]

    error += np.linalg.norm((fitted - sigma_fine[kk, :]) / fitted) / Np
    #print("Error on profile", kk, " ", np.linalg.norm((fitted - sigma_fine[p, :]) / fitted) / Np)

print("Average L2 norm of fractional error", error)

# Save the results
resonance_data = open("./data/simple_res_data_"+str(Nr)+"r_"+str(Np)+"sa_"+str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width)+"-"+str(sigma_width)+"w", "wb")
pickle.dump([energies, results_np, offsets, slopes, peaks, ab_widths], resonance_data)

# Save as other formats
import scipy.io
scipy.io.savemat("./data/simple_res_data_"+str(Nr)+"r_"+str(Np)+"sa_"+str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width)+"-"+str(sigma_width)+"w.mat",\
mdict={'results_np': results_np, 'offsets': offsets, 'slopes': slopes, 'peaks': peaks, 'absorption widths': ab_widths})

# Plot the two profiles compared
plt.figure()
plt.loglog(energies_fine, sigma_fine[Np-1, :], label="Generated XS")
plt.loglog(energies_fine, fitted, label="Fitted XS")
plt.xlabel("Energy (eV)")
plt.ylabel("Cross section (b)")
plt.legend()
plt.savefig("./visu_simple/fitted_vs_generated")
plt.show()

