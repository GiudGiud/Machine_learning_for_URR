import numpy as np
import random
import vectfit
from joblib import Parallel, delayed
import multiprocessing
import time
import pickle
start = time.time()

# Enable Matplotib to work for headless nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

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


# Measured resolved resonance parameters
# We'll do U238 absorption, most important in a reactor
# Measuring hasnt been done yet -> ENDF might give the data actually
average_spacing = 5     # eV
sigma_spacing   = 2   # eV
average_width   = 0.4   # eV
sigma_width     = 0.04  # eV

# We need to add scattering as well, as it interferes with the absorption XS. Let's think about it more

# Run parameters
Np = 10000 # Number of different profiles wanted
dE = 100   # Width of URR window
E0 = 1000  # Minimum energy
Ng = 200   # Number of points for vector fitting
Nt = 1000  # Number of points for testing the vector fitting
random.seed(42)
print("Number of sample profiles :", Np)
print("Average number of resonances per profile :", dE / average_spacing)

spacings = np.zeros([Np, dE//average_spacing * 2])
widths = np.zeros([Np, dE//average_spacing * 2])
peaks = np.zeros([Np, dE//average_spacing * 2])

# Loop on different profiles
for p in range(Np):

    # Starting energy
    E = E0 + average_spacing * random.random()  ## to make more precise later
    r = -1
    while E < E0 + dE - 1. * average_spacing:  ## to make more precise later
        r += 1

        # Sample resonance spacing
        spacing = random.gauss(average_spacing, sigma_spacing)
        spacings[p, r] = spacing
        peaks[p, r] = E

        # Advance energy
        E += spacing

        # Sample resonance widths  (need to change from Gamma to gamma)
        width = random.gauss(average_width, sigma_width)
        widths[p, r] = width

# Plot resonance profile
# Which model shall we assume?
# ENDF has Multi Level BW for scattering, SLBW everywhere. Reichmore is also an option
# SLBW for now

grid = np.linspace(E0, E0 + dE, Ng)
grid_fine = np.linspace(E0, E0 + dE, Nt)
sigma = np.zeros([Np, Ng])
sigma_fine = np.zeros([Np, Nt])

r0 = 100
for p in range(Np):
    Er = E0
    for i in range(np.count_nonzero(widths[p, :])):

        # Scattering not treated yet
        Er = peaks[p, i]
        sigma[p, :] += r0*np.sqrt(Er / grid) * 1./(1. + (2*(grid-Er)/widths[p, i])**2 )
        sigma_fine[p, :] += r0*np.sqrt(Er / grid_fine) * 1./(1. + (2*(grid_fine-Er)/widths[p, i])**2)

plt.figure()
plt.plot(grid, sigma[0, :], label="Generated profile")
plt.xlabel("Energy (eV)")
plt.ylabel("Cross section (b)")
plt.savefig("./images/cross_section_profile")
plt.show()

# Fit using vector fitting algorithm
num_cores = multiprocessing.cpu_count()
print("Using", num_cores, "processors")
results = Parallel(n_jobs=num_cores)(delayed(vector_fitting)(i,j,grid) for i,j in zip(sigma, widths))

print(time.time() - start, "s to run vector fitting for", Np, "samples of ", np.count_nonzero(widths) / Np, "resonances")

# Re-construct XS using poles and residues
# Save data in numpy array
results_np = np.zeros([Np, 2, np.shape(spacings)[1]+20], dtype=np.complex_)
offsets = np.zeros(Np)
slopes  = np.zeros(Np)
print(results_np.shape)
error = 0
for p in range(Np):
    poles = results[p][0]
    residues = results[p][1]
    offset = results[p][2]
    slope = results[p][3]
    fitted = vectfit.model(grid_fine, poles, residues, offset, slope)

    results_np[p, 0, 0:len(results[p][1])] = results[p][0]
    results_np[p, 1, 0:len(results[p][1])] = results[p][1]
    offsets[p]          = results[p][2]
    slopes[p]           = results[p][3]

    error += np.linalg.norm((fitted - sigma_fine[p, :]) / fitted) / Np

print("Average L2 norm of fractional error", error)

# Plot the two profiles compared
#plt.plot(grid, fitted, label="Multipole fit")
#plt.legend()
#plt.show()

# Save the results
resonance_data = open("./data/simplified_res_data_"+str(int(dE/average_spacing))+"r_"+str(Np)+"sa_"+str(average_spacing)+"sp_"+str(average_width)+"w", "wb")
pickle.dump([grid, results_np, offsets, slopes, peaks, widths], resonance_data)

# Save as other formats
import scipy.io
scipy.io.savemat("./data/simplified_res_data_"+str(int(dE/average_spacing))+"r_"+str(Np)+"sa_"+str(average_spacing)+"sp_"+str(average_width)+"w.mat",\
mdict={'results_np': results_np, 'offsets': offsets, 'slopes': slopes, 'peaks': peaks, 'widths': widths})


# Time cost with 10000 points (can do 100)
# 25 s for 20 samples of 20 res
# 50 s for 40 samples of 20 res
# 100 s for 80 samples of 20 res
# 200 s for 40 samples of 40 res
# 800 s for 40 samples of 80 res
