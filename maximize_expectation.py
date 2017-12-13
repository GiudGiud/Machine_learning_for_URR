# Enable Matplotib to work for headless nodes
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()

from sklearn import mixture

import numpy as np
import random
import pickle
import time

start = time.time()

###############################################################################
#                  Model parameters and loading the data                      #
###############################################################################

# Measured resolved resonance parameters
average_spacing = 5     # eV
sigma_spacing   = 0.5   # eV
average_width   = 0.4   # eV
sigma_width     = 0.05  # eV
E0              = 1e5   # eV

# Run parameters
Np     = 100    # Number of different training profiles
Nr     = 10     # Number of resonances
Np_gen = 10000  # Number of profiles to generate

print("\nNumber of sample profiles :", Np)
print("Average number of resonances per profile :", Nr)

# Load the data
resonance_data = open("./data/simple_res_data_"+str(Nr)+"r_"+str(Np)+"sa_"+\
     str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width)+"-"+\
     str(sigma_width)+"w", "rb")
[grid, results, offsets, slopes, peaks, widths] = pickle.load(resonance_data)

###############################################################################
#                           Feature selection                                 #
###############################################################################

# Select features
kk = 0
delta_poles = np.zeros(Np*(Nr-1), dtype = np.complex_)
imag_resid  = np.zeros(Np*(Nr-1), dtype = np.complex_)

for i in range(Np):
    for j in range(Nr-1):
        delta_poles[kk] = np.real(results[i, 0, 2*(j + 1)] - results[i, 0, 2*j])
        imag_resid[kk]  = np.imag(results[i, 1, 2*j])  #loosing one value
        kk += 1

NZ_max = kk - 1

# If issue in poles generation
delta_poles = delta_poles[delta_poles>0]
imag_resid = imag_resid[imag_resid>0]

###############################################################################
#                           Run EM algorithm                                  #
###############################################################################

# To study the effect of the number of resonances
Nr_vec = [NZ_max // 10000, NZ_max // 1000, NZ_max // 100, NZ_max// 10, NZ_max]
Nr_vec = [NZ_max]
res_dp = np.zeros([len(Nr_vec),2])
res_res = np.zeros([len(Nr_vec),2])

n_clusters = 1

ii = -1
for NR in Nr_vec:
    ii+=1

    # Fit the distribution of delta_poles
    xs = delta_poles[0:NR*(Nr-1)].reshape(-1, 1)
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full').fit(xs)

    print("\n\nFit of real(delta_poles)")
    print("Mean", gmm.means_)
    print("Std dev.", np.sqrt(gmm.covariances_))
    print("log likelihood", gmm.lower_bound_)
    res_dp[ii, :] = [gmm.means_, np.sqrt(gmm.covariances_)]

    # Fit the distribution of imaginary parts of residuals
    xs = imag_resid[0:NR*(Nr-1)].reshape(-1, 1)
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full').fit(xs)

    print("\n\nFit of imag(residue)")
    print("Mean", gmm.means_)
    print("Std dev.", np.sqrt(gmm.covariances_))
    print("log likelihood", gmm.lower_bound_)
    res_res[ii, :] = [gmm.means_, np.sqrt(gmm.covariances_)]

    # Future work?
    # Try to fit the two together

###############################################################################
#                   Plot CV of generation parameters                          #
###############################################################################

plt.figure()
plt.plot(np.log(Nr_vec) / np.log(10), res_dp[:, 0])
plt.plot(np.log(Nr_vec) / np.log(10), 5 * np.ones(len(Nr_vec)))
plt.xlabel("log(Number of resonances)")
plt.ylabel("Mean dp")
plt.savefig('./results_simple/cv_mean_dp')

plt.figure()
plt.plot(np.log(Nr_vec) / np.log(10), res_dp[:, 1])
plt.plot(np.log(Nr_vec) / np.log(10), 2 * np.ones(len(Nr_vec)))
plt.xlabel("log(Number of resonances)")
plt.ylabel("std dev dp")
plt.savefig('./results_simple/cv_sig_dp')

plt.figure()
plt.plot(np.log(Nr_vec) / np.log(10), res_res[:, 0])
#plt.plot(np.log(Nr_vec) / np.log(10), 0.4 * np.ones(len(Nr_vec)))
plt.xlabel("log(Number of resonances)")
plt.ylabel("Mean Im(p)")
plt.savefig('./results_simple/cv_mean_gam')

plt.figure()
plt.plot(np.log(Nr_vec) / np.log(10), res_res[:, 1])
plt.plot(np.log(Nr_vec) / np.log(10), 0.04 * np.ones(len(Nr_vec)))
plt.xlabel("log(Number of resonances)")
plt.ylabel("Std dev Im(p)")
plt.savefig('./results_simple/cv_sig_gam')

###############################################################################
#                           Generate samples                                  #
###############################################################################

# Allocate arrays for poles and residues
residues = np.zeros([Np_gen, 2*Nr], dtype = np.complex_)
poles    = np.zeros([Np_gen, 2*Nr], dtype = np.complex_)
d_poles  = np.zeros([Np_gen, Nr])

# Im(res) and Im(pol) are proportional
a = - 0.02

# Loop on generated profiles
for p in range(Np_gen):

    # Starting energy
    pole = E0 + average_spacing / 2

    # Draw delta_poles
    d_poles[p, :] = np.random.multivariate_normal(np.ones(Nr) * res_dp[-1, 0], np.eye(Nr) * res_dp[-1, 1]**2)

    # Draw imaginary parts of residues
    residues[p, 0:2*Nr:2] = 1j * np.random.multivariate_normal(np.ones(Nr) * res_res[-1, 0], np.eye(Nr) * res_res[-1, 1]**2)

    # Residues are conjugates
    residues[p, 1:2*Nr:2] = - residues[p, 0:2*Nr:2]

    # Reconstruct poles using d_poles and residues
    for i in range(Nr):
        poles[p, 2 * i]     = pole + a * residues[p, 2 * i]
        poles[p, 2 * i + 1] = pole - a * residues[p, 2 * i]
        pole += d_poles[p, i]

save_file = open("./em_gen_data/simple_generated_"+str(Nr)+"r_"+\
     str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width)+"-"+\
     str(sigma_width)+"w", "wb")

pickle.dump([poles, residues], save_file)




