# Enable Matplotib to work for headless nodes
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()

# Import libraries
import numpy as np
import random
from sklearn import mixture
from scipy import linalg
import pickle
import time

start = time.time()

###############################################################################
#                  Model parameters and loading the data                      #
###############################################################################

# Measured resolved resonance parameters - needed to retrieve data
average_spacing = 5     # eV
sigma_spacing   = 0.5   # eV
average_width   = 0.4   # eV
sigma_width     = 0.05  # eV
E0              = 1e5   # eV

# Run parameters
Np     = 1000   # Number of different training profiles
Np_gen = 10000 # Number of cross section profiles to generate
Nr     = 20    # Number of resonances
print("\nNumber of sample profiles :", Np)
print("Average number of resonances per profile :", Nr)

# Load the data
resonance_data = open("./data/complex_res_data_"+str(int(Nr))+"r_"+str(Np)+\
     "sa_"+str(average_spacing)+"sp_"+str(average_width)+"w", "rb")
[energies, results_np, offsets, slopes, peaks, el_widths, ab_widths] = pickle.load(resonance_data)


###############################################################################
#                           Feature selection                                 #
###############################################################################

# Simplify data labelling
poles_el = results_np[:, 0, 0, 0:2*Nr:2]
poles_ab = results_np[:, 1, 0, 0:2*Nr:2]
res_el = results_np[:, 0, 1, 0:2*Nr:2]
res_ab = results_np[:, 1, 1, 0:2*Nr:2]
#0:2*Nr:2 ok since we can find the other values with feature relations

# Obtain delta poles
kk = 0
delta_poles = np.zeros(Np*(Nr-1))
for i in range(Np):
    nz = np.count_nonzero(poles_ab[i])
    for j in range(Nr-1):
        delta_poles[kk] = np.real(poles_el[i, j+1] - poles_el[i, j])
        kk += 1
NZ_max = kk - 1

print("Number of resonance spacings available", NZ_max, "(", 19*Np, ")")
#plt.hist(delta_poles)
#plt.show()
##############################################################################
####                               Run EM                                 ####
##############################################################################

# Run EM on features
#NZ_vec = [NZ_max // 10000, NZ_max // 1000, NZ_max // 100, NZ_max// 10, NZ_max]
#NZ_vec = [NZ_max // 1000, NZ_max]
#NZ_vec = 3*np.array([1800, 1500, 1200, 1000, 800, 600, 400, 200, 100])
#NZ_vec = [5400]
NZ_vec = [NZ_max]

#n_comp_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
n_comp_vec = [1]

res_dp = np.zeros([len(NZ_vec), np.max(n_comp_vec), 2])
res_res = np.zeros([len(NZ_vec), np.max(n_comp_vec), 2])

ii = -1
for NZ in NZ_vec:
    ii+=1
    for n_clusters in n_comp_vec:

        # Fit the distribution of delta_poles independently
        xs = delta_poles[0:NZ*(Nr-1)].reshape(-1, 1)
        gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full').fit(xs)

        print("\n\nFit of real(delta_poles)")
        print("Mean", gmm.means_)
        print("Cov", gmm.covariances_)
        #print("log likelihood", gmm.lower_bound_)
        #res_dp[ii, 0:n_clusters+1] = [gmm.means_, np.sqrt(gmm.covariances_)]
        mean_dp = gmm.means_[0]
        cova_dp = gmm.covariances_[0][0]


        # Fit the distribution of all the rest
        print("\n\nFit of all other components")
        # [imag(poles), imag(res_ab), real(res_el), imag(res_el)]
        xs = np.concatenate((np.transpose(np.imag(poles_el.flatten()).reshape(-1, 1)), np.transpose(np.imag(res_ab.flatten()).reshape(-1, 1)), np.transpose(np.real(res_el.flatten()).reshape(-1, 1)), np.transpose(np.imag(res_el.flatten()).reshape(-1, 1))))
        #print(xs.shape)
        xs = np.transpose(xs)
        #print(xs.shape)
        gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full').fit(xs)

        print("Mean", gmm.means_)
        print("Cov.", gmm.covariances_)
        #print("log likelihood", gmm.lower_bound_)
        #res_res[ii, 0:n_clusters+1] = [gmm.means_, np.sqrt(gmm.covariances_)]
        mean_res = gmm.means_[0]
        cova_res = gmm.covariances_[0]


###############################################################################
#                   Plot CV of generation parameters                          #
###############################################################################

if False:
    plt.figure()
    for i in range(NZ):
        plt.plot(np.log(NZ_vec) / np.log(10), res_dp[:, i, 0])
    #plt.plot(np.log(NZ_vec) / np.log(10), 5 * np.ones(len(NZ_vec)))
    plt.xlabel("log(Number of resonances)")
    plt.ylabel("Mean dp")
    plt.savefig('./results_complex/cv_mean_dp')

    plt.figure()
    for i in range(NZ):
        plt.plot(np.log(NZ_vec) / np.log(10), res_dp[:, i, 1])
    plt.plot(np.log(NZ_vec) / np.log(10), 0.5 * np.ones(len(NZ_vec)))
    plt.xlabel("log(Number of resonances)")
    plt.ylabel("std dev dp")
    plt.savefig('./results_complex/cv_sig_dp')

    plt.figure()
    for i in range(NZ):
        plt.plot(np.log(NZ_vec) / np.log(10), res_res[:, i, 0])
    #plt.plot(np.log(NZ_vec) / np.log(10), 0.4 * np.ones(len(NZ_vec)))
    plt.xlabel("log(Number of resonances)")
    plt.ylabel("Mean Im(p)")
    plt.savefig('./results_complex/cv_mean_gam')

    plt.figure()
    for i in range(NZ):
        plt.plot(np.log(NZ_vec) / np.log(10), res_res[:, i, 1])
    #plt.plot(np.log(NZ_vec) / np.log(10), 0.02 * np.ones(len(NZ_vec)))
    plt.xlabel("log(Number of resonances)")
    plt.ylabel("Std dev Im(p)")
    plt.savefig('./results_complex/cv_sig_gam')


###############################################################################
#                           Generate samples                                  #
###############################################################################

# Allocate arrays for poles and residues
residues_el = np.zeros([Np_gen, 2*Nr], dtype = np.complex_)
residues_ab = np.zeros([Np_gen, 2*Nr], dtype = np.complex_)
poles_el    = np.zeros([Np_gen, 2*Nr], dtype = np.complex_)
poles_ab    = np.zeros([Np_gen, 2*Nr], dtype = np.complex_)
poles_im    = np.zeros([Np_gen, Nr], dtype = np.complex_)
d_poles  = np.zeros([Np_gen, Nr])

# Loop on generated profiles
for p in range(Np_gen):

    # Starting energy
    pole = E0 + average_spacing / 2

    # Draw delta_poles for all poles
    #print(mean_dp, cova_dp)
    d_poles[p, :] = np.random.multivariate_normal(np.ones(Nr) * mean_dp, np.eye(Nr) * cova_dp)

    for re in range(Nr):
        # Draw all 4 other variables, for each pole
        generated = np.random.multivariate_normal(mean_res, cova_res)
        # [imag(poles), imag(res_ab), real(res_el), imag(res_el)]

        # Neglecting real part of absorption (don't really have to)
        # Residues are conjugated 2 by 2
        residues_ab[p, 2*re]   = 0 + 1j * generated[1]
        residues_ab[p, 2*re+1] = 0 - 1j * generated[1]

        residues_el[p, 2*re]   = generated[2] + 1j * generated[3]
        residues_el[p, 2*re+1] = generated[2] - 1j * generated[3]

        # Imaginary part of poles, same for absorption and elastic
        poles_im[p, re] = generated[0]


    # Reconstruct poles using d_poles and residues
    for i in range(Nr):
        # Poles are the same for absorption and elastic
        poles_el[p, 2 * i]     = pole + 1j * poles_im[p, i]
        poles_el[p, 2 * i + 1] = pole - 1j * poles_im[p, i]
        poles_ab[p, 2 * i]     = poles_el[p, 2 * i]
        poles_ab[p, 2 * i + 1] = poles_el[p, 2 * i + 1]
        pole += d_poles[p, i]

# Save cross section profiles
save_file = open("./em_gen_data/complex_generated_"+str(Nr)+"r_"+\
     str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width)+"-"+\
     str(sigma_width)+"w", "wb")

pickle.dump([poles_el, poles_ab, residues_el, residues_ab], save_file)
print("Number of cross section profiles generated", Np_gen)



