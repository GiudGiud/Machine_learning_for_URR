# Enable Matplotib to work for headless nodes
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
from mpl_toolkits.mplot3d import Axes3D


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
sigma_spacing   = 2   # eV
average_width   = 0.4   # eV
sigma_width     = 0.04  # eV

# For later
#average_width_el   = 0.4   # eV
#sigma_width_el     = 0.05  # eV
#average_width_ab   = 0.1   # eV
#sigma_width_ab     = 0.002  # eV

# Run parameters
Np    = 1000   # Number of different profiles in file
Nplot = 1000   # Number of profiles to plot
Nr    = 20     # Number of resonances
print("\nNumber of sample profiles :", Np)
print("Average number of resonances per profile :", Nr)

# Load the data
resonance_data = open("./data/complex_res_data_"+str(int(Nr))+"r_"+str(Np)+\
     "sa_"+str(average_spacing)+"sp_"+str(average_width)+"w", "rb")
#resonance_data = open("./data/complex_res_data_"+str(Nr)+"r_"+str(Np)+"sa_"+\
#     str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width_el)+\
#     "-"+str(sigma_width_el)+str(average_width_ab)+"-"+str(sigma_width_ab)+"w", "rb")
[energies, results_np, offsets, slopes, peaks, el_widths, ab_widths] = pickle.load(resonance_data)
print("Dataset size", results_np.shape)

# Simplify data labelling
poles_el = results_np[:, 0, 0, :]
poles_ab = results_np[:, 1, 0, :]
res_el = results_np[:, 0, 1, :]
res_ab = results_np[:, 1, 1, :]

###############################################################################
#                                 Plot data                                   #
###############################################################################

################
# Poles vs poles
plt.figure()
for i in range(Np):
    nz = np.count_nonzero(poles_el[i])
    plt.scatter(np.real(poles_el[i, 0:nz:2]), np.real(poles_ab[i, 0:nz:2]))
plt.xlabel('Elastic xs poles')
plt.ylabel('Absorption xs poles')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/real_poles_el_vs_poles_ab')

plt.figure()
for i in range(Np):
    nz = np.count_nonzero(poles_el[i])
    plt.scatter(np.imag(poles_el[i, 0:nz:2]), np.imag(poles_ab[i, 0:nz:2]))
plt.xlabel('Elastic xs poles')
plt.ylabel('Absorption xs poles')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/imag_poles_el_vs_poles_ab')

######################
# Residues vs residues
plt.figure()
for i in range(Np):
    nz = np.count_nonzero(res_el[i])
    plt.scatter(np.real(res_el[i, 0:nz:2]), np.real(res_ab[i, 0:nz:2]))
plt.xlabel('Elastic xs real(residues)')
plt.ylabel('Absorption xs real(residues)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/real_res_el_vs_res_ab')

plt.figure()
for i in range(Np):
    nz = np.count_nonzero(res_el[i])
    plt.scatter(np.imag(res_el[i, 0:nz:2]), np.imag(res_ab[i, 0:nz:2]))
plt.xlabel('Elastic xs imag(residues)')
plt.ylabel('Absorption xs imag(residues)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/imag_res_el_vs_res_ab')

plt.figure()
for i in range(Np):
    nz = np.count_nonzero(res_el[i])
    plt.scatter(np.imag(res_el[i, 0:nz:2]), np.real(res_ab[i, 0:nz:2]))
plt.xlabel('Elastic xs imag(residues)')
plt.ylabel('Absorption xs real(residues)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/imag_res_el_vs_real_res_ab')

plt.figure()
for i in range(Np):
    nz = np.count_nonzero(res_ab[i])
    plt.scatter(np.imag(res_ab[i, 0:nz:2]), np.real(res_el[i, 0:nz:2]))
plt.xlabel('Absorption xs imag(residues)')
plt.ylabel('Elastic xs real(residues)')
plt.draw()
plt.savefig('./visu_complex/imag_res_ab_vs_real_res_el')

########################################
# Plot real and imaginary parts of poles
plt.figure()
for i in range(Np):
    nz = np.count_nonzero(poles_el[i])
    plt.scatter(np.real(poles_el[i, 0:nz:2]), np.imag(poles_el[i, 0:nz:2]))
plt.title("Top half of elastic poles")
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.tight_layout()
plt.savefig('./visu_complex/poles_el')

plt.figure()
for i in range(Np):
    nz = np.count_nonzero(poles_ab[i])
    plt.scatter(np.real(poles_ab[i, 0:nz:2]), np.imag(poles_ab[i, 0:nz:2]))
plt.title("Top half of absorption poles")
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.tight_layout()
plt.savefig('./visu_complex/poles_ab')

plt.figure()
plt.hist(np.real(poles_el[0:Nplot]).flatten(), bins=30)
plt.xlabel("Real(poles)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_complex/hist_real_poles_el')

plt.figure()
plt.hist(np.imag(poles_el[0:Nplot, 0:nz:2]).flatten(), bins=30)
plt.xlabel("Imag(poles)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_complex/hist_imag_poles_el')

###########################################
# Plot real and imaginary parts of residues
plt.figure()
for i in range(Np):
    nz = np.count_nonzero(res_el[i])
    plt.scatter(np.real(res_el[i, 0:nz:2]), np.imag(res_el[i, 0:nz:2]))
plt.title("Top half of elastic residues")
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.tight_layout()
plt.savefig('./visu_complex/res_el')

plt.figure()
for i in range(Np):
    nz = np.count_nonzero(res_ab[i])
    plt.scatter(np.real(res_ab[i, 0:nz:2]), np.imag(res_ab[i, 0:nz:2]))
plt.title("Top half of absorption residues")
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.tight_layout()
plt.savefig('./visu_complex/res_ab')

plt.figure()
plt.hist(np.real(res_el[0:Nplot]).flatten(), bins=30)
plt.xlabel("Real(residues_elastic)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_complex/hist_real_res_el')

plt.figure()
plt.hist(np.imag(res_el[0:Nplot]).flatten(), bins=30)
plt.xlabel("Imag(residues)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_complex/hist_imag_res_el')

plt.figure()
plt.hist(np.real(res_ab[0:Nplot]).flatten(), bins=30)
plt.xlabel("Real(residues_absorption)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_complex/hist_real_res_ab')

plt.figure()
plt.hist(np.imag(res_ab[0:Nplot, 0:nz:2]).flatten(), bins=30)
plt.xlabel("Imag(residues)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_complex/hist_imag_res_ab')

##############################################################
# Look if real poles has a dependence on generation parameters
plt.figure()
for i in range(Np):
    nz = np.count_nonzero(results_np[i][0])
    if np.count_nonzero(results_np[i][0]) == 2 * np.count_nonzero(peaks[i,:]): 
        plt.scatter(np.real(poles_ab[i, 0:nz:2]), peaks[i, 0:nz//2])
        plt.scatter(np.real(poles_ab[i, 0:nz:2]), peaks[i, 0:nz//2])
plt.xlabel('poles')
plt.ylabel('peaks')
plt.draw()
plt.tight_layout()
plt.savefig('./visu_complex/poles_vs_peaks')

plt.figure()
for i in range(Np):
    nz = np.count_nonzero(results_np[i][0])
    if np.count_nonzero(results_np[i][0]) == 2 * np.count_nonzero(el_widths[i,:]): 
        plt.scatter(np.real(poles_ab[i, 0:nz:2]), el_widths[i, 0:nz//2])
        plt.scatter(np.real(poles_ab[i, 0:nz:2]), el_widths[i, 0:nz//2])
plt.xlabel('poles')
plt.ylabel('widths')
plt.draw()
plt.tight_layout()
plt.savefig('./visu_complex/poles_vs_widths')


###############################################################################
#                        Plot possible features                               #
###############################################################################

##################
# Plot delta poles
kk = 0
dPOLES = np.zeros((Nr-1)*Np, dtype = np.complex_)
for i in range(Np):
    nz = np.count_nonzero(poles_ab[i])
    for j in range((nz-1)//2):
        dPOLES[kk] = np.real(poles_ab[i, 2*j+2] - poles_ab[i, 2*j])
        kk += 1
NZ = kk - 1

plt.figure()
plt.hist(np.real(dPOLES[0:NZ]), bins=100)
plt.xlabel("Delta real(poles)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_complex/real_Dpoles')

####################################################################
# Plot imaginary part of poles ab/el as a function of residues ab/el

############################################
# Influence of imag(res_ab) and real(res_el)
fig = plt.figure()
for i in range(Np):
    for k in np.linspace(0,2*Nr-2,Nr):
        k = int(np.floor(k))
        if np.real(res_el[i, k]) < 2.5:
            c = 'y'
        elif np.real(res_el[i, k]) > 2.5 and np.real(res_el[i, k]) < 2.65:
            c = 'r'
        elif np.real(res_el[i, k]) > 2.65:
            c = 'k'        
        plt.scatter(np.imag(res_ab[i, k]), np.imag(poles_ab[i, k]), color=c)
    #print("Coeff for this profile", np.sum(np.imag(res_ab[i, 0:nz:2])) / np.sum(np.imag(poles_ab[i, 0:nz:2])))
print("Coeff for all", np.sum(np.imag(res_ab[:, 0:nz:2])) / np.sum(np.imag(poles_ab[:, 0:nz:2])))
plt.xlabel('imag(res_ab)')
plt.ylabel('imag(poles_ab)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/FF_imag_poles_ab_vs_imag_res_ab_colorby_real_res_el')
plt.show()

fig = plt.figure()
for i in range(Np):
    for k in np.linspace(0,2*Nr-2,Nr):
        k = int(np.floor(k))
        if np.imag(res_ab[i, k]) < 1.22:
            c = 'y'
        elif np.imag(res_ab[i, k]) > 1.22 and np.imag(res_ab[i, k]) < 1.35:
            c = 'r'
        elif np.imag(res_ab[i, k]) > 1.35:
            c = 'k'

        plt.scatter(np.real(res_el[i, k]), np.imag(poles_ab[i, k]), color=c)
    #print("Coeff for this profile", np.sum(np.real(res_el[i, 0:nz:2])) / np.sum(np.imag(poles_ab[i, 0:nz:2])))
print("Coeff for all", np.sum(np.real(res_el[:, 0:nz:2])) / np.sum(np.imag(poles_ab[:, 0:nz:2])))
plt.xlabel('real(res_el)')
plt.ylabel('imag(poles_ab)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/FF_imag_poles_ab_vs_real_res_el')
plt.show()

############################################
# Influence of imag(res_ab) and imag(res_el)
fig = plt.figure()
for i in range(Np):
    for k in np.linspace(0,2*Nr-2,Nr):
        k = int(np.floor(k))
        if np.imag(res_el[i, k]) < -1:
            c = 'y'
        elif np.imag(res_el[i, k]) > -1 and np.imag(res_el[i, k]) < -0.5:
            c = 'r'
        elif np.imag(res_el[i, k]) > -0.5:
            c = 'k'        
        plt.scatter(np.imag(res_ab[i, k]), np.imag(poles_ab[i, k]), color=c)
    #print("Coeff for this profile", np.sum(np.imag(res_ab[i, 0:nz:2])) / np.sum(np.imag(poles_ab[i, 0:nz:2])))
print("Coeff for all", np.sum(np.imag(res_ab[:, 0:nz:2])) / np.sum(np.imag(poles_ab[:, 0:nz:2])))
plt.xlabel('imag(res_ab)')
plt.ylabel('imag(poles_ab)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/FF_imag_poles_ab_vs_imag_res_ab_colorby_imag_res_el')
plt.show()

fig = plt.figure()
for i in range(Np):
    for k in np.linspace(0,2*Nr-2,Nr):
        k = int(np.floor(k))
        if np.imag(res_ab[i, k]) < 1.22:
            c = 'y'
        elif np.imag(res_ab[i, k]) > 1.22 and np.imag(res_ab[i, k]) < 1.35:
            c = 'r'
        elif np.imag(res_ab[i, k]) > 1.35:
            c = 'k'

        plt.scatter(np.imag(res_el[i, k]), np.imag(poles_ab[i, k]), color=c)
    #print("Coeff for this profile", np.sum(np.real(res_el[i, 0:nz:2])) / np.sum(np.imag(poles_ab[i, 0:nz:2])))
print("Coeff for all", np.sum(np.real(res_el[:, 0:nz:2])) / np.sum(np.imag(poles_ab[:, 0:nz:2])))
plt.xlabel('imag(res_el)')
plt.ylabel('imag(poles_ab)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/FF_imag_poles_ab_vs_imag_res_el')
plt.show()

####################
# 3D scatter plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(Np):
    ax.scatter(np.real(res_ab[i, 0:nz:2]), np.imag(res_ab[i, 0:nz:2]), np.imag(poles_ab[i, 0:nz:2]))
ax.set_xlabel('real(res_ab)')
ax.set_ylabel('imag(res_ab)')
ax.set_zlabel('imag(poles_ab)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/imag_poles_ab_vs_real_res_ab_vs_imag_res_ab')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(Np):
    ax.scatter(np.real(res_ab[i, 0:nz:2]), np.real(res_el[i, 0:nz:2]), np.imag(poles_ab[i, 0:nz:2]))
ax.set_xlabel('real(res_ab)')
ax.set_ylabel('real(res_el)')
ax.set_zlabel('imag(poles_el)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/imag_poles_ab_vs_real_res_ab_vs_real_res_el')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(Np):
    ax.scatter(np.imag(res_ab[i, 0:nz:2]), np.real(res_el[i, 0:nz:2]), np.imag(poles_ab[i, 0:nz:2]))
ax.set_xlabel('imag(res_ab)')
ax.set_ylabel('real(res_el)')
ax.set_zlabel('imag(poles_el)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/imag_poles_ab_vs_imag_res_ab_vs_real_res_el')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(Np):
    ax.scatter(np.imag(res_ab[i, 0:nz:2]), np.imag(res_el[i, 0:nz:2]), np.imag(poles_ab[i, 0:nz:2]))
ax.set_xlabel('imag(res_an)')
ax.set_ylabel('imag(res_el)')
ax.set_zlabel('imag(poles_el)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_complex/imag_poles_ab_vs_imag_res_ab_vs_imag_res_el')
plt.show()


