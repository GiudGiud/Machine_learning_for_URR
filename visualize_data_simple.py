# Enable Matplotib to work for headless nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import random
import pickle
import time

start = time.time()

# Plot parameters
font = {'size'   : 16}
matplotlib.rc('font', **font)

###############################################################################
#                  Model parameters and loading the data                      #
###############################################################################

# Measured resolved resonance parameters
average_spacing = 5     # eV
sigma_spacing   = 0.5   # eV
average_width   = 0.4   # eV
sigma_width     = 0.05  # eV

# Run parameters
Np    = 100   # Number of different profiles wanted
Nplot = 100
Nr    = 10
print("Number of sample profiles :", Np)
print("Number of resonances per profile :", Nr)

# Load the data
resonance_data = open("./data/simple_res_data_"+str(Nr)+"r_"+str(Np)+"sa_"+\
     str(average_spacing)+"-"+str(sigma_spacing)+"sp_"+str(average_width)+"-"+\
     str(sigma_width)+"w", "rb")
[grid, results, offsets, slopes, peaks, widths] = pickle.load(resonance_data)

print("Dimension of data set", results.shape)

###############################################################################
#                                 Plot data                                   #
###############################################################################

#########################################
# See if poles and residues are connected
plt.figure()
for i in range(Nplot):
    plt.scatter(np.real(results[i, 0, np.nonzero(results[i, 0])]), np.real(results[i, 1, np.nonzero(results[i, 1])]))
plt.xlabel('real(poles)')
plt.ylabel('real(residues)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/real_poles_vs_real_residues')

plt.figure()
for i in range(Nplot):
    plt.scatter(np.real(results[i, 0, np.nonzero(results[i, 0])]), np.imag(results[i, 1, np.nonzero(results[i, 1])]))
plt.xlabel('real(poles)')
plt.ylabel('imag(residues)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/real_poles_vs_imag_residues')

plt.figure()
for i in range(Nplot):
    plt.scatter(np.imag(results[i, 0, np.nonzero(results[i, 0])]), np.imag(results[i, 1, np.nonzero(results[i, 1])]))
    a = np.sum(np.imag(results[i, 0, np.imag(results[i, 0, :]) > 0])) / np.sum(np.imag(results[i, 1, np.imag(results[i, 1, :]) > 0]))
    print("Coeff for this profile", a)
    if i==4:
        plt.plot(np.linspace(-0.3, 0.3, 100), -1/a * np.linspace(-0.3, 0.3, 100), label="Linear fit")

plt.legend()
plt.xlabel('imag(poles)')
plt.ylabel('imag(residues)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/imag_poles_vs_imag_residues')

plt.figure()
for i in range(Nplot):
    plt.scatter(np.imag(results[i, 0, np.nonzero(results[i, 0])]), np.real(results[i, 1, np.nonzero(results[i, 1])]))
plt.xlabel('imag(poles)')
plt.ylabel('real(residues)')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/imag_poles_vs_real_residues')

##############################################################
# Look if real poles has a dependence on generation parameters
plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][0])
    if np.count_nonzero(results[i][0]) == 2 * np.count_nonzero(peaks[i,:]): 
        plt.scatter(np.real(results[i][0][0:nz:2]), peaks[i, 0:nz//2])
        plt.scatter(np.real(results[i][0][1:nz:2]), peaks[i, 0:nz//2])
plt.xlabel('real(poles)')
plt.ylabel('peaks')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/real_poles_vs_peaks')

plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][0])
    if np.count_nonzero(results[i][0]) == 2 * np.count_nonzero(widths[i,:]): 
        plt.scatter(np.real(results[i][0][0:nz:2]), widths[i, 0:nz//2])
        plt.scatter(np.real(results[i][0][1:nz:2]), widths[i, 0:nz//2])
plt.xlabel('real(poles)')
plt.ylabel('widths')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/real_poles_vs_widths')

# Look if imag poles has a dependence on generation parameters
plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][0])
    if np.count_nonzero(results[i][0]) == 2 * np.count_nonzero(peaks[i,:]): 
        plt.scatter(np.imag(results[i][0][0:nz:2]), peaks[i, 0:nz//2])
        plt.scatter(np.imag(results[i][0][1:nz:2]), peaks[i, 0:nz//2])
plt.xlabel('imag(poles)')
plt.ylabel('peaks')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/imag_poles_vs_peaks')

plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][0])
    if np.count_nonzero(results[i][0]) == 2 * np.count_nonzero(widths[i,:]): 
        plt.scatter(np.imag(results[i][0][0:nz:2]), widths[i, 0:nz//2])
        plt.scatter(np.imag(results[i][0][1:nz:2]), widths[i, 0:nz//2])
plt.xlabel('imag(poles)')
plt.ylabel('widths')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/imag_poles_vs_widths')

####################################################################
# Look if real of residues has a dependence on generation parameters
plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][1])
    if np.count_nonzero(results[i][1]) == 2 * np.count_nonzero(widths[i,:]): 
        plt.scatter(np.real(results[i][1][0:nz:2]), widths[i, 0:nz//2])
        plt.scatter(np.real(results[i][1][1:nz:2]), widths[i, 0:nz//2])
plt.xlabel('real(residues)')
plt.ylabel('widths')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/real_residues_vs_widths')

plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][1])
    if np.count_nonzero(results[i][1]) == 2 * np.count_nonzero(peaks[i,:]): 
        plt.scatter(np.real(results[i][1][0:nz:2]), peaks[i, 0:nz//2])
        plt.scatter(np.real(results[i][1][1:nz:2]), peaks[i, 0:nz//2])
plt.xlabel('real(residues)')
plt.ylabel('peaks')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/real_residues_vs_peaks')

# Look if imag of residues has a dependence on generation parameters
plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][1])
    if np.count_nonzero(results[i][1]) == 2 * np.count_nonzero(widths[i,:]): 
        plt.scatter(np.imag(results[i][1][0:nz:2]), widths[i, 0:nz//2])
        plt.scatter(np.imag(results[i][1][1:nz:2]), widths[i, 0:nz//2])
plt.xlabel('imag(residues)')
plt.ylabel('widths')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/imag_residues_vs_widths')

plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][1])
    if np.count_nonzero(results[i][1]) == 2 * np.count_nonzero(peaks[i,:]): 
        plt.scatter(np.imag(results[i][1][0:nz:2]), peaks[i, 0:nz//2])
        plt.scatter(np.imag(results[i][1][1:nz:2]), peaks[i, 0:nz//2])
plt.xlabel('imag(residues)')
plt.ylabel('peaks')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/imag_residues_vs_peaks')

###############################################
# Plot generated XS parameters before multipole
plt.figure()
for i in range(Nplot):
    plt.scatter(peaks[i, np.nonzero(peaks[i,:])], widths[i, np.nonzero(widths[i,:])])
plt.xlabel('peaks')
plt.ylabel('widths')
plt.tight_layout()
plt.draw()
plt.savefig('./visu_simple/peaks_vs_widths')

plt.figure()
plt.hist(widths[np.nonzero(widths)])
plt.tight_layout()
plt.savefig('./visu_simple/widths')

plt.figure()
plt.hist(peaks[np.nonzero(peaks)])
plt.tight_layout()
plt.savefig('./visu_simple/peaks')

#####################################
# Scatter plots of poles and residues
plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][0])
    plt.scatter(np.real(results[i, 0, 0:nz:2]), np.imag(results[i, 0, 0:nz:2]))
plt.title("Top half of poles")
plt.xlabel("Real(poles)")
plt.ylabel("Imag(poles)")
plt.tight_layout()
plt.savefig('./visu_simple/poles')

plt.figure()
for i in range(Nplot):
    nz = np.count_nonzero(results[i][1])
    plt.scatter(np.real(results[i, 1, 0:nz:2]), np.imag(results[i, 1, 0:nz:2]))
plt.title("Top half of residues")
plt.tight_layout()
plt.savefig('./visu_simple/residues')


###############################################################################
#                        Plot possible features                               #
###############################################################################

kk = 0
POLES  = np.zeros(2*Np*Nr, dtype = np.complex_)
dPOLES = np.zeros(2*Np*Nr, dtype = np.complex_)
RESID  = np.zeros(2*Np*Nr, dtype = np.complex_)
for i in range(Nplot):
    for j in range(np.count_nonzero(results[i, 0, :]) - 2):
        POLES[kk] = results[i, 0, j]
        dPOLES[kk] = results[i, 0, j+2] - results[i, 0, j]
        RESID[kk] = results[i, 1, j]
        kk += 1
NZ = kk - 1

plt.figure()
plt.hist(np.real(dPOLES[0:NZ]), bins=100)
plt.xlabel("Delta real(poles)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_simple/real_Dpoles')

# Plot histograms of real and imaginary parts of poles
plt.figure()
plt.hist(np.real(POLES[0:NZ]), bins=100)
plt.tight_layout()
plt.savefig('./visu_simple/real_poles')

plt.figure()
plt.hist(np.imag(POLES[0:NZ]), bins=100)
plt.xlabel("Imag(poles)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_simple/imag_poles')

# Plot histograms of real and imaginary parts of residues
plt.figure()
plt.hist(np.real(RESID[0:NZ]), bins=100)
plt.xlabel("Real(residue)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_simple/real_resid')

plt.figure()
plt.hist(np.imag(RESID[0:NZ]), bins=100)
plt.xlabel("Imag(residue)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.savefig('./visu_simple/imag_resid')

