#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def tensor(op_list):
    """Finds tensor product of a list of operators"""
    product = np.eye(1, dtype=np.complex) # Initialise product as a 1x1 I matrix
    for op in op_list:
        product = np.kron(product, op)
    return product


def aklt(N, closed=True, costheta=1., sintheta=1./3):
    """N: number of spins in the chain.
    closed: is the spin chain periodic or not.
    costheta, sintheta: can be used to set the biquadratic coupling constant."""

    # Define the operators (normalised prefactors)
   
    si = np.eye(3, dtype=np.complex) # Identity matrix
    sx = np.sqrt(1./2)*np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.complex) # Pauli x matrix
    sy = -1j*np.sqrt(1./2)*np.array([[0,1,0],[-1,0,1],[0,-1,0]],dtype=np.complex) # Pauli y matrix
    sz = np.array([[1,0,0],[0,0,0],[0,0,-1]],dtype=np.complex) # Pauli z matrix
    
    # Calculate sx, sy and sx operators for all positions in the chain
    # E.g for the 2nd spin on a 4 spin chain: I.Sx.I.I

    sx_list = []
    sy_list = []
    sz_list = []
    
    # For each position in the chain
    for j in range(N):
        # Initialise list of identity matrices
        op_list = [si]*N
        
        # Replace jth identity matrix with Pauli x matrix
        op_list[j] = sx
        # Calculate tensor product of all matrices in the list
        sx_list.append(tensor(op_list))
        
        # Repeat for other Pauli matrices
        op_list[j] = sy
        sy_list.append(tensor(op_list))
        
        op_list[j] = sz
        sz_list.append(tensor(op_list))
    
    # Set the condition for an open or closed chain
    Noc = N if closed else N-1
    
    # Initialise Hamiltonian
    H = 0     

    # Sum each spin's contribution to the Hamiltonian
    for j in range(Noc):
        a, b = j, (j+1) % N # Gets the index of j and j+1, returning to 0 at j=n+1
        
        s_dot_s = 0 # Initialise S_j.S_j+1
        
        s_dot_s += np.dot(sx_list[a], sx_list[b])
        s_dot_s += np.dot(sy_list[a], sy_list[b])
        s_dot_s += np.dot(sz_list[a], sz_list[b])
        
        H += costheta * s_dot_s + sintheta * np.dot(s_dot_s, s_dot_s)
        
    return np.sort(eigh(H)[0])

def open_vs_closed():

    # Print the lowest 10 energy levels for closed chain
    aklt_closed = aklt(4)
    print("Closed chain:")
    print(np.round(aklt_closed[:10], 2))

    # Print the lowest 10 energy levels for open chain
    aklt_open = aklt(4, False)
    print("Open chain:")
    print(np.round(aklt_open[:10], 2))

    # Plot energy spectra

    fig, ax = plt.subplots()

    ax.plot(aklt_closed[:10], 'ks--', label='Closed')
    ax.plot(aklt_open[:10], 'ko:', mfc='white', label='Open')

    ax.set_ylabel("Energy")
    ax.set_xlabel("Level Index")
    ax.legend()

    fig.show()

#%% Plot energy spectrum at range of theta

fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

theta = np.linspace(0, np.pi, 9)

marker = ['x', '+', 'v', '^', '<', '>', 's', 'D', 'o']

for i in range(len(theta)):
    spectrum = aklt(4, True, costheta=np.cos(theta[i]), sintheta=np.sin(theta[i]))
    ax[0].plot(spectrum[:7], marker=marker[i], color='k', ls=':', mfc='none', label=str(i)+r'$\pi$/'+str(len(theta)-1))

theta = np.linspace(np.pi, 2*np.pi, 9)

for i in range(len(theta)):
    spectrum = aklt(4, True, costheta=np.cos(theta[i]), sintheta=np.sin(theta[i]))
    ax[1].plot(spectrum[:7], marker=marker[i], color='k', ls=':', mfc='none', label=str(8+i)+r'$\pi$/'+str(len(theta)-1))

ax[0].set_ylabel("Energy")
ax[1].set_ylabel("Energy")
ax[0].set_xlabel("Level Index")
ax[1].set_xlabel("Level Index")
ax[0].legend(loc='upper right')
ax[1].legend(loc='lower right')


fig.savefig("/Users/josh/OneDrive - University of Leeds/QMBP/theta_spectrum.pdf")

#%% Plot energy gap versus theta

def gap(theta, closed):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    levels = aklt(5, closed, costheta, sintheta)
    return levels[1] - levels[0]

vgap = np.vectorize(gap)

theta = np.linspace(0, 2*np.pi, 100)

gap_closed = vgap(theta, True)
gap_open = vgap(theta, False)

fig, ax = plt.subplots()

ax.plot(theta/np.pi, gap_closed, color='k')

ax.set_ylabel(r"$\Delta{}E$")
ax.set_xlabel(r"$\theta/\pi$")

plt.axvspan(0, 0.25, facecolor='k', alpha=0.1)
plt.axvspan(1.25, 2, facecolor='k', alpha=0.1)

#plt.style.use("ggplot")

fig.savefig("/Users/josh/OneDrive - University of Leeds/QMBP/theta_vs_egap.pdf")
