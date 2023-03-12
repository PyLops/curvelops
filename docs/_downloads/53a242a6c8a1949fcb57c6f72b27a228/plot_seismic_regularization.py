r"""
3. Seismic Regularization
=========================
This example shows how to use the Curvelet transform to
condition a missing-data seismic regularization problem.
"""

# %%
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pylops
from pylops.optimization.sparsity import fista
from scipy.signal import convolve

from curvelops import FDCT2D

np.random.seed(0)
warnings.filterwarnings("ignore")

# %%
# Setup
# =====
inputfile = "../testdata/seismic.npz"
inputdata = np.load(inputfile)

x = inputdata["R"][50, :, ::2]
x = x / np.abs(x).max()
taxis, xaxis = inputdata["t"][::2], inputdata["r"][0]

par = {}
par["nx"], par["nt"] = x.shape
par["dx"] = inputdata["r"][0, 1] - inputdata["r"][0, 0]
par["dt"] = inputdata["t"][1] - inputdata["t"][0]

# Add wavelet
wav = inputdata["wav"][::2]
wav_c = np.argmax(wav)
x = np.apply_along_axis(convolve, 1, x, wav, mode="full")
x = x[:, wav_c:][:, : par["nt"]]

# Gain
gain = np.tile((taxis**2)[:, np.newaxis], (1, par["nx"])).T
x *= gain

# Subsampling locations
perc_subsampling = 0.5
Nsub = int(np.round(par["nx"] * perc_subsampling))
iava = np.sort(np.random.permutation(np.arange(par["nx"]))[:Nsub])

# Restriction operator
Rop = pylops.Restriction((par["nx"], par["nt"]), iava, axis=0, dtype="float64")

y = Rop @ x
xadj = Rop.H @ y

# Apply mask
ymask = Rop.mask(x)

# %%
# Curvelet transform
# ==================

# %%
DCTOp = FDCT2D((par["nx"], par["nt"]), nbscales=4)

yc = DCTOp @ x
xcadj = DCTOp.H @ yc

# %%
opts_plot = dict(
    cmap="gray",
    vmin=-0.1,
    vmax=0.1,
    extent=(xaxis[0], xaxis[-1], taxis[-1], taxis[0]),
)

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 7))
axs[0].imshow(x.T, **opts_plot)
axs[0].set_title("Data")
axs[0].axis("tight")
axs[1].imshow(np.real(xcadj).T, **opts_plot)
axs[1].set_title("Adjoint curvelet")
axs[1].axis("tight")

# %%
# Reconstruction based on Curvelet transform
# ##########################################

# %%
# Combined modelling operator
RCop = Rop @ DCTOp.H
RCop.dims = (RCop.shape[1],)  # flatten
RCop.dimsd = (RCop.shape[0],)

# Inverse
pl1, _, cost = fista(RCop, y.ravel(), niter=100, eps=1e-3, show=True)
xl1 = (DCTOp.H @ pl1).real.reshape(x.shape)

# %%
fig, axs = plt.subplots(1, 4, sharey=True, figsize=(16, 7))
axs[0].imshow(x.T, **opts_plot)
axs[0].set_title("Data")
axs[0].axis("tight")
axs[1].imshow(ymask.T, **opts_plot)
axs[1].set_title("Masked data")
axs[1].axis("tight")
axs[2].imshow(xl1.T, **opts_plot)
axs[2].set_title("Reconstructed data")
axs[2].axis("tight")
axs[3].imshow((x - xl1).T, **opts_plot)
axs[3].set_title("Reconstruction error")
axs[3].axis("tight")

# %%
fig, ax = plt.subplots(figsize=(16, 2))
ax.plot(range(1, len(cost) + 1), cost, "k")
ax.set(xlim=[1, len(cost)])
fig.suptitle("FISTA convergence")
