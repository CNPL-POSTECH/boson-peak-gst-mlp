import numpy as np
import matplotlib.pyplot as plt

from dynasor import compute_dynamic_structure_factors, Trajectory
from dynasor.qpoints import get_spherical_qpoints
from dynasor.post_processing import compute_spherical_qpoint_average

# set log level
from dynasor.logging_tools import set_logging_level
set_logging_level('INFO')

trajectory_filename = '../../out.dump'
traj = Trajectory(
    trajectory_filename,
    trajectory_format='lammps_internal',
    frame_stop=40000)

print(traj)

q_points = get_spherical_qpoints(traj.cell, q_max=1, max_points=4000)

plt.figure(figsize=(3.4, 2.5), dpi=140)
plt.hist(np.linalg.norm(q_points, axis=1), bins=50)
plt.xlabel(r'$|\mathbf{q}|$ (1/Å)')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('figure1.ps', format='ps', bbox_inches='tight')

sample_raw = compute_dynamic_structure_factors(
    traj, q_points, dt=25.0, window_size=2000,
    window_step=500, calculate_currents=True)

print(sample_raw)

sample_raw.write_to_npz('test.npz')

sample_averaged = compute_spherical_qpoint_average(sample_raw, q_bins=60)

conversion_factor = 658.2119  # conversion from 1/fs to meV
sample_averaged.omega *= conversion_factor

fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=140)

ax.pcolormesh(sample_averaged.q_norms, sample_averaged.omega,
              sample_averaged.Sqw_coh.T,
              cmap='Blues', vmin=0, vmax=4)
ax.text(0.05, 0.85, '$S(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
#ax.plot([0, 1], [0, 36], alpha=0.5, ls='--', c='0.3', lw=2)

ax.set_xlabel('$|\mathbf{q}|$ (1/Å)')
ax.set_ylabel('Frequency (meV)')
ax.set_ylim([0, 25])

plt.savefig('figure2.ps', format='ps', bbox_inches='tight')

fig.tight_layout()

fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=140,
                         sharex=True, sharey=True)

ax = axes[0]
ax.pcolormesh(sample_averaged.q_norms, sample_averaged.omega,
              sample_averaged.Clqw.T, cmap='Reds', vmin=0, vmax=1000)
ax.text(0.05, 0.85, '$C_L(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
#ax.plot([0, 1.5], [0, 54], alpha=0.5, ls='--', c='0.3', lw=2)

ax = axes[1]
ax.pcolormesh(sample_averaged.q_norms, sample_averaged.omega,
              sample_averaged.Ctqw.T, cmap='Oranges', vmin=0, vmax=1000)
ax.text(0.05, 0.85, '$C_T(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
#ax.plot([0, 1.5], [0, 32], alpha=0.5, ls='--', c='0.3', lw=2)

ax.set_xlabel('$|\mathbf{q}|$ (1/Å)')
ax.set_ylabel('Frequency (meV)', y=1)
ax.set_ylim([0, 25])

fig.tight_layout()
plt.savefig('figure3.ps', format='ps', bbox_inches='tight')

plt.subplots_adjust(hspace=0)


q_inds = [8, 20, 32]  # slices in heatmap

fig, axes = plt.subplots(figsize=(6.2, 3.8), nrows=2, ncols=2,
                         sharex='col', dpi=140)

#-----
# Intermediate scattering function and dynamic structure factor
for q_ind in q_inds:
    label = fr'$|\mathbf{{q}}|$={sample_averaged.q_norms[q_ind]:.2f} 1/Å'
    axes[0][0].plot(sample_averaged.time, sample_averaged.Fqt[q_ind, :] * 100,
                    label=label, alpha=0.8)
    axes[0][1].plot(sample_averaged.omega, sample_averaged.Sqw[q_ind, :],
                    label=label, alpha=0.8)

ax = axes[0][0]
ax.set_ylabel(r'$F(|\mathbf{q}|, t) \times 10^2$')
ax.legend(frameon=False)

ax = axes[0][1]
ax.set_ylabel('$S(|\mathbf{q}|, \omega)$')

#-----
# Current correlation functions
q_ind = q_inds[0]
axes[1][0].plot(sample_averaged.time, sample_averaged.Clqt[q_ind, :],
                label='L')
axes[1][0].plot(sample_averaged.time, sample_averaged.Ctqt[q_ind, :],
                label='T', c='C3')
axes[1][1].plot(sample_averaged.omega, sample_averaged.Clqw[q_ind, :] / 1000,
                label='L')
axes[1][1].plot(sample_averaged.omega, sample_averaged.Ctqw[q_ind, :] / 1000,
                label='T', c='C3')

ax = axes[1][0]
ax.set_xlabel('Time (fs)')
ax.set_xlim([0, 10000])
ax.set_ylabel('$C(|\mathbf{q}|, t)$')
ax.legend(frameon=False)
ax.text(0.4, 0.85, fr'$|\mathbf{{q}}|$={sample_averaged.q_norms[q_ind]:.2f} 1/Å',
        ha='center', transform=ax.transAxes)

ax = axes[1][1]
ax.set_xlabel('Frequency (meV)')
ax.set_xlim([0, 25])
ax.set_ylim([0, 0.8])
ax.set_ylabel(r'$C(|\mathbf{q}|, \omega) \times 10^{-3}$')

fig.tight_layout()
plt.savefig('figure4.ps', format='ps', bbox_inches='tight')

plt.subplots_adjust(hspace=0)
fig.align_ylabels(axes)

# Define the frequency threshold (in meV)
frequency_threshold = 0.0  # Frequencies below this value will be excluded from the calculation

# Get the omega array and omega^2
omega = sample_averaged.omega  # Original omega array in meV
omega_squared = omega**2       # Square of omega

# Identify indices where omega >= frequency_threshold
valid_omega_indices = omega >= frequency_threshold

# Create a mask for omega^2 to avoid division by zero or small values
omega_squared_masked = np.where(valid_omega_indices, omega_squared, np.nan)  # Set invalid omega_squared to NaN

# Compute q^2
q_squared = sample_averaged.q_norms**2  # q_squared array

# Compute S_L and S_T, avoiding division by zero
S_L = (q_squared[:, np.newaxis]) * sample_averaged.Clqw / omega_squared_masked[np.newaxis, :]
S_T = (q_squared[:, np.newaxis]) * sample_averaged.Ctqw / omega_squared_masked[np.newaxis, :]

# Replace NaNs resulting from division by zero or frequencies below threshold with zeros
S_L = np.nan_to_num(S_L, nan=0.0)
S_T = np.nan_to_num(S_T, nan=0.0)

# Now, you can plot S_L and S_T starting from 0 meV
# Plotting S_L and S_T
fig, axes = plt.subplots(figsize=(3.4, 3.8), nrows=2, dpi=140, sharex=True, sharey=True)

# Plot S_L
ax = axes[0]
pcm = ax.pcolormesh(sample_averaged.q_norms, omega, S_L.T, cmap='Purples', shading='auto',vmin=0, vmax=60)
ax.text(0.05, 0.85, r'$S_L(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
fig.colorbar(pcm, ax=ax, label='Intensity')

# Plot S_T
ax = axes[1]
pcm = ax.pcolormesh(sample_averaged.q_norms, omega, S_T.T, cmap='Greens', shading='auto', vmin=0, vmax=60)
ax.text(0.05, 0.85, r'$S_T(|\mathbf{q}|, \omega)$', transform=ax.transAxes,
        bbox={'color': 'white', 'alpha': 0.8, 'pad': 3})
fig.colorbar(pcm, ax=ax, label='Intensity')
# Set labels and limits
ax.set_xlabel(r'$|\mathbf{q}|$ (1/Å)')
ax.set_ylabel('Frequency (meV)')
ax.set_ylim([0, 25])

fig.tight_layout()
plt.savefig('figure_S_L_T_above_1meV.ps', format='ps', bbox_inches='tight')


# Select specific q indices from your q_inds list
# Select a specific q index from your q_inds list
q_ind = q_inds[0]


# Plotting S_L and S_T at specific q points
fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=140)

q_value = sample_averaged.q_norms[q_ind]
label = fr'$|\mathbf{{q}}|$={q_value:.2f} 1/Å'
    
# Compute q^2
q_squared = q_value**2
    
# Compute S_L and S_T at the specific q point
S_L_q = q_squared * sample_averaged.Clqw[q_ind, :] / omega_squared_masked
S_T_q = q_squared * sample_averaged.Ctqw[q_ind, :] / omega_squared_masked
    
# Replace NaNs resulting from division by zero or frequencies below threshold with zeros
S_L_q = np.nan_to_num(S_L_q, nan=0.0)
S_T_q = np.nan_to_num(S_T_q, nan=0.0)


fig_SL_q, ax_SL_q = plt.subplots(figsize=(6, 5), dpi=140)
fig_ST_q, ax_ST_q = plt.subplots(figsize=(6, 5), dpi=140)
ax_SL_q.plot(omega, S_L_q / 1000, label=label)

ax_SL_q.set_xlabel('Frequency (meV)')
ax_SL_q.set_xlim([0, 25])
ax_SL_q.set_ylim([0, 0.02])
ax_SL_q.set_ylabel(r'$S_L(|\mathbf{q}|, \omega) \times 10^{-3}$')
ax_SL_q.legend(frameon=False)
fig_SL_q.tight_layout()

plt.savefig('S_L_q_points.ps', format='ps', bbox_inches='tight')
ax_ST_q.plot(omega, S_T_q / 1000, label=label)

ax_ST_q.set_xlabel('Frequency (meV)')
ax_ST_q.set_xlim([0, 25])
ax_ST_q.set_ylim([0, 0.02])
ax_ST_q.set_ylabel(r'$S_T(|\mathbf{q}|, \omega) \times 10^{-3}$')
ax_ST_q.legend(frameon=False)
fig_ST_q.tight_layout()
plt.savefig('S_T_q_points.ps', format='ps', bbox_inches='tight')
plt.show() 
