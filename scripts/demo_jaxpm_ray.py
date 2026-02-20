"""
End-to-end demo: JaxPM post-Born ray tracing with step-by-step diagnostics.

This script traces key quantities in each KDK step:
- shell geometry (chi range, z index used from the mesh),
- force magnitude on the current mesh,
- ray kick/drift increments,
- distortion matrix evolution and observables.
"""
import os
from pathlib import Path

# Avoid JAX int64 truncation warnings in this debug/demo script.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxpm_raytrace import ray_tracing_postborn
from jaxpm_raytrace.pm import linear_field, lpt, nbody_init_acc, nbody_kdk, nbody_kdk_step, nbody_ray_kdk

jax.config.update("jax_enable_x64", True)

def _scalar(x):
    return float(jnp.asarray(x).reshape(()))


def _maxabs(x):
    return _scalar(jnp.max(jnp.abs(x)))


def _print_coverage_checks(theta, a_steps, cosmo, box_size, mesh_shape):
    chi = jnp.array(
        [_scalar(jc.background.radial_comoving_distance(cosmo, _scalar(a))) for a in a_steps]
    )
    chi_max = _scalar(jnp.max(chi))
    print("\n--- Geometry checks ---")
    print(f"chi(a_min) = {chi_max:.3f} [Mpc/h], box_size_z = {box_size[2]:.3f} [Mpc/h]")
    dx = box_size[0] / mesh_shape[0]
    dy = box_size[1] / mesh_shape[1]
    dz = box_size[2] / mesh_shape[2]
    print(f"cell sizes [Mpc/h]: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
    if abs(dx - dy) > 1e-12 or abs(dx - dz) > 1e-12:
        print("WARNING: non-uniform grid spacing.")
    if chi_max > box_size[2]:
        print("WARNING: source distance exceeds box_size_z; z-slice indexing will clip/wrap.")
    theta_x_max = _maxabs(theta[:, 0])
    theta_y_max = _maxabs(theta[:, 1])
    x_extent = theta_x_max * chi_max
    y_extent = theta_y_max * chi_max
    print(
        f"max transverse ray extent at source: x={x_extent:.3f}, y={y_extent:.3f} [Mpc/h], "
        f"half-box: ({box_size[0] / 2:.3f}, {box_size[1] / 2:.3f})"
    )
    if x_extent > box_size[0] / 2 or y_extent > box_size[1] / 2:
        print("WARNING: rays leave x/y box footprint at source.")
    idx = [int(round(c / dz)) for c in chi]
    print(f"z-index coverage (rounded): {idx}, dz={dz:.3f}")
    print("-----------------------\n")


def _a_at_chi_target(cosmo, chi_target, a_min=1e-4, n_iter=80):
    """Invert chi(a) by bisection for flat LCDM tables in jax_cosmo."""
    if chi_target <= 0.0:
        return 1.0
    chi_min = _scalar(jc.background.radial_comoving_distance(cosmo, a_min))
    if chi_target >= chi_min:
        return float(a_min)
    lo = float(a_min)  # chi(lo) is large
    hi = 1.0           # chi(hi)=0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        chi_mid = _scalar(jc.background.radial_comoving_distance(cosmo, mid))
        if chi_mid > chi_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build_backward_a_steps_to_box(cosmo, chi_max, n_shells):
    """Build descending a-steps whose chi-edges tile [0, chi_max]."""
    chi_edges = jnp.linspace(0.0, float(chi_max), n_shells + 1, dtype=jnp.float64)
    a_edges = jnp.array([_a_at_chi_target(cosmo, _scalar(c)) for c in chi_edges], dtype=jnp.float64)
    return a_edges, chi_edges


# Config
# Coarser transverse resolution + many more z points, with uniform spacing.
mesh_shape = (64, 64, 1024)
box_size = (256.0, 256.0, 4096.0)  # dx = dy = dz = 16 Mpc/h
ray_grid_shape = (64, 64)
cosmo = jc.Planck15()
# Ray shells are defined in chi-space so tracing stops exactly at box far side (z = box_size[2]).
n_ray_shells = 8
a_steps, chi_edges = _build_backward_a_steps_to_box(cosmo, box_size[2], n_ray_shells)
# Match ray-grid source-plane footprint to simulation x/y box extent.
r_src = _scalar(jc.background.transverse_comoving_distance(cosmo, _scalar(a_steps[-1])))
ray_cell_size = (
    box_size[0] / ((ray_grid_shape[0] - 1) * r_src),
    box_size[1] / ((ray_grid_shape[1] - 1) * r_src),
)
a_lpt = 0.1
a_forward = jnp.linspace(a_lpt, 1.0, 20, dtype=jnp.float64)

# LPT IC, following Introduction.ipynb pattern:
# 1) linear density from P(k), 2) LPT at a_lpt, 3) evolve to a=1.
# Use a dedicated cosmology object for P(k), so LPT sees a fresh growth cache.
cosmo_pk = jc.Planck15()
k = jnp.logspace(-4, 1, 128)
pk = jc.power.linear_matter_power(cosmo_pk, k)
pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(x.shape)
initial_conditions = linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(0))
particles_grid = jnp.stack(
    jnp.meshgrid(*[jnp.arange(s, dtype=jnp.float64) for s in mesh_shape], indexing="ij"), axis=-1
).reshape(-1, 3)
dx, p, _ = lpt(cosmo, initial_conditions, particles_grid, a_lpt)
particles, vel = nbody_kdk(particles_grid + dx, p, a_forward, cosmo, mesh_shape)

# Rays on image plane
theta0, eta0, A0, B0 = ray_tracing_postborn.init_ray_grid(
    ray_grid_shape, ray_cell_size, dtype=jnp.float64
)
theta, eta, A, B = theta0, eta0, A0, B0

print("IC mode: LPT @ a=0.1 then nbody_kdk -> a=1.0")
print(f"max |dx_lpt|: {_maxabs(dx):.3e}, max |p_lpt|: {_maxabs(p):.3e}")
print(
    "ray grid angular cell [rad]: "
    f"sx={_scalar(ray_cell_size[0]):.3e}, sy={_scalar(ray_cell_size[1]):.3e}; "
    f"source chi={_scalar(chi_edges[-1]):.2f} [Mpc/h]"
)
_print_coverage_checks(theta0, a_steps, cosmo, box_size, mesh_shape)

# Manual traced loop with per-step diagnostics; record ray (theta, chi) for 3D trajectories
pos, v = particles, vel
acc = nbody_init_acc(pos, mesh_shape, cosmo)
chis_list = [_scalar(jc.background.radial_comoving_distance(cosmo, _scalar(a_steps[0])))]
thetas_history = [theta]
for i in range(len(a_steps) - 1):
    a_prev = _scalar(a_steps[i])
    a_next = _scalar(a_steps[i + 1])
    a_mid = 0.5 * (a_prev + a_next)
    chi_prev = _scalar(jc.background.radial_comoving_distance(cosmo, a_prev))
    chi_next = _scalar(jc.background.radial_comoving_distance(cosmo, a_next))
    delta_chi = abs(chi_next - chi_prev)
    chi_lo = min(chi_prev, chi_next)
    chi_mid = chi_lo + 0.5 * delta_chi
    r_mid = _scalar(jc.background.transverse_comoving_distance(cosmo, a_mid))
    dz = box_size[2] / mesh_shape[2]
    z_idx = int(jnp.clip(jnp.rint(chi_mid / (dz + 1e-30)), 0, mesh_shape[2] - 1))

    grad_phi_3d = ray_tracing_postborn.pm_gradient_field(pos, mesh_shape, cosmo)
    deta_preview = ray_tracing_postborn.ray_force_at_positions(
        grad_phi_3d, theta, r_mid, box_size, mesh_shape, delta_chi, chi_lo
    )

    theta_old, eta_old = theta, eta
    theta, eta, A, B = ray_tracing_postborn.ray_step_kdk(
        theta, eta, A, B, a_prev, a_next, grad_phi_3d, cosmo, box_size, mesh_shape
    )
    chis_list.append(chi_next)
    thetas_history.append(theta)
    pos, v, acc = nbody_kdk_step(a_prev, a_next, pos, v, acc, cosmo, mesh_shape)

    kappa_i, g1_i, g2_i, omega_i = ray_tracing_postborn.observe_from_A(A)
    trace_dev = _maxabs((A[:, 0, 0] + A[:, 1, 1]) - 2.0)
    print(
        f"step {i:02d} a:{a_prev:.3f}->{a_next:.3f} chi_mid:{chi_mid:.1f} z_idx:{z_idx:2d} "
        f"|grad_phi|max:{_maxabs(grad_phi_3d):.3e} |deta|max:{_maxabs(deta_preview):.3e} "
        f"|dtheta|max:{_maxabs(theta-theta_old):.3e} |deta_step|max:{_maxabs(eta-eta_old):.3e} "
        f"|A-I|max:{_maxabs(A-jnp.eye(2, dtype=A.dtype)):.3e} tr(A)-2:{trace_dev:.3e} "
        f"|k|max:{_maxabs(kappa_i):.3e} |g1|max:{_maxabs(g1_i):.3e} |g2|max:{_maxabs(g2_i):.3e} "
        f"|w|max:{_maxabs(omega_i):.3e}"
    )

# Cross-check manual trace vs nbody_ray_kdk wrapper
(_, _, observables_fn) = nbody_ray_kdk(
    particles, vel, theta0, eta0, A0, B0, a_steps, cosmo, mesh_shape, box_size
)
kappa, gamma1, gamma2, omega = ray_tracing_postborn.observe_from_A(A)
print("\n--- Wrapper consistency check ---")
print("max |kappa_manual - kappa_fn| :", _maxabs(kappa - observables_fn[0]))
print("max |gamma1_manual - gamma1_fn|:", _maxabs(gamma1 - observables_fn[1]))
print("max |gamma2_manual - gamma2_fn|:", _maxabs(gamma2 - observables_fn[2]))
print("max |omega_manual - omega_fn| :", _maxabs(omega - observables_fn[3]))
print("---------------------------------\n")

# Reshape to 2D for plotting
nx, ny = ray_grid_shape
kappa_2d = jnp.reshape(kappa, (nx, ny))
gamma1_2d = jnp.reshape(gamma1, (nx, ny))
gamma2_2d = jnp.reshape(gamma2, (nx, ny))

print("final max |kappa| :", _maxabs(kappa_2d))
print("final max |gamma1|:", _maxabs(gamma1_2d))
print("final max |gamma2|:", _maxabs(gamma2_2d))
print("final max |omega| :", _maxabs(omega))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
axes[0].imshow(jnp.asarray(kappa_2d), origin="lower")
axes[0].set_title("kappa")
axes[0].set_xlabel("x")
axes[1].imshow(jnp.asarray(gamma1_2d), origin="lower")
axes[1].set_title("gamma1")
axes[1].set_xlabel("x")
axes[2].imshow(jnp.asarray(gamma2_2d), origin="lower")
axes[2].set_title("gamma2")
axes[2].set_xlabel("x")
plt.tight_layout()
out_dir = Path(__file__).resolve().parent
fig.savefig(out_dir / "demo_jaxpm_ray_maps.png", dpi=120)
plt.close()
print("Saved", out_dir / "demo_jaxpm_ray_maps.png")

# 3D plot: final particle positions (black dots) and ray trajectories (red). Flat: r(chi)=chi.
from mpl_toolkits.mplot3d import Axes3D
chis_arr = jnp.array(chis_list)
thetas_stack = jnp.stack(thetas_history, axis=0)
Lx, Ly, Lz = box_size[0], box_size[1], box_size[2]
ray_x = Lx / 2.0 + thetas_stack[:, :, 0] * chis_arr[:, None]
ray_y = Ly / 2.0 + thetas_stack[:, :, 1] * chis_arr[:, None]
ray_z = chis_arr[:, None] + jnp.zeros_like(ray_x)
n_ray = ray_x.shape[1]
max_pts = 8000
cell_size = jnp.array([Lx / mesh_shape[0], Ly / mesh_shape[1], Lz / mesh_shape[2]])
pos_mpch = pos * cell_size
rng = jax.random.PRNGKey(1)
idx = jax.random.permutation(rng, jnp.arange(pos.shape[0]))[:max_pts]
pos_plot = jnp.asarray(pos_mpch[idx])
ray_step = max(1, n_ray // 64)
fig3d = plt.figure(figsize=(8, 8))
ax3d = fig3d.add_subplot(111, projection="3d")
ax3d.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2], c="black", s=0.3, alpha=0.5)
for j in range(0, n_ray, ray_step):
    ax3d.plot(
        jnp.asarray(ray_x[:, j]),
        jnp.asarray(ray_y[:, j]),
        jnp.asarray(ray_z[:, j]),
        color="red",
        alpha=0.6,
        linewidth=0.5,
    )
ax3d.set_xlabel("x [Mpc/h]")
ax3d.set_ylabel("y [Mpc/h]")
ax3d.set_zlabel("z (chi) [Mpc/h]")
ax3d.set_title("Particles (black) and ray trajectories (red)")
plt.tight_layout()
fig3d.savefig(out_dir / "demo_jaxpm_ray_3d.png", dpi=120)
plt.close()
print("Saved", out_dir / "demo_jaxpm_ray_3d.png")
