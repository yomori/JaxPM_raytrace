"""
Minimal demo: run N-body + post-Born ray tracing and plot maps + 3D.
No cross-check or per-step diagnostics. Uses nbody_ray_kdk wrapper.
"""
from pathlib import Path
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt

from jaxpm_raytrace import ray_tracing_postborn
from jaxpm_raytrace.pm import linear_field, lpt, nbody_kdk, nbody_ray_kdk

jax.config.update("jax_enable_x64", True)


def _scalar(x):
    return float(jnp.asarray(x).reshape(()))


def _a_at_chi_target(cosmo, chi_target, a_min=1e-4, n_iter=80):
    """Invert chi(a) by bisection. Returns scale factor where chi(a) = chi_target."""
    if chi_target <= 0.0:
        return 1.0
    chi_min = _scalar(jc.background.radial_comoving_distance(cosmo, a_min))
    if chi_target >= chi_min:
        return float(a_min)
    lo, hi = float(a_min), 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        chi_mid = _scalar(jc.background.radial_comoving_distance(cosmo, mid))
        if chi_mid > chi_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build_backward_a_steps(cosmo, chi_max, n_shells):
    """Descending a-steps with chi in [0, chi_max]; ray tracing stops at box far side."""
    chi_edges = jnp.linspace(0.0, float(chi_max), n_shells + 1, dtype=jnp.float64)
    a_steps = jnp.array([_a_at_chi_target(cosmo, _scalar(c)) for c in chi_edges], dtype=jnp.float64)
    return a_steps


# Config
mesh_shape = (64, 64, 1024)
box_size = (256.0, 256.0, 4096.0)
ray_grid_shape = (32, 32)
n_ray_shells = 8
a_lpt = 0.1
n_forward = 20

cosmo = jc.Planck15()
a_steps = _build_backward_a_steps(cosmo, box_size[2], n_ray_shells)
r_src = _scalar(jc.background.transverse_comoving_distance(cosmo, _scalar(a_steps[-1])))
ray_cell_size = (
    box_size[0] / ((ray_grid_shape[0] - 1) * r_src),
    box_size[1] / ((ray_grid_shape[1] - 1) * r_src),
)
a_forward = jnp.linspace(a_lpt, 1.0, n_forward, dtype=jnp.float64)

# N-body IC and forward run
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

# Ray grid and combined N-body + ray run (wrapper)
theta0, eta0, A0, B0 = ray_tracing_postborn.init_ray_grid(
    ray_grid_shape, ray_cell_size, dtype=jnp.float64
)
(pos, vel), (theta, eta, A, B), (kappa, gamma1, gamma2, omega), (ray_chi_per_step, ray_theta_per_step) = nbody_ray_kdk(
    particles, vel, theta0, eta0, A0, B0, a_steps, cosmo, mesh_shape, box_size, return_ray_history=True
)

# Observables 2D maps
nx, ny = ray_grid_shape
kappa_2d = jnp.reshape(kappa, (nx, ny))
gamma1_2d = jnp.reshape(gamma1, (nx, ny))
gamma2_2d = jnp.reshape(gamma2, (nx, ny))

out_dir = Path(__file__).resolve().parent
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
axes[0].imshow(jnp.asarray(kappa_2d), origin="lower")
axes[0].set_title("kappa")
axes[1].imshow(jnp.asarray(gamma1_2d), origin="lower")
axes[1].set_title("gamma1")
axes[2].imshow(jnp.asarray(gamma2_2d), origin="lower")
axes[2].set_title("gamma2")
plt.tight_layout()
fig.savefig(out_dir / "demo_fast_maps.png", dpi=120)
plt.close()
print("Saved", out_dir / "demo_fast_maps.png")

# 3D plot: particles (one periodic box) and ray trajectories from per-step history
# Particle positions: wrap into [0, L) so the plot shows one periodic box.
Lx, Ly, Lz = box_size[0], box_size[1], box_size[2]
cell_size = jnp.array([Lx / mesh_shape[0], Ly / mesh_shape[1], Lz / mesh_shape[2]])
pos_mpch = jnp.mod(pos * cell_size, jnp.array([Lx, Ly, Lz]))

# Ray 3D from returned history: ray_theta_per_step (n_steps+1, n_ray, 2), ray_chi_per_step (n_steps+1,)
ray_x = Lx / 2.0 + ray_theta_per_step[:, :, 0] * ray_chi_per_step[:, None]
ray_y = Ly / 2.0 + ray_theta_per_step[:, :, 1] * ray_chi_per_step[:, None]
ray_z = ray_chi_per_step[:, None] + jnp.zeros_like(ray_x)
n_ray = ray_x.shape[1]
# Downsample rays by factor 4 so we plot 1/4 of them
ray_downsample = 4
ray_indices = jnp.arange(0, n_ray, ray_downsample)

max_pts = 8000
rng = jax.random.PRNGKey(1)
idx = jax.random.permutation(rng, jnp.arange(pos.shape[0]))[:max_pts]
pos_plot = jnp.asarray(pos_mpch[idx])

fig3d = plt.figure(figsize=(8, 8))
ax3d = fig3d.add_subplot(111, projection="3d")
ax3d.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2], c="black", s=0.3, alpha=0.5)
for idx_j in range(ray_indices.shape[0]):
    j = int(ray_indices[idx_j])
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
ax3d.set_title("Particles (black, periodic box) and ray paths (red)")
plt.tight_layout()
fig3d.savefig(out_dir / "demo_fast_3d.png", dpi=120)
plt.close()
print("Saved", out_dir / "demo_fast_3d.png")
