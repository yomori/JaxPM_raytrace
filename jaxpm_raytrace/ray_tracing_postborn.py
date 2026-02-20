"""
On-the-fly post-Born ray tracing (JCAP theory: 2hamilton, 3integration, 4observables).

Ray state: theta (n_ray, 2) [rad], eta (n_ray, 2) [H0 L^2], A (n_ray, 2, 2), B (n_ray, 2, 2).
Units: angles [rad], distances [Mpc/h], c in [Mpc/h].
"""
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from jaxpm_raytrace.kernels import fftk, gradient_kernel, invlaplace_kernel
from jaxpm_raytrace.painting import cic_paint, cic_read_2d

_C_MPC_H = 299792.458 / 100.0


def init_ray_grid(ray_grid_shape, ray_cell_size, dtype=jnp.float32):
    """Initialize ray state on a uniform 2D image-plane grid. theta [rad], eta=0, A=I, B=0."""
    nx, ny = ray_grid_shape
    n_ray = nx * ny
    ax = (jnp.arange(nx, dtype=dtype) - (nx - 1) / 2) * ray_cell_size
    ay = (jnp.arange(ny, dtype=dtype) - (ny - 1) / 2) * ray_cell_size
    theta = jnp.stack(jnp.meshgrid(ax, ay, indexing="ij"), axis=-1).reshape(-1, 2)
    eta = jnp.zeros((n_ray, 2), dtype=dtype)
    A = jnp.tile(jnp.eye(2, dtype=dtype), (n_ray, 1, 1))
    B = jnp.zeros((n_ray, 2, 2), dtype=dtype)
    return theta, eta, A, B


def observe_from_A(A):
    """Map final A to observables: kappa, gamma1, gamma2, omega (each n_ray,)."""
    C1 = (A[..., 0, 0] + A[..., 1, 1]) / 2
    C2 = (A[..., 0, 1] - A[..., 1, 0]) / 2
    C3 = (A[..., 1, 1] - A[..., 0, 0]) / 2
    C4 = (A[..., 0, 1] + A[..., 1, 0]) / 2
    kappa = 1 - C1
    omega = C2 / (1 - kappa)
    gamma1 = C3 + C4 * omega
    gamma2 = -C4 + C3 * omega
    return kappa, gamma1, gamma2, omega


def _chi_a(a, cosmo):
    """Radial comoving distance [Mpc/h] at scale factor a."""
    out = jc.background.radial_comoving_distance(cosmo, float(a))
    return jnp.asarray(out).reshape(())


def _r_a(a, cosmo):
    """Transverse comoving distance [Mpc/h]. Flat: r = chi."""
    out = jc.background.transverse_comoving_distance(cosmo, float(a))
    return jnp.asarray(out).reshape(())


def ray_force_at_positions(grad_phi_3d, theta, r_mid, box_size, mesh_shape, delta_chi, chi_lo, c_mpc_h=_C_MPC_H):
    """d eta at ray positions from 3D gradient. Kick: d eta = (2/c^2)(delta_chi/r) grad_perp."""
    nx, ny, nz = mesh_shape
    cell_size_x = box_size[0] / nx
    cell_size_y = box_size[1] / ny
    cell_size_z = box_size[2] / nz
    chi_mid = chi_lo + delta_chi / 2.0
    z_index = jnp.clip(jnp.rint(chi_mid / (cell_size_z + 1e-30)).astype(jnp.int32), 0, nz - 1)
    grad_2d = grad_phi_3d[:, :, z_index, :2]
    Lx2 = box_size[0] / 2.0
    Ly2 = box_size[1] / 2.0
    pos_ij = jnp.stack([
        (Lx2 + theta[:, 0] * r_mid) / (cell_size_x + 1e-30),
        (Ly2 + theta[:, 1] * r_mid) / (cell_size_y + 1e-30),
    ], axis=-1)
    grad_at_rays = cic_read_2d(grad_2d, pos_ij)
    factor = (2.0 / (c_mpc_h ** 2)) * (delta_chi / (r_mid + 1e-30))
    deta = factor * grad_at_rays
    return deta


def ray_step_kdk(theta, eta, A, B, a_prev, a_next, grad_phi_3d, cosmo, box_size, mesh_shape, c_mpc_h=_C_MPC_H):
    """One KDK step: kick, drift, kick; A/B transport. Backward in a."""
    a_mid = (a_prev + a_next) / 2.0
    chi_prev = _chi_a(a_prev, cosmo)
    chi_next = _chi_a(a_next, cosmo)
    delta_chi = jnp.abs(chi_next - chi_prev)
    r_vel = _r_a(a_mid, cosmo)
    factor_drift = delta_chi / (r_vel ** 2 + 1e-30) / c_mpc_h

    deta1, dB1 = _kick_and_dB(theta, eta, A, B, grad_phi_3d, a_prev, a_mid, cosmo, box_size, mesh_shape, c_mpc_h)
    eta = eta + deta1
    B = B + dB1
    theta = theta + factor_drift * eta
    A = A + factor_drift * B
    deta2, dB2 = _kick_and_dB(theta, eta, A, B, grad_phi_3d, a_mid, a_next, cosmo, box_size, mesh_shape, c_mpc_h)
    eta = eta + deta2
    B = B + dB2
    return theta, eta, A, B


def _kick_and_dB(theta, eta, A, B, grad_phi_3d, a_i, a_f, cosmo, box_size, mesh_shape, c_mpc_h):
    """Kick: deta at theta and dB = (d deta / d theta) @ A via JVP."""
    r_mid = _r_a((a_i + a_f) / 2.0, cosmo)
    chi_i = _chi_a(a_i, cosmo)
    chi_f = _chi_a(a_f, cosmo)
    delta_chi = jnp.abs(chi_f - chi_i)
    chi_lo = jnp.minimum(chi_i, chi_f)

    def deflection_fn(th):
        return ray_force_at_positions(
            grad_phi_3d, th, r_mid, box_size, mesh_shape, delta_chi, chi_lo, c_mpc_h
        )

    deta, f_linear = jax.linearize(deflection_fn, theta)
    Ax = A[:, :, 0]
    Ay = A[:, :, 1]
    dBx = f_linear(Ax)
    dBy = f_linear(Ay)
    dB = jnp.stack([dBx, dBy], axis=-1)
    return deta, dB


def pm_gradient_field(positions, mesh_shape, cosmo):
    """3D gradient of potential on mesh (nx, ny, nz, 3) in 1/[Mpc/h]."""
    delta = cic_paint(jnp.zeros(mesh_shape), positions)
    delta_k = jnp.fft.rfftn(delta)
    kvec = fftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec)
    grad = jnp.stack([
        jnp.fft.irfftn(-gradient_kernel(kvec, i) * pot_k) for i in range(3)
    ], axis=-1)
    return grad * (1.5 * cosmo.Omega_m)
