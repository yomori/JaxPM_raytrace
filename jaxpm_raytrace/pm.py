import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology

from jaxpm_raytrace.growth import (
    E,
    Gf,
    dEa,
    dGfa,
    dGf2a,
    growth_factor,
    growth_rate,
    growth_d2_dlna2,
    growth_factor_second,
    growth_rate_second,
)
from jaxpm_raytrace.kernels import fftk, gradient_kernel, invlaplace_kernel, longrange_kernel
from jaxpm_raytrace.painting import cic_paint, cic_read
from jaxpm_raytrace import ray_tracing_postborn



def pm_forces(positions, mesh_shape, delta=None, r_split=0):
    """
    Computes gravitational forces on particles using a PM scheme
    """
    if delta is None:
        delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), positions))
    elif jnp.isrealobj(delta):
        delta_k = jnp.fft.rfftn(delta)
    else:
        delta_k = delta

    # Computes gravitational potential
    kvec = fftk(mesh_shape)
    pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(kvec, r_split=r_split)
    # Computes gravitational forces
    return jnp.stack([cic_read(jnp.fft.irfftn(- gradient_kernel(kvec, i) * pot_k), positions) 
                      for i in range(3)], axis=-1)


def lpt(cosmo:Cosmology, init_mesh, positions, a, order=1):
    """
    Computes first and second order LPT displacement and momentum, 
    e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    a = jnp.atleast_1d(a)
    E = jnp.sqrt(jc.background.Esqr(cosmo, a)) 
    delta_k = jnp.fft.rfftn(init_mesh) # TODO: pass the modes directly to save one or two fft?
    mesh_shape = init_mesh.shape

    init_force = pm_forces(positions, mesh_shape, delta=delta_k)
    dx = growth_factor(cosmo, a) * init_force
    p = a**2 * growth_rate(cosmo, a) * E * dx
    f = a**2 * E * dGfa(cosmo, a) * init_force

    if order == 2:
        kvec = fftk(mesh_shape)
        pot_k = delta_k * invlaplace_kernel(kvec)

        delta2 = 0
        shear_acc = 0
        # for i, ki in enumerate(kvec):
        for i in range(3):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            # shear_ii = jnp.fft.irfftn(- ki**2 * pot_k)
            nabla_i_nabla_i = gradient_kernel(kvec, i)**2
            shear_ii = jnp.fft.irfftn(nabla_i_nabla_i * pot_k)
            delta2 += shear_ii * shear_acc 
            shear_acc += shear_ii

            # for kj in kvec[i+1:]:
            for j in range(i+1, 3):
                # Substract squared strict-up-triangle terms
                # delta2 -= jnp.fft.irfftn(- ki * kj * pot_k)**2
                nabla_i_nabla_j = gradient_kernel(kvec, i) * gradient_kernel(kvec, j)
                delta2 -= jnp.fft.irfftn(nabla_i_nabla_j * pot_k)**2

        init_force2 = pm_forces(positions, mesh_shape, delta=jnp.fft.rfftn(delta2))
        # NOTE: growth_factor_second is renormalized: - D2 = 3/7 * growth_factor_second
        dx2 = 3/7 * growth_factor_second(cosmo, a) * init_force2
        p2 = a**2 * growth_rate_second(cosmo, a) * E * dx2
        f2 = a**2 * E * dGf2a(cosmo, a) * init_force2

        dx += dx2
        p  += p2
        f  += f2

    return dx, p, f


def linear_field(mesh_shape, box_size, pk, seed):
    """
    Generate initial conditions.
    """
    kvec = fftk(mesh_shape)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2
                for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2])

    field = jax.random.normal(seed, mesh_shape)
    field = jnp.fft.rfftn(field) * pkmesh**0.5
    field = jnp.fft.irfftn(field)
    return field


def make_ode_fn(mesh_shape):

    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(pos, mesh_shape=mesh_shape) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return nbody_ode

def get_ode_fn(cosmo:Cosmology, mesh_shape):

    def nbody_ode(a, state, args):
        """
        State is an array [position, velocities]

        Compatible with [Diffrax API](https://docs.kidger.site/diffrax/)
        """
        pos, vel = state
        forces = pm_forces(pos, mesh_shape) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.stack([dpos, dvel])

    return nbody_ode


def _G_K(cosmo, a):
    """Growth factor for kick denominator: a^3 E^2 (d2D/dln2 + (2 + dlnH/dlna) f). Units [H0^2]."""
    Ea = E(cosmo, a)
    Esqr = jc.background.Esqr(cosmo, a)
    f = growth_rate(cosmo, a)
    d2_dlna2 = growth_d2_dlna2(cosmo, a)
    dlnH_dlna = a * dEa(cosmo, a) / (Ea + 1e-30)
    return (a ** 3) * Esqr * (d2_dlna2 + (2.0 + dlnH_dlna) * f)


def drift_factor_kdk(a_vel, a_prev, a_next, cosmo):
    """Drift factor for KDK: (D(a_next)-D(a_prev)) / G_D(a_vel). G_D = Gf. In [1/H0]."""
    D_next = growth_factor(cosmo, jnp.atleast_1d(a_next)).reshape(())
    D_prev = growth_factor(cosmo, jnp.atleast_1d(a_prev)).reshape(())
    G_D = Gf(cosmo, jnp.atleast_1d(a_vel)).reshape(())
    return (D_next - D_prev) / (G_D + 1e-30)


def kick_factor_kdk(a_acc, a_prev, a_next, cosmo):
    """Kick factor for KDK: (G_D(a_next)-G_D(a_prev)) / G_K(a_acc). In [1/H0]."""
    G_D_next = Gf(cosmo, jnp.atleast_1d(a_next)).reshape(())
    G_D_prev = Gf(cosmo, jnp.atleast_1d(a_prev)).reshape(())
    G_K = _G_K(cosmo, jnp.atleast_1d(a_acc)).reshape(())
    return (G_D_next - G_D_prev) / (G_K + 1e-30)


def nbody_init_acc(pos, mesh_shape, cosmo):
    """Initial acceleration at pos: 1.5 * Omega_m * pm_forces(pos). Shape (npart, 3)."""
    return pm_forces(pos, mesh_shape) * (1.5 * cosmo.Omega_m)


def nbody_kdk_step(a_prev, a_next, pos, vel, acc, cosmo, mesh_shape):
    """One KDK step: kick half, drift full, acc at new pos, kick half. Returns (pos, vel, acc)."""
    a_half = (a_prev + a_next) / 2.0
    k1 = kick_factor_kdk(a_prev, a_prev, a_half, cosmo)
    vel_half = vel + acc * k1
    d = drift_factor_kdk(a_half, a_prev, a_next, cosmo)
    pos_next = pos + vel_half * d
    acc_next = nbody_init_acc(pos_next, mesh_shape, cosmo)
    k2 = kick_factor_kdk(a_next, a_half, a_next, cosmo)
    vel_next = vel_half + acc_next * k2
    return pos_next, vel_next, acc_next


def nbody_kdk(pos, vel, a_steps, cosmo, mesh_shape, return_all=False):
    """
    N-body evolution with pmwd-style growth-factor KDK.
    a_steps: ascending scale factors [a_init, ..., a_final].
    Returns (pos_f, vel_f) or list of (pos, vel) if return_all.
    """
    growth_factor(cosmo, jnp.atleast_1d(a_steps[0]))
    acc = nbody_init_acc(pos, mesh_shape, cosmo)
    if return_all:
        out = [(pos, vel)]
    for i in range(len(a_steps) - 1):
        a_prev = float(a_steps[i])
        a_next = float(a_steps[i + 1])
        pos, vel, acc = nbody_kdk_step(a_prev, a_next, pos, vel, acc, cosmo, mesh_shape)
        if return_all:
            out.append((pos, vel))
    if return_all:
        return out
    return pos, vel


def nbody_ray_kdk(pos, vel, theta, eta, A, B, a_steps, cosmo, mesh_shape, box_size, return_ray_history=False):
    """
    N-body and ray tracing in lockstep (pmwd-style KDK, backward in a).
    a_steps: descending [a0=1, ..., a_min] (observer to source).
    Returns (pos, vel), (theta, eta, A, B), (kappa, gamma1, gamma2, omega).
    If return_ray_history is True, also returns (ray_chi_per_step, ray_theta_per_step):
      ray_chi_per_step: (n_steps+1,) chi at each step; ray_theta_per_step: (n_steps+1, n_ray, 2).
    """
    def _chi_scalar(a):
        chi = jc.background.radial_comoving_distance(cosmo, a)
        return float(jnp.asarray(chi).reshape(-1)[0])

    jc.background.radial_comoving_distance(cosmo, 1.0)
    growth_factor(cosmo, jnp.atleast_1d(a_steps[0]))
    acc = nbody_init_acc(pos, mesh_shape, cosmo)
    n = len(a_steps)
    chis_list = [_chi_scalar(a_steps[0])]
    thetas_list = [theta]
    for i in range(n - 1):
        a_prev = float(a_steps[i])
        a_next = float(a_steps[i + 1])
        chi_prev = _chi_scalar(a_prev)
        chi_next = _chi_scalar(a_next)
        if chi_prev >= float(box_size[2]):
            break
        if chi_next > float(box_size[2]):
            lo = a_next
            hi = a_prev
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                chi_mid = _chi_scalar(mid)
                if chi_mid > float(box_size[2]):
                    lo = mid
                else:
                    hi = mid
            a_next = 0.5 * (lo + hi)
            chi_next = _chi_scalar(a_next)
        grad_phi_3d = ray_tracing_postborn.pm_gradient_field(pos, mesh_shape, cosmo)
        theta, eta, A, B = ray_tracing_postborn.ray_step_kdk(
            theta, eta, A, B, a_prev, a_next, grad_phi_3d, cosmo, box_size, mesh_shape
        )
        chis_list.append(chi_next)
        thetas_list.append(theta)
        pos, vel, acc = nbody_kdk_step(a_prev, a_next, pos, vel, acc, cosmo, mesh_shape)
        if chi_next > float(box_size[2]):
            break
    kappa, gamma1, gamma2, omega = ray_tracing_postborn.observe_from_A(A)
    out = (pos, vel), (theta, eta, A, B), (kappa, gamma1, gamma2, omega)
    if return_ray_history:
        ray_chi_per_step = jnp.array(chis_list)
        ray_theta_per_step = jnp.stack(thetas_list, axis=0)
        return out + ((ray_chi_per_step, ray_theta_per_step),)
    return out