import jax_cosmo as jc
from jaxpm.growth import *

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
    State: (pos, vel); acc computed each step. Returns (pos_f, vel_f) or list of (pos, vel) if return_all.
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


def nbody_ray_kdk(pos, vel, theta, eta, A, B, a_steps, cosmo, mesh_shape, box_size):
    """
    N-body and ray tracing in lockstep (pmwd-style KDK, backward in a).
    a_steps: descending scale factors [a0=1, ..., a_min] (observer to source).
    Per step: grad_phi from current pos, ray KDK step, then particle KDK step.
    Returns (pos, vel), (theta, eta, A, B), (kappa, gamma1, gamma2, omega).
    """
    jc.background.radial_comoving_distance(cosmo, 1.0)
    growth_factor(cosmo, jnp.atleast_1d(a_steps[0]))
    acc = nbody_init_acc(pos, mesh_shape, cosmo)
    n = len(a_steps)
    for i in range(n - 1):
        a_prev = float(a_steps[i])
        a_next = float(a_steps[i + 1])
        grad_phi_3d = ray_tracing_postborn.pm_gradient_field(pos, mesh_shape, cosmo)
        theta, eta, A, B = ray_tracing_postborn.ray_step_kdk(
            theta, eta, A, B, a_prev, a_next, grad_phi_3d, cosmo, box_size, mesh_shape
        )
        pos, vel, acc = nbody_kdk_step(a_prev, a_next, pos, vel, acc, cosmo, mesh_shape)
    kappa, gamma1, gamma2, omega = ray_tracing_postborn.observe_from_A(A)
    return (pos, vel), (theta, eta, A, B), (kappa, gamma1, gamma2, omega)