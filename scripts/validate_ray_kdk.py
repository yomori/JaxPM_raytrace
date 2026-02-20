# Deterministic validation for post-Born ray KDK. Run: python validate_ray_kdk.py
# Checks: observe_from_A(I)=0; zero grad_phi => A=I,B=0; nonzero grad => A deviates; nbody_ray_kdk returns finite.
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm_raytrace import ray_tracing_postborn
from jaxpm_raytrace.pm import nbody_ray_kdk, linear_field, lpt
from jaxpm_raytrace.growth import growth_factor


def test_observe_from_A_identity():
    """A = I => kappa = gamma1 = gamma2 = omega = 0."""
    n_ray = 4
    A = jnp.tile(jnp.eye(2, dtype=jnp.float32), (n_ray, 1, 1))
    kappa, g1, g2, omega = ray_tracing_postborn.observe_from_A(A)
    assert jnp.allclose(kappa, 0.0)
    assert jnp.allclose(g1, 0.0)
    assert jnp.allclose(g2, 0.0)
    assert jnp.allclose(omega, 0.0)


def test_zero_force_ray_step():
    """Zero grad_phi => one KDK step preserves A=I, B=0, eta=0."""
    cosmo = jc.Planck15()
    mesh_shape = (16, 16, 16)
    box_size = (32.0, 32.0, 32.0)
    theta, eta, A, B = ray_tracing_postborn.init_ray_grid((4, 4), 1e-5, dtype=jnp.float32)
    grad_phi_3d = jnp.zeros((*mesh_shape, 3), dtype=jnp.float32)
    theta_f, eta_f, A_f, B_f = ray_tracing_postborn.ray_step_kdk(
        theta, eta, A, B, 1.0, 0.5, grad_phi_3d, cosmo, box_size, mesh_shape
    )
    I = jnp.eye(2, dtype=jnp.float32)
    atol = 1e-5
    assert jnp.allclose(A_f, jnp.tile(I, (theta.shape[0], 1, 1)), atol=atol)
    assert jnp.allclose(B_f, 0.0, atol=atol)
    assert jnp.allclose(eta_f, 0.0, atol=atol)


def test_nonzero_grad_changes_A():
    """Non-zero grad_phi in one KDK step => A deviates from I."""
    cosmo = jc.Planck15()
    mesh_shape = (16, 16, 16)
    box_size = (32.0, 32.0, 32.0)
    theta, eta, A, B = ray_tracing_postborn.init_ray_grid((4, 4), 1e-5, dtype=jnp.float32)
    grad_phi_3d = jnp.zeros((*mesh_shape, 3), dtype=jnp.float32)
    nz = mesh_shape[2]
    grad_phi_3d = grad_phi_3d.at[8, 8, nz - 1, 0].set(1e6)
    theta_f, eta_f, A_f, B_f = ray_tracing_postborn.ray_step_kdk(
        theta, eta, A, B, 1.0, 0.5, grad_phi_3d, cosmo, box_size, mesh_shape
    )
    I = jnp.eye(2, dtype=jnp.float32)
    assert jnp.max(jnp.abs(A_f - jnp.tile(I, (theta.shape[0], 1, 1)))) > 1e-8
    kappa, g1, g2, _ = ray_tracing_postborn.observe_from_A(A_f)
    assert jnp.max(jnp.abs(kappa)) > 1e-10 or jnp.max(jnp.abs(g1)) > 1e-10 or jnp.max(jnp.abs(g2)) > 1e-10


def test_nbody_ray_kdk_finite():
    """nbody_ray_kdk with uniform particles: completes and returns finite observables."""
    mesh_shape = (32, 32, 32)
    box_size = (32.0, 32.0, 32.0)
    cosmo = jc.Planck15()
    particles = jnp.stack(
        jnp.meshgrid(*[jnp.arange(s, dtype=jnp.float32) for s in mesh_shape]), axis=-1
    ).reshape(-1, 3)
    vel = jnp.zeros_like(particles)
    theta, eta, A, B = ray_tracing_postborn.init_ray_grid((8, 8), 1e-5, dtype=jnp.float32)
    a_steps = jnp.array([1.0, 0.3])
    (_, _, (kappa, g1, g2, _)) = nbody_ray_kdk(
        particles, vel, theta, eta, A, B, a_steps, cosmo, mesh_shape, box_size
    )
    assert jnp.all(jnp.isfinite(kappa)) and jnp.all(jnp.isfinite(g1)) and jnp.all(jnp.isfinite(g2))


if __name__ == "__main__":
    test_observe_from_A_identity()
    print("observe_from_A(I)=0 OK")
    test_zero_force_ray_step()
    print("Zero-force ray step OK")
    test_nonzero_grad_changes_A()
    print("Nonzero grad changes A OK")
    test_nbody_ray_kdk_finite()
    print("nbody_ray_kdk finite OK")
