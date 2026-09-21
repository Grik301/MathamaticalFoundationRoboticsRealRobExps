"""Regression checks for the lecture's numerical models, without a Bullet world.

These checks use conservation laws, known moments, and independent matrix
identities so sign, square-root, transpose, and factor-of-two changes fail.
"""
import io

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from lecture6_sim.mathematics import (
    TopicExperiment,
    box_muller,
    sphere_inverse,
    sphere_rejection,
    wrap,
    wrapped_gaussian,
)


def test_circle_seam_is_periodic_and_linear_average_is_misleading():
    angles = np.array([np.pi - .04, -np.pi + .04])
    circular_mean = np.angle(np.mean(np.exp(1j * angles)))
    assert abs(wrap(circular_mean - np.pi)) < 1e-14
    assert abs(angles.mean()) < 1e-14
    assert abs(wrap(angles[1] - angles[0]) - .08) < 1e-14

    grid = np.linspace(-np.pi, np.pi, 3001)
    for variance in (.06, 3.2):
        density = wrapped_gaussian(grid, np.pi - .08, variance)
        images = wrapped_gaussian(grid, np.pi - .08, variance, 'images')
        assert_allclose(density, images, atol=2e-13)
        assert_allclose(density[0], density[-1], atol=2e-13)
        assert_allclose(np.trapezoid(density, grid), 1., atol=2e-12)
        assert density.min() >= -1e-14


def test_sphere_samplers_have_uniform_area_and_rejection_volume_ratio():
    rng = np.random.default_rng(17)
    inverse = sphere_inverse(rng, 20000)
    rejected, rate = sphere_rejection(rng, 20000)
    for points in (inverse, rejected):
        assert_allclose(np.linalg.norm(points, axis=1), 1., atol=3e-15)
        assert_allclose(points.mean(axis=0), 0., atol=.015)
        # Uniform theta (instead of uniform cos(theta)) gives E[z^2]=1/2
        # and therefore fails this uniform surface-area check.
        assert_allclose(points.T @ points / len(points), np.eye(3) / 3, atol=.012)
    assert abs(rate - np.pi / 6) < .015


def test_box_muller_and_affine_covariance_are_not_just_matching_shapes():
    draws = box_muller(np.random.default_rng(10), 30000)
    assert_allclose(draws.mean(axis=0), 0., atol=.02)
    assert_allclose(np.cov(draws, rowvar=False, ddof=0), np.eye(2), atol=.025)
    assert_allclose(np.mean(draws**4, axis=0), [3., 3.], atol=.16)
    experiment = TopicExperiment(33, 'drone', 453)
    assert_allclose(experiment.samples[:, :2].mean(axis=0), [.35, -.15], atol=.045)
    assert_allclose(np.cov(experiment.samples[:, :2], rowvar=False, ddof=0),
                    [[.64, .384], [.384, .4804]], atol=.05)
    assert abs(experiment.metrics()['transformed_density_integral'] - 1.) < 1e-5
    assert_allclose(experiment.metrics()['jacobian_determinant'], .4, atol=1e-14)


def test_convolution_preserves_binomial_parity_and_variance():
    experiment = TopicExperiment(40, 'mobile', 523)
    for n in (1, 2, 9, experiment.walk_steps):
        probability = experiment.convolutions[n]
        positions = np.arange(-n, n + 1)
        assert_allclose(probability.sum(), 1., atol=1e-14)
        assert np.all(probability[(positions + n) % 2 == 1] == 0.)
        assert_allclose(probability @ positions, 0., atol=1e-12)
        assert_allclose(probability @ (positions**2), n, atol=1e-12)
    assert_allclose(experiment.convolutions[2], [.25, 0., .5, 0., .25])


def test_em_uses_ito_drift_and_coupled_refinement_improves_error():
    experiment = TopicExperiment(46, 'drone', 583)
    metrics = experiment.metrics()
    assert metrics['em_fine_dt'] == metrics['em_coarse_dt'] / 2
    assert 0 < metrics['em_fine_rmse'] < .9 * metrics['em_coarse_rmse']
    assert metrics['em_coupled_difference_rmse'] > 0
    assert_allclose(experiment.dw.var(), experiment.dt, rtol=.015)
    # The -sigma^2/2 term distinguishes the Ito solution from a
    # Stratonovich interpretation of the same written drift.
    log_solution = (.04 - .35**2 / 2) * experiment.times[:, None] + .35 * experiment.wiener
    assert_allclose(np.log(experiment.gbm_exact), log_solution, atol=2e-15)
    assert_allclose(experiment.gbm[-1].mean(), np.exp(.04 * 6), rtol=.06)


def test_fpe_zero_boundary_flux_conserves_mass_and_has_half_B_diffusion():
    experiment = TopicExperiment(49, 'mobile', 614)
    density, grid = experiment.fpe, experiment.grid
    dx = grid[1] - grid[0]
    assert density.min() >= 0.
    assert_allclose(density.sum(axis=1) * dx, 1., atol=3e-14)
    assert_allclose(np.asarray(experiment.fluxes)[:, [0, -1]], 0., atol=0.)
    mean = (density * grid).sum(axis=1) * dx
    variance = (density * (grid[None, :] - mean[:, None])**2).sum(axis=1) * dx
    # Independent continuum moment laws: d(mean)/dt=h and d(var)/dt=B.
    # The first-order upwind flux adds a small known numerical diffusion.
    assert_allclose(mean, -.7 + .16 * experiment.times, atol=2e-5)
    assert_allclose(variance, .06 + .22 * experiment.times, atol=.042)
    assert experiment.metrics()['fokker_planck_coefficient'] == .22 / 2
    assert experiment.metrics()['fpe_l1_error_against_gaussian'] < .03


def test_oscillator_covariance_has_correct_restoring_signs_and_noise_factor():
    experiment = TopicExperiment(51, 'drone', 633)
    assert_allclose(experiment.gamma, [[0., -1.], [2., 1.2]], atol=0.)
    assert_allclose(experiment.c, [[0.], [.75]], atol=0.)
    # Equipartition ratio Var(v)/Var(x)=k/m and stationary force intensity.
    expected = np.diag([.75**2 / (2 * 1.2 * 2.), .75**2 / (2 * 1.2)])
    assert_allclose(experiment.stationary, expected, atol=2e-15)
    initial = experiment.initial_mean
    assert_allclose(experiment.ou_mean[-1], expm(-experiment.gamma * 6) @ initial,
                    atol=2e-15)


def test_spectral_lyapunov_and_integral_agree_for_complex_nonnormal_modes():
    experiment = TopicExperiment(57, 'mobile', 694)
    gamma, b, stationary = experiment.gamma, experiment.b, experiment.stationary
    values, right = np.linalg.eig(gamma)
    assert np.any(abs(values.imag) > .1)  # Exercise complex conjugate modes.
    assert not np.allclose(gamma @ gamma.T, gamma.T @ gamma)
    left = np.linalg.inv(right)
    spectral = np.zeros(gamma.shape, dtype=complex)
    for i in range(2):
        for j in range(2):
            # Bilinear transpose, not a Hermitian inner product.
            spectral += np.outer(right[:, i], right[:, j]) * (left[i] @ b @ left[j]) / (values[i] + values[j])
    assert_allclose(spectral, stationary, atol=2e-15)
    assert_allclose(gamma @ stationary + stationary @ gamma.T, b, atol=2e-15)
    assert np.linalg.eigvalsh(stationary).min() > 0
    for index in (30, 90, 180):
        phi = expm(-gamma * experiment.times[index])
        assert_allclose(experiment.covariance_integral[index],
                        stationary - phi @ stationary @ phi.T, atol=2e-15)
    assert experiment.metrics()['quadrature_lyapunov_error'] < 1e-5
    assert experiment.metrics()['kronecker_covariance_residual'] < 1e-14


def test_scalar_ou_mean_and_variance_match_closed_form():
    experiment = TopicExperiment(58, 'drone', 703)
    times = experiment.times
    expected_mean = np.exp(-1.1 * times)
    stationary_variance = .6**2 / (2 * 1.1)
    expected_variance = .025 * np.exp(-2 * 1.1 * times) + stationary_variance * (1 - np.exp(-2 * 1.1 * times))
    assert_allclose(experiment.ou_mean[:, 0], expected_mean, atol=1e-15)
    assert_allclose(experiment.ou_cov[:, 0, 0], expected_variance, atol=1e-15)
    assert_allclose(experiment.ou[-1, :, 0].var(), expected_variance[-1], atol=.012)


def test_render_work_never_changes_a_stochastic_robot_command():
    experiment = TopicExperiment(58, 'mobile', 704)
    full = experiment.step(3.2)
    command_only = experiment.step(3.2, include_panel=False)
    for key in ('target', 'yaw', 'force'):
        assert_allclose(full[key], command_only[key], atol=0.)
    assert full['panel']['curves']
    assert command_only['panel'] == {}


@pytest.mark.parametrize('page', [26, 31, 35, 39, 41, 46, 48, 55, 58, 61])
def test_archive_roundtrips_without_pickle_and_preserves_numeric_source_data(page):
    experiment = TopicExperiment(page, 'drone', 123 + 10 * page)
    archive = experiment.archive()
    assert_allclose(archive['times'], experiment.times, atol=0.)
    assert_allclose(archive['reference_paths'], experiment.paths, atol=0.)
    assert_allclose(archive['external_forces'], experiment.forces, atol=0.)
    for key, value in archive.items():
        assert isinstance(value, np.ndarray), key
        assert value.dtype.kind in 'biufc', (key, value.dtype)
        assert np.isfinite(value).all(), key
        if key.endswith('_paths') and key != 'reference_paths':
            assert value.shape[1] <= 64, key
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **archive)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as restored:
        assert set(restored.files) == set(archive)
        for key, value in archive.items():
            assert_allclose(restored[key], value, atol=0.)
