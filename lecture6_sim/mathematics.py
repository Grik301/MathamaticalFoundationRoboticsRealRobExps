"""Numerical counterparts of lecture 6, PDF pages 26--61.

The mathematical ensembles in this module are reduced stochastic models, not
multiple simulated robot bodies. One sampled reference is sent to the actual
actuated URDF, whose pose/velocity/contact evolution is handled by PyBullet.
Every random experiment is reproducible; numerical checks are included in its
metrics. No analytical curve is presented as a physical robot measurement.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.special import ndtr
from .topics import get_topic


TAU = 2 * np.pi


def wrap(theta):
    return (np.asarray(theta) + np.pi) % TAU - np.pi


def gaussian(x, mean=0., variance=1.):
    variance = max(float(variance), 1e-12)
    return np.exp(-.5 * (np.asarray(x) - mean)**2 / variance) / np.sqrt(TAU * variance)


def wrapped_gaussian(theta, mean=0., variance=1., method='fourier'):
    theta = np.asarray(theta)
    if method == 'images':
        shifts = np.arange(-12, 13)[:, None]
        return gaussian(theta[None, :] - TAU * shifts, mean, variance).sum(axis=0)
    modes = np.arange(1, 100)[:, None]
    return (1 + 2 * (np.exp(-.5 * variance * modes**2) *
                     np.cos(modes * (theta[None, :] - mean))).sum(axis=0)) / TAU


def box_muller(rng, n):
    u = np.maximum(rng.random(n), np.finfo(float).tiny)
    phi = TAU * rng.random(n)
    r = np.sqrt(-2 * np.log(u))
    return r[:, None] * np.column_stack((np.cos(phi), np.sin(phi)))


def sphere_inverse(rng, n):
    z = 1 - 2 * rng.random(n)
    phi = TAU * rng.random(n)
    r = np.sqrt(np.maximum(0, 1 - z*z))
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def sphere_rejection(rng, n):
    accepted, proposed, total = [], 0, 0
    while total < n:
        v = rng.uniform(-1, 1, (max(100, 2*(n-total)), 3))
        norms = np.linalg.norm(v, axis=1)
        mask = (norms > 0) & (norms <= 1)
        accepted.append(v[mask] / norms[mask, None])
        total += int(mask.sum())
        proposed += len(v)
    return np.concatenate(accepted)[:n], total / proposed


def _curve(x, y, label):
    return {'x': np.asarray(x).tolist(), 'y': np.asarray(y).tolist(), 'label': label}


def _hist(values, bins=40, limits=None):
    counts, edges = np.histogram(values, bins=bins, range=limits)
    density = counts / (max(1, len(values)) * np.diff(edges))
    return .5*(edges[1:] + edges[:-1]), density


def _panel(title, xlabel, ylabel, curves=(), samples=None, notes=(), xlim=None, ylim=None,
           samples_label='Numerical sample draws'):
    out = {'title': title, 'xlabel': xlabel, 'ylabel': ylabel,
           'curves': list(curves), 'notes': list(notes)}
    if samples is not None:
        out['samples'] = np.asarray(samples).tolist()
        out['samples_label'] = samples_label
    if xlim is not None:
        out['xlim'] = list(xlim)
    if ylim is not None:
        out['ylim'] = list(ylim)
    return out


def _trapz(y, x, axis=-1):
    # NumPy 1.x and 2.x compatibility.
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x, axis=axis)


def _ks_standard_normal(values):
    x = np.sort(values)
    n = len(x)
    f = ndtr(x)
    return float(max(np.max(np.arange(1, n+1)/n - f),
                     np.max(f - np.arange(n)/n)))


def _ou_paths(rng, times, count, gamma, c, mean0, cov0):
    """Exact linear Gaussian transition, including degenerate diffusion."""
    gamma, c = np.asarray(gamma), np.asarray(c)
    stationary = solve_continuous_lyapunov(gamma, c @ c.T)
    stationary = .5*(stationary + stationary.T)
    a = expm(-gamma * (times[1]-times[0]))
    q = stationary - a @ stationary @ a.T
    vals, vecs = np.linalg.eigh(.5*(q+q.T))
    factor = vecs @ np.diag(np.sqrt(np.maximum(vals, 0)))
    states = np.empty((len(times), count, len(gamma)))
    states[0] = rng.multivariate_normal(mean0, cov0, count)
    for k in range(len(times)-1):
        states[k+1] = states[k] @ a.T + rng.normal(size=(count, len(gamma))) @ factor.T
    means, covariances = [], []
    for t in times:
        phi = expm(-gamma*t)
        means.append(phi @ mean0)
        covariances.append(stationary + phi @ (cov0-stationary) @ phi.T)
    return states, np.asarray(means), np.asarray(covariances), stationary


class TopicExperiment:
    """Reproducible mathematical experiment with a bounded robot reference.

    ``step(t)`` is side-effect free, so physics/control/render rates may differ.
    ``force`` is an external world-frame force in Newtons. ``target`` is a
    world-frame position; the mobile robot uses only its x/y components.
    ``panel`` is serializable and contains explicit numerical-ensemble labels.
    """
    ensemble_size = 4096

    def __init__(self, page: int, kind: str, seed: int, duration: float=6., dt: float=1/30):
        self.topic = get_topic(page)
        if not (0 < duration <= 30 and 0 < dt <= duration):
            raise ValueError('Require 0 < dt <= duration <= 30 seconds')
        self.page, self.kind, self.seed = page, kind, int(seed)
        self.duration = float(duration)
        self.steps = max(2, int(np.ceil(duration/dt)))
        self.times = np.linspace(0, duration, self.steps+1)
        self.dt = self.times[1] - self.times[0]
        self.rng = np.random.default_rng(seed)
        self.paths = np.zeros((self.steps+1, 3))
        self.forces = np.zeros_like(self.paths)
        self.heading = np.zeros(self.steps+1)
        self._metrics = {'pdf_page': page, 'printed_slide': page-1, 'seed': int(seed),
                         'model_scope': 'Mathematical ensemble drives one actuated URDF reference; physical tracking is measured separately.',
                         'ensemble_size': self.ensemble_size, 'mathematical_dt_s': self.dt,
                         'stochastic_convention': 'Ito', 'checks': {}}
        if page <= 29:
            self._init_circle()
        elif page <= 37:
            self._init_sampling()
        elif page <= 42:
            self._init_walk()
        elif page <= 47:
            self._init_sde()
        elif page <= 49:
            self._init_fpe()
        elif page <= 58 or page >= 60:
            self._init_ou()
        else:
            self._metrics['model_scope'] = 'SLAM is implemented by the separate joint estimator, not this numerical reference.'
            self.paths[:, 0] = .6*np.sin(.6*self.times)
            self.paths[:, 1] = .5*(1-np.cos(.6*self.times))

    def _check(self, name, passed, **values):
        self._metrics['checks'][name] = bool(passed)
        self._metrics.update(values)

    def _reference_from_samples(self, values):
        values = np.asarray(values)
        if values.ndim == 1:
            values = np.column_stack((values, np.zeros(len(values))))
        # Interpolate 8 successive draws to give the real actuators time to move.
        knots = np.linspace(0, self.duration, 8)
        for j in range(min(3, values.shape[1])):
            self.paths[:, j] = np.interp(self.times, knots, values[:8, j])

    def _init_circle(self):
        n = self.ensemble_size
        self.mu, self.variance = np.pi-.15, .42**2
        self.angular_samples = wrap(self.rng.normal(self.mu, np.sqrt(self.variance), n))
        z = np.mean(np.exp(1j*self.angular_samples))
        theta = np.linspace(-np.pi, np.pi, 301)
        f = wrapped_gaussian(theta, self.mu, self.variance)
        image_f = wrapped_gaussian(theta, self.mu, self.variance, 'images')
        self._check('wrapped_normal_normalizes', abs(_trapz(f, theta)-1) < 1e-8,
                    wrapped_density_integral=float(_trapz(f, theta)))
        self._check('fourier_matches_shifted_gaussians', np.max(abs(f-image_f)) < 1e-8,
                    fourier_images_max_error=float(np.max(abs(f-image_f))))
        angle_error = float(abs(wrap(np.angle(z)-self.mu)))
        self._check('circular_mean_consistent', angle_error < .06,
                    circular_mean_rad=float(np.angle(z)), arithmetic_mean_rad=float(self.angular_samples.mean()),
                    circular_mean_error_rad=angle_error, resultant_length=float(abs(z)),
                    predicted_resultant_length=float(np.exp(-self.variance/2)))
        self.kappa = .8
        increments = np.sqrt(self.kappa*self.dt)*self.rng.normal(size=(self.steps, n))
        self.angle_paths = np.vstack((np.zeros((1,n)), np.cumsum(increments, axis=0)))
        if self.page == 26:
            self.heading = np.linspace(2.6, 4.1, self.steps+1)
        elif self.page == 29:
            self.heading = self.angle_paths[:, 0]
            final_resultant = abs(np.mean(np.exp(1j*self.angle_paths[-1])))
            expected = np.exp(-self.kappa*self.duration/2)
            self._check('circle_diffusion_resultant', abs(final_resultant-expected)<.06,
                        final_resultant_length=float(final_resultant), diffusion_predicted_resultant=float(expected))
        else:
            self.heading = np.interp(self.times, np.linspace(0,self.duration,8),
                                     np.unwrap(self.angular_samples[:8]))
        self.paths[:, :2] = np.column_stack((np.cos(self.heading), np.sin(self.heading)))

    def _init_sampling(self):
        n, rng = self.ensemble_size, self.rng
        self.samples = np.zeros((n, 3))
        self.sampling_means = {}
        if self.page == 31:
            self.rate = 1.5
            self.uniform = rng.random(n)
            self.values = -np.log1p(-self.uniform)/self.rate
            self.samples[:, 0] = self.values - 1/self.rate
            self.samples[:, 1] = .3*rng.normal(size=n)
            self._check('inverse_cdf_roundtrip', np.max(abs(1-np.exp(-self.rate*self.values)-self.uniform))<1e-12,
                        inverse_cdf_max_error=float(np.max(abs(1-np.exp(-self.rate*self.values)-self.uniform))),
                        empirical_mean=float(self.values.mean()), predicted_mean=1/self.rate,
                        empirical_variance=float(self.values.var()), predicted_variance=1/self.rate**2)
            self._check('exponential_mean', abs(self.values.mean()-1/self.rate)<.06)
        elif self.page in (33,35):
            z = box_muller(rng,n)
            self.mu_vec = np.array([.35,-.15])
            self.factor = np.array([[.8,0],[.48,.5]])
            self.covariance = self.factor @ self.factor.T
            self.samples[:,:2] = self.mu_vec + z @ self.factor.T
            empirical = np.cov(self.samples[:,:2], rowvar=False, ddof=0)
            self._check('affine_covariance', np.linalg.norm(empirical-self.covariance)<.10,
                        sample_mean=self.samples[:,:2].mean(axis=0).tolist(), expected_mean=self.mu_vec.tolist(),
                        sample_covariance=empirical.tolist(), expected_covariance=self.covariance.tolist(),
                        jacobian_determinant=float(np.linalg.det(self.factor)))
            grid=np.linspace(-5,5,321)
            xx,yy=np.meshgrid(grid,grid)
            d=np.stack((xx-self.mu_vec[0], yy-self.mu_vec[1]),axis=-1)
            precision=np.linalg.inv(self.covariance)
            density=np.exp(-.5*np.einsum('...i,ij,...j->...',d,precision,d))/(TAU*abs(np.linalg.det(self.factor)))
            integral=float(_trapz(_trapz(density,grid,axis=1),grid))
            self._check('jacobian_density_normalizes', abs(integral-1)<1e-5,
                        transformed_density_integral=integral)
        elif self.page == 34:
            self.samples[:,:2] = box_muller(rng,n)
            covariance=np.cov(self.samples[:,:2],rowvar=False,ddof=0)
            self._check('box_muller_moments', np.linalg.norm(covariance-np.eye(2))<.10 and np.linalg.norm(self.samples[:,:2].mean(0))<.08,
                        sample_mean=self.samples[:,:2].mean(0).tolist(),sample_covariance=covariance.tolist())
            self._check('box_muller_gaussian_ks', _ks_standard_normal(self.samples[:,0])<.035,
                        gaussian_ks_distance=_ks_standard_normal(self.samples[:,0]))
        else:
            self.sphere_a = sphere_inverse(rng,n)
            normals = rng.normal(size=(n,3))
            self.sphere_b = normals/np.linalg.norm(normals,axis=1)[:,None]
            self.sphere_c, self.acceptance = sphere_rejection(rng,n)
            self.samples = self.sphere_c if self.page in (36,37) else self.sphere_a
            cov=np.einsum('ni,nj->ij',self.samples,self.samples)/n
            self._check('sphere_unit_length', np.max(abs(np.linalg.norm(self.samples,axis=1)-1))<1e-12,
                        maximum_sphere_norm_error=float(np.max(abs(np.linalg.norm(self.samples,axis=1)-1))))
            self._check('sphere_isotropic_second_moment', np.linalg.norm(cov-np.eye(3)/3)<.055,
                        sphere_second_moment=cov.tolist(), sphere_mean=self.samples.mean(0).tolist())
            self._check('sphere_rejection_rate', abs(self.acceptance-np.pi/6)<.025,
                        rejection_acceptance_observed=float(self.acceptance), rejection_acceptance_expected=float(np.pi/6))
            if self.page==30:
                for name,sample in [('inverse_cdf',self.sphere_a),('normal_direction',self.sphere_b),('ball_rejection',self.sphere_c)]:
                    self.sampling_means[name]=np.mean(sample**2,axis=0).tolist()
                self._metrics['three_sampler_second_moments']=self.sampling_means
            if self.page==37:
                self.dimensions=np.arange(2,11)
                self.expected_acceptance=np.array([np.pi**(d/2)/(2**int(d)*math.gamma(d/2+1)) for d in self.dimensions])
                self.measured_acceptance=[]
                proposal_count=80000
                for d in self.dimensions:
                    proposals=rng.uniform(-1,1,(proposal_count,d))
                    self.measured_acceptance.append(float(np.mean(np.sum(proposals**2,axis=1)<=1)))
                self.measured_acceptance=np.asarray(self.measured_acceptance)
                sigma=np.sqrt(self.expected_acceptance*(1-self.expected_acceptance)/proposal_count)
                self._check('dimension_acceptance_rates', np.all(abs(self.measured_acceptance-self.expected_acceptance)<6*sigma),
                            acceptance_dimensions=self.dimensions.tolist(), acceptance_expected=self.expected_acceptance.tolist(),
                            acceptance_measured=self.measured_acceptance.tolist(), proposals_per_dimension=proposal_count)
        self._reference_from_samples(self.samples)

    def _init_walk(self):
        n=self.ensemble_size
        self.walk_steps=max(24,int(self.duration*10))
        self.walk_times=np.linspace(0,self.duration,self.walk_steps+1)
        increments=self.rng.choice([-1.,1.],size=(self.walk_steps,n))
        self.walk=np.vstack((np.zeros((1,n)),np.cumsum(increments,axis=0)))
        self.convolutions=[np.array([1.])]
        for _ in range(self.walk_steps):
            self.convolutions.append(np.convolve(self.convolutions[-1],[.5,0,.5]))
        p=self.convolutions[-1]
        positions=np.arange(-self.walk_steps,self.walk_steps+1)
        binomial=np.array([math.comb(self.walk_steps,(self.walk_steps+k)//2)/2**self.walk_steps
                           if (self.walk_steps+k)%2==0 else 0 for k in positions])
        self._check('convolution_matches_binomial', np.max(abs(p-binomial))<1e-12,
                    convolution_binomial_max_error=float(np.max(abs(p-binomial))))
        self._check('random_walk_mean_variance', abs(self.walk[-1].mean())<.5 and abs(self.walk[-1].var()-self.walk_steps)<.08*self.walk_steps,
                    walk_empirical_mean=float(self.walk[-1].mean()),walk_empirical_variance=float(self.walk[-1].var()),
                    walk_expected_variance=self.walk_steps)
        self._check('parity_support', np.all(p[(self.walk_steps+positions)%2==1]==0),probability_mass=float(p.sum()))
        scale=.12
        if self.page==41:
            uniform=self.rng.uniform(-np.sqrt(3),np.sqrt(3),(self.walk_steps,n))
            self.clt=np.cumsum(uniform,axis=0)/np.sqrt(np.arange(1,self.walk_steps+1))[:,None]
            ks=_ks_standard_normal(self.clt[-1])
            self._check('clt_gaussian_limit',ks<.04,clt_final_ks_distance=ks,
                        clt_first_ks_distance=_ks_standard_normal(self.clt[0]))
        if self.page==42:
            self.diffusion=.35
            self.walk_dt=self.duration/self.walk_steps
            self.step_length=np.sqrt(self.diffusion*self.walk_dt)
            scale=self.step_length
            var=self.walk[-1].var()*scale**2
            self._check('diffusion_variance', abs(var-self.diffusion*self.duration)<.10*self.diffusion*self.duration,
                        step_length=float(self.step_length),walk_time_step=self.walk_dt,
                        h_squared_over_dt=float(self.step_length**2/self.walk_dt),
                        diffusion_empirical_variance=float(var),diffusion_expected_variance=self.diffusion*self.duration,
                        fokker_planck_second_derivative_coefficient=self.diffusion/2)
        self.paths[:,0]=np.interp(self.times,self.walk_times,scale*self.walk[:,0])
        self.paths[:,1]=np.interp(self.times,self.walk_times,scale*self.walk[:,1])

    def _init_sde(self):
        n=self.ensemble_size
        self.dw=np.sqrt(self.dt)*self.rng.normal(size=(self.steps,n))
        self.wiener=np.vstack((np.zeros((1,n)),np.cumsum(self.dw,axis=0)))
        self._check('wiener_increment_scaling',abs(self.dw.var()/self.dt-1)<.02,
                    increment_variance=float(self.dw.var()),expected_increment_variance=self.dt,
                    wiener_terminal_variance=float(self.wiener[-1].var()),expected_wiener_variance=self.duration)
        lag=float(np.mean(self.dw[1:]*self.dw[:-1])/self.dt)
        self._check('wiener_independent_increments',abs(lag)<.01,lag_one_increment_correlation=lag)
        self.paths[:,:2]=.35*self.wiener[:,:2]
        if self.page in (45,46):
            self.drift,self.noise=.04,.35
            self.gbm=np.ones((self.steps+1,n))
            for k in range(self.steps):
                self.gbm[k+1]=self.gbm[k]+self.drift*self.gbm[k]*self.dt+self.noise*self.gbm[k]*self.dw[k]
            self.gbm_exact=np.exp((self.drift-.5*self.noise**2)*self.times[:,None]+self.noise*self.wiener)
            self.paths[:,:2]=self.gbm[:,:2]-1
            expected_mean=np.exp(self.drift*self.duration)
            exact_var=np.exp(2*self.drift*self.duration)*np.expm1(self.noise**2*self.duration)
            self._check('ito_gbm_moments',abs(self.gbm[-1].mean()-expected_mean)<.10,
                        gbm_empirical_mean=float(self.gbm[-1].mean()),gbm_expected_mean=float(expected_mean),
                        gbm_empirical_variance=float(self.gbm[-1].var()),gbm_exact_variance=float(exact_var))
            if self.page==46:
                # Coupled increments: each coarse increment is the sum of its two fine increments.
                fine_steps=2*self.steps
                fine_dt=self.duration/fine_steps
                fine_dw=np.sqrt(fine_dt)*self.rng.normal(size=(fine_steps,n))
                xfine=np.ones(n)
                for dw in fine_dw:
                    xfine += self.drift*xfine*fine_dt + self.noise*xfine*dw
                coarse_dw=fine_dw.reshape(self.steps,2,n).sum(axis=1)
                xcoarse=np.ones(n)
                for dw in coarse_dw:
                    xcoarse += self.drift*xcoarse*self.dt + self.noise*xcoarse*dw
                xexact=np.exp((self.drift-.5*self.noise**2)*self.duration+self.noise*fine_dw.sum(0))
                coarse_error=float(np.sqrt(np.mean((xcoarse-xexact)**2)))
                fine_error=float(np.sqrt(np.mean((xfine-xexact)**2)))
                self._check('em_refinement_improves_strong_error',fine_error<coarse_error,
                            em_coarse_dt=self.dt,em_fine_dt=fine_dt,em_coarse_rmse=coarse_error,
                            em_fine_rmse=fine_error,em_coupled_difference_rmse=float(np.sqrt(np.mean((xfine-xcoarse)**2))))
        if self.page==47:
            self.scalar_rate,self.scalar_noise=1.,.65
            values,mu,cov,stationary=_ou_paths(self.rng,self.times,n,np.array([[self.scalar_rate]]),
                                            np.array([[self.scalar_noise]]),np.array([1.]),np.array([[.025]]))
            self.ou_scalar=values[:,:,0]
            self.scalar_mean=mu[:,0]
            self.scalar_var=cov[:,0,0]
            self.paths[:,:2]=.7*values[:,:2,0]
            self._check('ou_ensemble_density_moments',abs(self.ou_scalar[-1].mean()-self.scalar_mean[-1])<.04 and abs(self.ou_scalar[-1].var()-self.scalar_var[-1])<.03,
                        terminal_empirical_mean=float(self.ou_scalar[-1].mean()),terminal_expected_mean=float(self.scalar_mean[-1]),
                        terminal_empirical_variance=float(self.ou_scalar[-1].var()),terminal_expected_variance=float(self.scalar_var[-1]))

    def _init_fpe(self):
        # Conservative finite-volume advection + diffusion, with zero boundary flux.
        self.fpe_drift,self.fpe_diffusion=.16,.22
        self.grid=np.linspace(-6,6,301)
        dx=self.grid[1]-self.grid[0]
        density=gaussian(self.grid,-.7,.06)
        density/=density.sum()*dx
        snapshots=[density.copy()]
        stable_dt=min(.4*dx/max(abs(self.fpe_drift),1e-9),.4*dx*dx/self.fpe_diffusion)
        substeps=max(1,int(np.ceil(self.dt/stable_dt)))
        subdt=self.dt/substeps
        self.fluxes=[]
        for _ in range(self.steps):
            for _ in range(substeps):
                flux=np.zeros(len(density)+1)
                # Positive drift: conservative upwind face value.
                flux[1:-1]=self.fpe_drift*density[:-1]-.5*self.fpe_diffusion*np.diff(density)/dx
                density -= subdt*np.diff(flux)/dx
            snapshots.append(density.copy())
            self.fluxes.append(flux.copy())
        self.fpe=np.asarray(snapshots)
        self.fpe_mean=-.7+self.fpe_drift*self.times
        self.fpe_var=.06+self.fpe_diffusion*self.times
        noise=self.rng.normal(size=(self.steps,self.ensemble_size))*np.sqrt(self.fpe_diffusion*self.dt)
        initial=self.rng.normal(-.7,np.sqrt(.06),self.ensemble_size)
        self.fpe_samples=np.vstack((initial[None,:], initial+self.fpe_drift*self.times[1:,None]+np.cumsum(noise,axis=0)))
        self.paths[:,:2]=.5*self.fpe_samples[:,:2]
        mass=self.fpe.sum(axis=1)*dx
        expected=gaussian(self.grid,self.fpe_mean[-1],self.fpe_var[-1])
        error=float(np.sum(abs(self.fpe[-1]-expected))*dx)
        self._check('fpe_probability_conservation',np.max(abs(mass-1))<1e-10,
                    maximum_probability_mass_error=float(np.max(abs(mass-1))),minimum_density=float(self.fpe.min()),
                    fpe_l1_error_against_gaussian=error,finite_volume_dx=float(dx),finite_volume_dt=float(subdt),
                    diffusion_B=self.fpe_diffusion,fokker_planck_coefficient=self.fpe_diffusion/2,
                    boundary_condition='zero probability flux')
        self._check('fpe_positive_density',self.fpe.min()>=-1e-12)
        self._check('fpe_matches_gaussian',error<.06)
        self._check('sde_matches_fpe_moments',abs(self.fpe_samples[-1].mean()-self.fpe_mean[-1])<.08 and abs(self.fpe_samples[-1].var()-self.fpe_var[-1])<.12,
                    sde_terminal_mean=float(self.fpe_samples[-1].mean()),analytic_terminal_mean=float(self.fpe_mean[-1]),
                    sde_terminal_variance=float(self.fpe_samples[-1].var()),analytic_terminal_variance=float(self.fpe_var[-1]))

    def _init_ou(self):
        n=self.ensemble_size
        self.gamma=np.array([[1.1,-.7],[.25,.8]])
        self.c=np.array([[.5,.1],[.05,.4]])
        self.initial_mean=np.array([.8,-.3])
        self.initial_cov=np.diag([.025,.02])
        if self.page in (50,51):
            self.mass,self.damping,self.stiffness,self.forcing=1.,1.2,2.,.75
            self.gamma=np.array([[0.,-1.],[self.stiffness/self.mass,self.damping/self.mass]])
            self.c=np.array([[0.],[self.forcing/self.mass]])
        if self.page==58:
            self.gamma=np.array([[1.1]])
            self.c=np.array([[.6]])
            self.initial_mean=np.array([1.])
            self.initial_cov=np.array([[.025]])
        self.ou,self.ou_mean,self.ou_cov,self.stationary=_ou_paths(
            self.rng,self.times,n,self.gamma,self.c,self.initial_mean,self.initial_cov)
        dim=len(self.gamma)
        self.b=self.c@self.c.T
        residual=self.gamma@self.stationary+self.stationary@self.gamma.T-self.b
        self._check('lyapunov_residual',np.linalg.norm(residual)<1e-10,
                    lyapunov_residual_norm=float(np.linalg.norm(residual)),
                    stationary_covariance=self.stationary.tolist(), gamma=self.gamma.tolist(), noise_C=self.c.tolist())
        eigenvalues,u=np.linalg.eig(self.gamma)
        vt=np.linalg.inv(u)
        projected=vt@self.b@vt.T
        spectral=u@(projected/(eigenvalues[:,None]+eigenvalues[None,:]))@u.T
        self._check('stable_drift',np.all(eigenvalues.real>0),drift_eigenvalue_real_parts=eigenvalues.real.tolist())
        self._check('positive_stationary_covariance',np.all(np.linalg.eigvalsh(self.stationary)>0),
                    stationary_covariance_eigenvalues=np.linalg.eigvalsh(self.stationary).tolist())
        self._check('biorthogonal_eigenvectors',np.linalg.norm(vt@u-np.eye(dim))<1e-10,
                    eigenvector_biorthogonality_residual=float(np.linalg.norm(vt@u-np.eye(dim))))
        self._check('eigen_reconstruction',np.linalg.norm(u@np.diag(eigenvalues)@vt-self.gamma)<1e-10,
                    eigendecomposition_reconstruction_residual=float(np.linalg.norm(u@np.diag(eigenvalues)@vt-self.gamma)))
        self._check('spectral_covariance_matches_lyapunov',np.linalg.norm(spectral-self.stationary)<1e-10,
                    spectral_covariance_residual=float(np.linalg.norm(spectral-self.stationary)))
        kronecker=np.kron(np.eye(dim),self.gamma)+np.kron(self.gamma,np.eye(dim))
        vectorized=np.linalg.solve(kronecker,self.b.reshape(-1,order='F')).reshape((dim,dim),order='F')
        self._check('vectorized_covariance_matches_lyapunov',np.linalg.norm(vectorized-self.stationary)<1e-10,
                    kronecker_covariance_residual=float(np.linalg.norm(vectorized-self.stationary)))
        empirical_mean=self.ou[-1].mean(axis=0)
        empirical_cov=np.atleast_2d(np.cov(self.ou[-1],rowvar=False,ddof=0))
        self._check('ou_ensemble_mean',np.linalg.norm(empirical_mean-self.ou_mean[-1])<.05,
                    final_empirical_mean=empirical_mean.tolist(),final_expected_mean=self.ou_mean[-1].tolist())
        self._check('ou_ensemble_covariance',np.linalg.norm(empirical_cov-self.ou_cov[-1])<.06,
                    final_empirical_covariance=empirical_cov.tolist(),final_expected_covariance=self.ou_cov[-1].tolist())
        self.covariance_integral=[]
        for t in self.times:
            phi=expm(-self.gamma*t)
            self.covariance_integral.append(self.stationary-phi@self.stationary@phi.T)
        self.covariance_integral=np.asarray(self.covariance_integral)
        if self.page in (55,57):
            horizon=max(12.,12./float(eigenvalues.real.min()))
            quadrature_t=np.linspace(0,horizon,3001)
            # Diagonalization evaluates the integral independently from Lyapunov.
            terms=[]
            for t in quadrature_t:
                phi=(u@np.diag(np.exp(-eigenvalues*t))@vt).real
                terms.append(phi@self.b@phi.T)
            quadrature=_trapz(np.asarray(terms),quadrature_t,axis=0)
            self._check('covariance_integral_matches_lyapunov',np.linalg.norm(quadrature-self.stationary)<1e-4,
                        quadrature_horizon=horizon,quadrature_lyapunov_error=float(np.linalg.norm(quadrature-self.stationary)))
        self.paths[:,0]=self.ou[:,0,0]
        self.paths[:,1]=self.ou[:,0,1] if dim>1 else self.ou[:,1,0]
        force_scale=.002 if self.kind in ('crazyflie','drone') else .5
        self.forces[:,:2]=force_scale*self.rng.normal(size=(self.steps+1,2))/np.sqrt(self.dt)
        self._metrics['physical_gust_force_std_N']=float(force_scale/np.sqrt(self.dt))
        if self.page==58:
            self._metrics['scalar_stationary_variance']=float(self.c[0,0]**2/(2*self.gamma[0,0]))
        if self.page>=60:
            x,y=self.rng.normal(size=(2,20000))
            independent_residual=float(abs(np.var(x+y)-np.var(x)-np.var(y)))
            correlated_residual=float(abs(np.var(x+x)-2*np.var(x)))
            self._check('independent_covariance_addition',independent_residual<.06,
                        independent_variance_addition_residual=independent_residual,
                        correlated_variance_addition_error=correlated_residual)
            self._check('correlated_addition_requires_cross_covariance',correlated_residual>1.5)
            draws=self.rng.normal(size=100000)
            ratios=[float(np.var(np.sqrt(h)*draws)/h) for h in (.1,.01,.001)]
            self._check('wiener_sqrt_dt_scaling',max(abs(np.asarray(ratios)-1))<.03,
                        increment_variance_divided_by_dt=ratios)
            stable=self.c[0,0]**2/(2*self.gamma[0,0])*(1-np.exp(-2*self.gamma[0,0]*self.times))
            unstable=self.c[0,0]**2/.8*np.expm1(.8*self.times)
            self.stability_curves=(stable,unstable)
            self._metrics['unstable_drift_example_variance_at_T']=float(unstable[-1])
            self._metrics['stability_condition']='All eigenvalues of Gamma have strictly positive real part for the demonstrated nondegenerate stationary Gaussian.'

    def step(self, t: float, *, include_panel: bool=True):
        """Return the same command with optional plotting work at camera ticks."""
        t=float(np.clip(t,0,self.duration))
        k=min(self.steps,int(round(t/self.dt)))
        ramp=1-np.exp(-2*t)
        # The raw mathematical state is plotted separately. A bounded transform
        # maps it into the robot's finite reachable demonstration area.
        target=np.array([.85*np.tanh(self.paths[k,0]),.85*np.tanh(self.paths[k,1]),1.+.2*np.tanh(self.paths[k,2])])
        target[:2]*=ramp
        target[2]=1.+(target[2]-1.)*ramp
        if self.page<=29:
            yaw=float(wrap(self.heading[k]))
        else:
            delta=self.paths[min(k+1,self.steps),:2]-self.paths[max(0,k-1),:2]
            yaw=float(np.arctan2(delta[1],delta[0])) if np.linalg.norm(delta)>1e-8 else 0.
        return {'target':target.tolist(),'yaw':yaw,'force':self.forces[k].tolist(),
                'panel':self._make_panel(k,t) if include_panel else {}}

    def _make_panel(self,k,t):
        p=self.page
        if p<=29:
            if p==26:
                theta=self.heading[:k+1]
                return _panel('Heading crosses the branch cut','time [s]','angle [rad]',[
                    _curve(self.times[:k+1],theta,'unwrapped heading reference'),
                    _curve(self.times[:k+1],wrap(theta),'wrapped heading')],
                    notes=['Endpoint identification preserves physical orientation.'],xlim=[0,self.duration],ylim=[-np.pi,4.3])
            variance=self.kappa*max(t,self.dt) if p==29 else self.variance
            mu=0. if p==29 else self.mu
            samples=wrap(self.angle_paths[k]) if p==29 else self.angular_samples
            x=np.linspace(-np.pi,np.pi,181)
            centers,hist=_hist(samples,36,(-np.pi,np.pi))
            curves=[_curve(x,wrapped_gaussian(x,mu,variance),'wrapped Gaussian theory'),
                    _curve(centers,hist,'mathematical ensemble density')]
            if p==28:
                curves.append(_curve(x,wrapped_gaussian(x,mu,variance,'images'),'shifted Gaussian sum'))
            curves.append(_curve(x,np.full_like(x,1/TAU),'uniform circle'))
            return _panel('Circular probability; one normalized period','angle [rad]','density',curves,
                          notes=[f'circular mean={np.angle(np.mean(np.exp(1j*samples))):.2f} rad',
                                 f'arithmetic mean={samples.mean():.2f} rad'],xlim=[-np.pi,np.pi])
        if p<=37:
            count=max(40,min(self.ensemble_size,int(self.ensemble_size*(.08+.92*t/self.duration))))
            if p==31:
                x=np.linspace(0,4,150)
                centers,hist=_hist(self.values[:count],32,(0,4))
                return _panel('Inverse-CDF exponential sampling','sample x','density',[
                    _curve(x,self.rate*np.exp(-self.rate*x),'target exponential density'),
                    _curve(centers,hist,'mathematical sample histogram')],
                    notes=[f'{count} independent inverse-CDF draws; exponential rate 1.5'],xlim=[0,4])
            if p==37:
                return _panel('Rejection rate collapses with dimension','dimension d','acceptance probability',[
                    _curve(self.dimensions,self.expected_acceptance,'ball/cube volume ratio'),
                    _curve(self.dimensions,self.measured_acceptance,'measured acceptance rate')],
                    notes=[f'd=10: predicted {100*self.expected_acceptance[-1]:.3f}%'],xlim=[2,10],ylim=[0,.85])
            if p in (30,32,36):
                x=np.linspace(-1,1,60)
                centers,hist=_hist(self.samples[:count,2],24,(-1,1))
                curves=[_curve(x,np.full_like(x,.5),'uniform sphere z density'),
                        _curve(centers,hist,'ball-rejection sample density' if p==36 else 'inverse-CDF sample density')]
                if p==30:
                    for name,samp in [('normalized-Gaussian sample density',self.sphere_b),('ball-rejection sample density',self.sphere_c)]:
                        cx,cy=_hist(samp[:count,2],24,(-1,1))
                        curves.append(_curve(cx,cy,name))
                return _panel('Uniform sphere area requires latitude weighting','sphere direction z','density',curves,
                              notes=[f'{count} mathematical directions',f'ball rejection rate: {100*self.acceptance:.1f}%'],xlim=[-1,1],ylim=[0,.9])
            if p in (33,35):
                angle=np.linspace(0,TAU,121)
                ellipse=self.mu_vec[:,None]+2*self.factor@np.stack((np.cos(angle),np.sin(angle)))
                curves=[_curve(ellipse[0],ellipse[1],'2-standard-deviation ellipse')]
                note=f'Jacobian determinant={np.linalg.det(self.factor):.3f}'
            else:
                angle=np.linspace(0,TAU,121)
                curves=[_curve(2*np.cos(angle),2*np.sin(angle),'radius 2; standard normal scale')]
                note='Independent standard normals from independent uniform draws'
            return _panel('Transformed mathematical sample cloud','first coordinate','second coordinate',curves,
                          samples=self.samples[:min(count,500),:2],notes=[note],xlim=[-3,3],ylim=[-3,3],
                          samples_label=('affine Gaussian sample pairs' if p in (33,35)
                                         else 'Box-Muller normal sample pairs'))
        if p<=42:
            n=max(1,min(self.walk_steps,int(t/self.duration*self.walk_steps)))
            if p==38:
                return _panel('Independent discrete random walkers','step n','lattice position K',[
                    _curve(np.arange(n+1),self.walk[:n+1,i],f'random-walk sample path {i+1}') for i in range(5)],
                    notes=[f'n={n}; empirical variance={self.walk[n].var():.2f}; theory={n}'],xlim=[0,self.walk_steps])
            if p in (39,40):
                x=np.arange(-n,n+1)
                empirical=np.array([np.mean(self.walk[n]==value) for value in x])
                return _panel('Convolution and parity of the binomial law','lattice position k','probability mass',[
                    _curve(x,self.convolutions[n],'exact repeated convolution'),
                    _curve(x,empirical,'mathematical ensemble frequency')],
                    notes=[f'{n} steps; reachable sites share the step count parity',f'total probability={self.convolutions[n].sum():.6f}'])
            if p==41:
                x=np.linspace(-4,4,150)
                centers,hist=_hist(self.clt[n-1],35,(-4,4))
                return _panel('Centered, scaled iid sums','standardized sum','density',[
                    _curve(x,gaussian(x),'standard normal limit'),_curve(centers,hist,f'uniform sum; n={n}')],
                    notes=[f'KS distance={_ks_standard_normal(self.clt[n-1]):.3f}'],xlim=[-4,4])
            tm=n*self.walk_dt
            lattice=np.arange(-n,n+1,2)*self.step_length
            pmf=self.convolutions[n][::2]/(2*self.step_length)
            xx=np.linspace(-4,4,150)
            return _panel('Random-walk diffusion limit','position x','density',[
                _curve(xx,gaussian(xx,0,self.diffusion*tm),'Gaussian diffusion law'),
                _curve(lattice,pmf,'lattice density')],
                notes=[f'Variance growth rate {self.diffusion:.2f}; diffusion coefficient {self.diffusion/2:.3f}'],xlim=[-4,4])
        if p<=47:
            if p==43:
                z=self.dw[:max(1,k)].reshape(-1)/np.sqrt(self.dt)
                centers,hist=_hist(z,35,(-4,4))
                x=np.linspace(-4,4,150)
                return _panel('Wiener increments: square-root time scaling','standardized increment','density',[
                    _curve(x,gaussian(x),'standard normal'),_curve(centers,hist,'mathematical increments')],
                    notes=[f'Increment variance {self.dw.var():.5f}; step duration {self.dt:.5f} s'],xlim=[-4,4])
            if p==44:
                return _panel('Wiener sample paths','time [s]','Wiener state',[
                    _curve(self.times[:k+1],self.wiener[:k+1,i],f'Wiener sample path {i+1}') for i in range(6)],
                    notes=[f'Ensemble variance {self.wiener[k].var():.3f}; predicted variance {t:.3f}'],xlim=[0,self.duration])
            if p in (45,46):
                notes=['Itô: drift and diffusion use the left endpoint.']
                if p==46:
                    notes.append(f'Coupled RMSE: coarse step {self._metrics["em_coarse_rmse"]:.4f}; half step {self._metrics["em_fine_rmse"]:.4f}')
                return _panel('Multiplicative-noise SDE','time [s]','state',[
                    _curve(self.times[:k+1],self.gbm[:k+1,0],'Euler-Maruyama numerical path'),
                    _curve(self.times[:k+1],self.gbm_exact[:k+1,0],'exact path; same Wiener input'),
                    _curve(self.times[:k+1],np.exp(self.drift*self.times[:k+1]),'analytical ensemble mean')],notes=notes,xlim=[0,self.duration])
            x=np.linspace(-1.7,2,160)
            centers,hist=_hist(self.ou_scalar[k],35,(-1.7,2))
            return _panel('An ensemble slice is an evolving density','state x','density',[
                _curve(x,gaussian(x,self.scalar_mean[k],self.scalar_var[k]),'analytical OU marginal'),
                _curve(centers,hist,'4096 independent numerical paths')],
                notes=[f't={t:.2f}s; variance={self.scalar_var[k]:.3f}'],xlim=[-1.7,2])
        if p<=49:
            centers,hist=_hist(self.fpe_samples[k],40,(-3.5,3.5))
            curves=[_curve(self.grid,self.fpe[k],'conservative finite-volume FPE'),
                    _curve(self.grid,gaussian(self.grid,self.fpe_mean[k],self.fpe_var[k]),'analytical Gaussian'),
                    _curve(centers,hist,'mathematical SDE ensemble')]
            notes=[f'probability mass={self.fpe[k].sum()*(self.grid[1]-self.grid[0]):.8f}',
                   f'Drift {self.fpe_drift}; diffusion coefficient {self.fpe_diffusion/2}']
            if p==49 and k>0:
                curves.append(_curve(self.grid,self.fluxes[k-1][1:],'probability flux J'))
                notes.append('Boundary probability flux is zero.')
            return _panel('Density transport and probability conservation','state x','density / flux',curves,notes=notes,xlim=[-3.5,3.5])
        if p==59:
            return _panel('Joint SLAM is provided by the estimator','x [m]','y [m]',notes=['Unknown landmarks and pose are estimated together.'])
        return self._ou_panel(k,t)

    def _ou_panel(self,k,t):
        p=self.page
        ts=self.times[:k+1]
        if p in (50,51,52):
            curves=[_curve(ts,self.ou[:k+1,0,0],
                           'model displacement' if p in (50,51) else 'model state 1')]
            if self.ou.shape[-1]>1:
                curves.append(_curve(ts,self.ou[:k+1,0,1],
                                     'model velocity' if p in (50,51) else 'model state 2'))
            return _panel('Linear stochastic reference process','time [s]','model state',curves,
                          notes=['URDF motion uses force/torque actuation.', 'Reduced-model paths are not measured rigid-body states.'],xlim=[0,self.duration])
        if p in (53,58,60):
            return _panel('OU moments: numerical ensemble and theory','time [s]','mean / variance',[
                _curve(ts,self.ou[:k+1,:,0].mean(axis=1),'mathematical ensemble mean'),
                _curve(ts,self.ou_mean[:k+1,0],'analytical mean'),
                _curve(ts,self.ou[:k+1,:,0].var(axis=1),'mathematical ensemble variance'),
                _curve(ts,self.ou_cov[:k+1,0,0],'analytical variance'),
                _curve(ts,np.full(k+1,self.stationary[0,0]),'stationary variance')],
                notes=[f'Lyapunov residual={self._metrics["lyapunov_residual_norm"]:.1e}'],xlim=[0,self.duration])
        if p==55:
            curves=[_curve(ts,self.covariance_integral[:k+1,i,i],f'noise-integral variance, state {i+1}') for i in range(len(self.gamma))]
            curves += [_curve(ts,np.full(k+1,self.stationary[i,i]),f'stationary variance, state {i+1}') for i in range(len(self.gamma))]
            return _panel('Covariance integral converges','integration horizon [s]','covariance',curves,
                          notes=[f'quadrature vs Lyapunov error={self._metrics["quadrature_lyapunov_error"]:.1e}'],xlim=[0,self.duration])
        if p==56:
            eigenvalues,u=np.linalg.eig(self.gamma)
            # Complex conjugate modes are legitimate for a real nonnormal drift.
            modes=np.exp(-eigenvalues[:,None]*self.times)
            return _panel('Decay modes of a nonsymmetric drift','time [s]','real / imaginary mode',[
                _curve(ts,modes[i,:k+1].real,f'Mode {i+1}, real part') for i in range(len(eigenvalues))]+
                [_curve(ts,modes[0,:k+1].imag,'Mode 1, imaginary part')],
                notes=[f'Biorthogonality residual {self._metrics["eigenvector_biorthogonality_residual"]:.1e}',
                       'Left eigenvectors form the inverse right-eigenvector matrix.'],xlim=[0,self.duration])
        if p==61:
            stable,unstable=self.stability_curves
            return _panel('Stable and unstable OU drift','time [s]','variance',[
                _curve(ts,stable[:k+1],'stable scalar restoring drift'),
                _curve(ts,unstable[:k+1],'unstable scalar drift')],
                notes=['Independent covariances add; correlated ones need cross terms.',
                       'Increment variance equals time step; FPE halves noise covariance.',
                       'Stable drift is required for stationarity here.'],xlim=[0,self.duration])
        x=np.linspace(-1.4,1.4,150)
        centers,hist=_hist(self.ou[k,:,0],32,(-1.4,1.4))
        title='Spectral, vectorized and Lyapunov covariance' if p==57 else 'Stationary Gaussian from Lyapunov covariance'
        return _panel(title,'first state coordinate','marginal density',[
            _curve(x,gaussian(x,0,self.stationary[0,0]),'stationary Gaussian'),
            _curve(x,gaussian(x,self.ou_mean[k,0],self.ou_cov[k,0,0]),'finite-time analytical density'),
            _curve(centers,hist,'mathematical ensemble')],
            notes=[f'Lyapunov residual={self._metrics["lyapunov_residual_norm"]:.1e}',
                   f'spectral residual={self._metrics["spectral_covariance_residual"]:.1e}'],xlim=[-1.4,1.4])

    def archive(self):
        """Compact numeric source data for replay and independent analysis.

        Keys are unprefixed because the runner adds ``model_`` beside the
        measured physics arrays. Dynamic ensembles retain the first 64 paths;
        their mean and variance use the full ensemble. Static sampling draws
        are small enough to retain in full. All arrays can be loaded with
        ``numpy.load(..., allow_pickle=False)``.
        """
        archive = {
            'times': self.times.copy(),
            'reference_paths': self.paths.copy(),
            'external_forces': self.forces.copy(),
            'heading': self.heading.copy(),
            'seed': np.asarray(self.seed, dtype=np.int64),
            'pdf_page': np.asarray(self.page, dtype=np.int64),
            'ensemble_size': np.asarray(self.ensemble_size, dtype=np.int64),
        }

        def add(name, value):
            archive[name] = np.asarray(value).copy()

        def ensemble(name, values):
            values = np.asarray(values)
            add(name + '_paths', values[:, :64])
            add(name + '_mean', values.mean(axis=1))
            add(name + '_variance', values.var(axis=1))

        if self.page <= 29:
            add('angular_samples', self.angular_samples)
            ensemble('angular', self.angle_paths)
            add('angular_mean_cosine', np.cos(self.angle_paths).mean(axis=1))
            add('angular_mean_sine', np.sin(self.angle_paths).mean(axis=1))
            add('circle_parameters_mu_variance_kappa', [self.mu, self.variance, self.kappa])
        elif self.page <= 37:
            add('sampling_draws', self.samples)
            for attribute in ('uniform', 'values', 'sphere_a', 'sphere_b', 'sphere_c',
                              'mu_vec', 'factor', 'covariance', 'dimensions',
                              'expected_acceptance', 'measured_acceptance'):
                if hasattr(self, attribute):
                    add(attribute, getattr(self, attribute))
            if hasattr(self, 'rate'):
                add('exponential_rate', self.rate)
        elif self.page <= 42:
            add('walk_times', self.walk_times)
            ensemble('walk', self.walk)
            add('walk_lattice', np.arange(-self.walk_steps, self.walk_steps + 1))
            # Equal-sized rows avoid an object array for the growing lattice.
            add('walk_pmf', np.stack([
                np.pad(pmf, (self.walk_steps - n, self.walk_steps - n))
                for n, pmf in enumerate(self.convolutions)
            ]))
            if hasattr(self, 'clt'):
                ensemble('clt', self.clt)
                add('clt_terminal_samples', self.clt[-1])
            if hasattr(self, 'diffusion'):
                add('walk_diffusion_D_h_dt', [self.diffusion, self.step_length, self.walk_dt])
        elif self.page <= 47:
            ensemble('wiener', self.wiener)
            add('wiener_increments', self.dw[:, :64])
            if hasattr(self, 'gbm'):
                ensemble('gbm_em', self.gbm)
                ensemble('gbm_exact', self.gbm_exact)
                add('gbm_drift_noise', [self.drift, self.noise])
            if hasattr(self, 'ou_scalar'):
                ensemble('ou_scalar', self.ou_scalar)
                add('ou_scalar_analytic_mean', self.scalar_mean)
                add('ou_scalar_analytic_variance', self.scalar_var)
                add('ou_scalar_rate_noise', [self.scalar_rate, self.scalar_noise])
        elif self.page <= 49:
            add('fpe_grid', self.grid)
            add('fpe_density', self.fpe)
            add('fpe_face_flux', self.fluxes)
            add('fpe_analytic_mean', self.fpe_mean)
            add('fpe_analytic_variance', self.fpe_var)
            add('fpe_drift_diffusion_B', [self.fpe_drift, self.fpe_diffusion])
            ensemble('fpe_sde', self.fpe_samples)
        elif hasattr(self, 'ou'):
            ensemble('ou', self.ou)
            centered = self.ou - self.ou.mean(axis=1)[:, None, :]
            add('ou_empirical_covariance',
                np.einsum('tni,tnj->tij', centered, centered) / self.ensemble_size)
            for key, attribute in (
                ('ou_drift_gamma', 'gamma'), ('ou_noise_C', 'c'),
                ('ou_diffusion_B', 'b'), ('ou_initial_mean', 'initial_mean'),
                ('ou_initial_covariance', 'initial_cov'),
                ('ou_analytic_mean', 'ou_mean'), ('ou_analytic_covariance', 'ou_cov'),
                ('ou_stationary_covariance', 'stationary'),
                ('ou_finite_covariance_integral', 'covariance_integral'),
            ):
                add(key, getattr(self, attribute))
            if hasattr(self, 'stability_curves'):
                add('stable_scalar_variance', self.stability_curves[0])
                add('unstable_scalar_variance', self.stability_curves[1])
        return archive

    def metrics(self):
        """Return JSON-safe numerical results; physics metrics belong to runner."""
        result=dict(self._metrics)
        result['checks']=dict(self._metrics['checks'])
        result['all_numerical_checks_passed']=all(result['checks'].values())
        return result
