"""Coverage of the supplied lecture, indexed by one-based PDF page number.

The PDF's printed footer is one less than the PDF page number. Each page has
its own experiment, including the overview and end-of-lecture questions.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    page: int
    title: str
    equation: str
    interpretation: str
    category: str

    @property
    def printed_number(self):
        return self.page - 1


_ROWS = [
    (26, 'Probability beyond Euclidean space', 'theta = theta + 2 pi; theta in S1', 'A physical heading crosses the angular branch cut while its circular representation stays continuous.', 'circle'),
    (27, 'Circular statistics', 'mean = atan2(E sin(theta), E cos(theta)); R = |E exp(i theta)|', 'Headings around -pi and +pi require a circular mean; arithmetic averaging gives the wrong direction.', 'circle'),
    (28, 'Gaussians wrapped around the circle', 'rho_W = sum_k N(theta - 2 pi k; mu, sigma^2)', 'Shifted Gaussian copies and the Fourier series describe the same normalized angular density.', 'circle'),
    (29, 'Diffusion on the circle', 'df/dt = (kappa/2) d2f/dtheta2', 'Independent angular Wiener increments spread a wrapped density towards the uniform circle.', 'circle'),
    (30, 'Sampling from a probability density', 'inverse CDF | Jacobian transform | sample and reject', 'Three independently implemented samplers produce uniform sphere directions for robot references.', 'sampling'),
    (31, 'The inverse-CDF transform', 'U ~ Uniform(0,1); X = F^-1(U)', 'Transform uniform samples through an exponential inverse CDF and compare the empirical distribution.', 'sampling'),
    (32, 'Inverse-CDF example: sphere directions', 'theta = acos(1-2U); phi = 2 pi V', 'Uniform area on the sphere gives a uniform z coordinate and E[qq^T] = I/3.', 'sampling'),
    (33, 'The geometric transformation method', 'pY(y) = pX(psi^-1(y)) |det Dpsi^-1(y)|', 'An invertible affine transform changes sample geometry and density by the inverse determinant.', 'sampling'),
    (34, 'The Box-Muller transformation', 'Z = sqrt(-2 log U) [cos(2 pi V), sin(2 pi V)]', 'Transform two uniform draws into two independent standard normals.', 'sampling'),
    (35, 'Sampling a multivariate Gaussian', 'X = mu + LZ; LL^T = Sigma', 'A Cholesky factor gives correlated Gaussian position references with the prescribed covariance.', 'sampling'),
    (36, 'Sample and reject: sphere directions', 'q = v / ||v||; accept iff 0 < ||v|| <= 1', 'Reject cube samples outside the unit ball before normalizing the accepted direction.', 'sampling'),
    (37, 'Rejection efficiency and dimension', 'P(accept) = pi^(d/2) / (2^d Gamma(d/2+1))', 'Measured acceptance rates match volume ratios and decrease rapidly with dimension.', 'sampling'),
    (38, 'The discrete random walker', 'K(n+1) = K(n) + xi; P(xi = +/-1) = 1/2', 'Independent fair steps drive a stochastic position reference and a numerical ensemble.', 'walk'),
    (39, 'A walk update is discrete convolution', 'p(n+1,k) = [p(n,k-1) + p(n,k+1)] / 2', 'Repeated convolution exactly agrees with the binomial position distribution.', 'walk'),
    (40, 'The binomial solution and parity', 'p(n,k) = 2^-n choose(n,(n+k)/2)', 'Only lattice sites of the correct parity are reachable; the variance is n.', 'walk'),
    (41, 'The central limit theorem', 'sum_i(Xi-mu) / (sigma sqrt(n)) -> N(0,1)', 'Centered, rescaled sums of independent uniform draws converge towards a standard Gaussian.', 'walk'),
    (42, 'The diffusion limit of a random walk', 'h^2 / dt = D; df/dt = (D/2) d2f/dx2', 'The spatial step shrinks with sqrt(dt), giving variance D t and the correct diffusion coefficient.', 'walk'),
    (43, 'Wiener increments and white noise', 'dW = sqrt(dt) Z; Var(dW) = dt', 'Independent Gaussian increments have variance dt; finite-step white-noise force scales as 1/sqrt(dt).', 'sde'),
    (44, 'Sample paths of a Wiener process', 'W(k+1) = W(k) + sqrt(dt) Zk', 'A slice across independent sample paths has the Gaussian distribution N(0,t).', 'sde'),
    (45, 'Stochastic differential equations', 'dX = h(X,t)dt + H(X,t)dW  [Ito]', 'A multiplicative-noise SDE is integrated using left-endpoint coefficients and checked against its exact law.', 'sde'),
    (46, 'Euler-Maruyama simulation', 'X(k+1) = Xk + h(Xk)dt + H(Xk)sqrt(dt)Zk', 'Coarse and fine Euler-Maruyama solutions share Wiener increments to measure strong time-step error.', 'sde'),
    (47, 'Trajectories and an evolving density', 'X_t samples <-> f(x,t)', 'An independent mathematical ensemble agrees with the evolving Ornstein-Uhlenbeck marginal density.', 'sde'),
    (48, 'The Fokker-Planck equation', 'df/dt = -div(hf) + (1/2) d_ij(B_ij f); B=HH^T', 'A conservative finite-volume solution is compared to an SDE ensemble and the analytic Gaussian.', 'fpe'),
    (49, 'Transport, diffusion and conservation', 'df/dt + div J = 0; J = h f - (B/2) grad f', 'Zero-flux finite-volume boundaries conserve probability while drift transports and noise spreads it.', 'fpe'),
    (50, 'A mechanical system with random forcing', 'm xddot + c xdot + k x = F(t)', 'A spring-damper reduced model supplies a stochastic reference; an external gust also acts on the rigid body.', 'ou'),
    (51, 'First-order stochastic state-space model', 'd[x,v] = [v,-(k/m)x-(c/m)v]dt + [0,q/m]dW', 'The same randomly forced oscillator is represented by a two-state linear stochastic system.', 'ou'),
    (52, 'The Ornstein-Uhlenbeck process', 'dX = -Gamma X dt + C dW; B = C C^T', 'Linear drift and additive diffusion admit an exact finite-step Gaussian sampler.', 'ou'),
    (53, 'Ornstein-Uhlenbeck mean and covariance', 'mu_dot=-Gamma mu; Sigma_dot=-Gamma Sigma-Sigma Gamma^T+B', 'Ensemble means and covariances are compared with matrix-exponential moment evolution.', 'ou'),
    (54, 'Stationary Gaussian and Lyapunov equation', 'Gamma Sigma_inf + Sigma_inf Gamma^T = B', 'The stationary covariance solves a continuous Lyapunov equation and is positive definite.', 'ou'),
    (55, 'The stationary covariance as an integral', 'Sigma_inf = integral_0^inf exp(-Gamma t) B exp(-Gamma^T t) dt', 'Numerical quadrature approaches the Lyapunov solution; stable eigenvalues ensure convergence.', 'ou'),
    (56, 'Right and left eigenvectors', 'Gamma = U Lambda V^T; V^T U = I', 'Biorthogonal eigenvectors reconstruct a nonsymmetric stable drift matrix.', 'ou'),
    (57, 'The spectral covariance solution', 'Sigma_inf = sum_ij u_i u_j^T (v_i^T B v_j)/(lambda_i+lambda_j)', 'The spectral formula, a Kronecker solve, quadrature and the Lyapunov equation agree.', 'ou'),
    (58, 'A scalar Ornstein-Uhlenbeck example', 'sigma^2(t) = exp(-2 gamma t)sigma0^2 + c^2(1-exp(-2 gamma t))/(2 gamma)', 'Exact scalar OU paths approach stationary variance c^2/(2 gamma).', 'ou'),
    (59, 'A robotics example: SLAM', 'p(x,y,theta,map | measurements)', 'Joint pose and unknown-landmark estimation uses uncertain odometry and range-bearing observations.', 'slam'),
    (60, 'Key relationships', 'convolution -> Gaussian sampling -> SDE -> FPE -> Lyapunov', 'A compact synthesis compares independently summed noise, OU trajectories and predicted covariance.', 'summary'),
    (61, 'Questions: quantitative checks', 'independence | sqrt(dt) | forward equation | stable drift', 'Four numerical checks answer the lecture questions, including covariance dependence and OU stability.', 'summary'),
]

TOPICS = {row[0]: Topic(*row) for row in _ROWS}


def get_topic(page: int) -> Topic:
    """Return metadata for one-based PDF pages 26 through 61."""
    if page not in TOPICS:
        raise ValueError('Expected a PDF page in 26..61')
    return TOPICS[page]
