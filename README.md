# MathamaticalFoundationRoboticsRealRobExps

Probability, stochastic dynamics, and SLAM experiments using PyBullet models of the **Bitcraze Crazyflie** drone and **Clearpath Husky** mobile robot. The project covers lecture pages **26–61**, with one experiment per robot and page: **72 experiments**.

Robots move through rotor forces or wheel motors in the physics simulator. These are simulations of real robot designs, not physical-hardware experiments or calibrated digital twins.

This repository contains source code, configuration, tests, and documentation. Robot models are downloaded during setup; equation artwork and experiment outputs are generated locally. PDFs, datasets, GIFs, videos, caches, and environments are excluded.

## Installation

Tested with **Python 3.13**. Equation generation also requires **Node.js 18 or newer**, npm, and a system Cairo library supported by CairoSVG. A desktop display is optional; recording works headlessly through PyBullet.

```bash
git clone https://github.com/Grik301/MathamaticalFoundationRoboticsRealRobExps.git
cd MathamaticalFoundationRoboticsRealRobExps

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download pinned upstream URDFs, meshes, and their licenses.
python -m lecture6_sim.download_assets

# Generate equation artwork from the project's LaTeX source.
npm ci --prefix .mathjax
python -m lecture6_sim.typeset
```

Setup needs internet access. Subsequent experiments use the downloaded models and generated artwork locally. See [robot models and assumptions](docs/ROBOT_MODELS.md) for provenance and required model preparation.

## Run experiments

With the virtual environment activated:

```bash
# Record all 72 experiments with real-time pacing.
python -m lecture6_sim

# Run SLAM on both robots.
python -m lecture6_sim --pages 59

# Show one drone experiment in a desktop PyBullet window.
python -m lecture6_sim --pages 46 --robot drone --gui

# Select pages and save to a separate output directory.
python -m lecture6_sim --pages 26-29,50-58 --robot mobile --output results_selection

# Continue a partially completed batch.
python -m lecture6_sim --resume

# Audit the complete default batch and generate its report.
python -m lecture6_sim.verify
```

Open the generated `results/index.html` for the GIF gallery and `results/SLIDE_COVERAGE.md` for the page-by-page mapping. The lecture PDF is not distributed. Page numbers refer to PDF pages; page 26 corresponds to printed footer 25 in the original lecture.

Ordinary experiments simulate **6 seconds**; SLAM simulates **18 seconds**. GIFs play at **10 frames per second and 1× speed**. A complete batch takes several minutes. The target is under 30 seconds of total processing per experiment, including initialization, rendering, and encoding; actual wall time depends on hardware and the rendering backend. JSON logs record measured timings, and the verifier checks the limit. The CLI caps simulation duration at 24 seconds to leave encoding margin. `--fast` disables wall-clock pacing for development.

Every GIF includes the PyBullet camera, physical trajectory, mathematical or SLAM plots, **typeset LaTeX equations**, and **explicit color legends** explaining the paths, curves, clouds, and markers. `--camera-width` controls the embedded camera resolution; the default is 360 pixels.

## Implemented topics

| PDF pages | Experiment family |
|---|---|
| 26–29 | Circular statistics, wrapped distributions, Fourier representation, circular diffusion |
| 30–37 | Inverse CDF, sphere sampling, Jacobians, Box–Muller, correlated Gaussians, rejection sampling |
| 38–42 | Random walks, convolution, binomial/parity laws, central limit theorem, diffusion scaling |
| 43–47 | Wiener processes, Itô SDEs, Euler–Maruyama convergence, evolving ensemble densities |
| 48–49 | Conservative finite-volume Fokker–Planck solver, probability flux, mass conservation |
| 50–58 | Stochastic oscillators, Ornstein–Uhlenbeck processes, moments, Lyapunov covariance, eigenvector and spectral representations |
| 59 | Joint pose-and-map EKF-SLAM with initially unknown landmark positions and revisits |
| 60–61 | Relationships between models, covariance assumptions, Wiener scaling, stable and unstable OU dynamics |

The numerical probability models are separate from the physical robot simulation. Their samples drive bounded position/heading references or external disturbances. The controller and rigid body introduce actuator limits, inertia, contact, and tracking error; the robot trajectory is not assumed to follow a reduced stochastic equation exactly. Numerical ensembles are arrays of model samples, not thousands of separately simulated robots. Mathematical checks and physical tracking statistics are recorded separately.

## Physics and SLAM

PyBullet advances at **240 Hz**, with controller feedback at every physics step and reference updates at 30 Hz. Crazyflie uses four rotor forces and reaction torques; Husky uses four force-limited wheel motors and simulated ground contact. Robot motion is actuator-driven after initialization. Rendering prefers EGL/OpenGL, with a TinyRenderer fallback. `LECTURE6_RENDERER=tiny` selects that fallback explicitly; `LP_NUM_THREADS` controls software-renderer worker count.

Both SLAM experiments estimate a joint Gaussian state containing planar robot pose and landmark coordinates. They start with an empty map, initialize landmarks with pose/map cross-covariances, wrap bearing residuals, and use Joseph-form covariance updates. Revisited landmarks correct the joint estimate.

Sensors are synthetic: noisy relative body odometry and gyro increments, plus range/bearing observations with a 3.6 m range, 140° field of view, 10 Hz observations, dropout, and PyBullet ray-test occlusion. Known tag IDs provide data association; landmark positions are unknown to the estimator. Ground truth generates sensor measurements and evaluation metrics but is not supplied as a filter correction.

Drone SLAM is **planar landmark EKF-SLAM at controlled altitude**, not full 3D visual-inertial SLAM. There is no camera-image feature extractor or occupancy-grid frontend. Physical controllers use simulated state feedback to follow a scheduled loop; the SLAM estimate does not control navigation.

## Outputs and reproducibility

The runner generates:

- `results/gifs/` and `results/previews/`: animations and gallery previews.
- `results/logs/*.json`: seeds, timings, physics statistics, numerical checks, and SLAM errors.
- `results/logs/*.csv`: time, physical state, reference commands, and applied disturbances in SI units.
- `results/logs/*.npz`: physical trajectories, numerical-model arrays, and joint SLAM state/covariance and sensor histories. Load with `numpy.load(path, allow_pickle=False)`.
- `results/manifest.json`, `results/verification.json`, and `results/REPORT.md`: experiment inventory and verification evidence.

Physical state vectors use `[x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]`, with XYZW quaternions and world-frame angular velocity. Rotor actions are thrusts in newtons; wheel actions are speeds in radians per second. SLAM histories use `NaN` padding before landmarks are discovered; mobile terminal heading can be unconstrained.

Equation source lives in `lecture6_sim/equations.py`. MathJax produces SVG glyph paths and CairoSVG produces the equation images before an experiment begins. To rebuild equations and update existing recordings without rerunning physics:

```bash
python -m lecture6_sim.typeset
python -m lecture6_sim.refresh_gifs --force
python -m lecture6_sim.verify
```

Presentation refresh requires the existing GIFs and logs. It verifies that camera pixels, playback durations, and raw CSV/NPZ files are preserved. Generated model assets, equation images, and results remain outside version control.

## Gymnasium and Gym

```python
import gymnasium as gym
import lecture6_sim.envs  # Registers both environments.

env = gym.make("Lecture6-Crazyflie-v0", render_mode="rgb_array", horizon=6)
obs, info = env.reset(seed=123)
hover = [0.027 * 9.81 / 4] * 4  # Four rotor thrusts, in N.
obs, reward, terminated, truncated, info = env.step(hover)
rgb = env.render()
env.close()
```

`Lecture6-Husky-v0` accepts four wheel-speed actions ordered front-left, front-right, rear-left, rear-right. Both environments expose actuator control through the five-value step API; the lecture runner supplies higher-level controllers. A Gym 0.26 adapter is available as `lecture6_sim.envs.legacy_gym`.

## Tests

After setup:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

Tests cover numerical invariants, physical actuation, Gym compatibility, and joint EKF-SLAM behavior. Disabling plugin autoload isolates the tests from unrelated globally installed pytest plugins. The separate GIF verifier checks complete recorded batches, including timing, equations, color legends, and numerical results.

## License

Project source code is licensed under the [MIT License](LICENSE), copyright 2026 Grik Tadevosyan (Grik301). Downloaded robot descriptions and meshes retain their upstream licenses; the downloader preserves their notices. See [robot model provenance](docs/ROBOT_MODELS.md). The lecture PDF and generated recordings are not included in this repository.
