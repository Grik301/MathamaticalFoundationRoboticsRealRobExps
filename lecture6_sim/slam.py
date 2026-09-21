"""Joint EKF-SLAM driven by synthetic sensors on moving PyBullet robots.

The estimator has no simulator handle and no access to ground-truth map or pose.
It starts with an empty map and a known launch-frame origin; measurements create
landmarks, including their correlations with the entire existing joint state.
Landmark identifiers represent decoded fiducial tags, not known coordinates.

Both robots estimate planar (x, y, yaw) and unknown landmark (x, y).  Crazyflie
flies at a nominal one-metre altitude: this is explicitly a planar SLAM slice,
not a claim of visual 3-D SLAM. Range/bearing and relative body odometry are
simulated noisy sensor measurements, not images processed by a vision frontend.

Reference: Durrant-Whyte and Bailey, "Simultaneous Localisation and Mapping:
Part I The Essential Algorithms", IEEE Robotics & Automation Magazine, 2006.
https://www-personal.acfr.usyd.edu.au/tbailey/publications/slamtutorial1.htm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SLAM_LANDMARKS = np.array(
    [(-1.8, -0.6), (0.0, -1.3), (1.8, -0.5), (2.0, 1.4),
     (1.3, 2.9), (-0.5, 3.0), (-1.9, 2.0)], dtype=float
)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Canonical circular residual, including across the +/-pi branch cut."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def motion(pose: np.ndarray, body_delta: np.ndarray) -> np.ndarray:
    """Compose a measured body displacement with a planar pose."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([pose[0] + c * body_delta[0] - s * body_delta[1],
                     pose[1] + s * body_delta[0] + c * body_delta[1],
                     wrap_angle(pose[2] + body_delta[2])])


def covariance_ellipse(mean: np.ndarray, covariance: np.ndarray,
                       probability: float = 0.95) -> np.ndarray:
    """A 2-D Gaussian confidence contour, not a one-dimensional sigma interval."""
    values, vectors = np.linalg.eigh(np.asarray(covariance))
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    # Chi-square(2) has CDF 1-exp(-x/2), so its quantile is exact here.
    radius = np.sqrt(-2.0 * np.log1p(-probability))
    circle = np.stack([np.cos(theta), np.sin(theta)])
    return np.asarray(mean)[:2] + (vectors @ np.diag(np.sqrt(np.maximum(values, 0))) @ circle).T * radius


@dataclass
class ObservationResult:
    initialized: bool
    accepted: bool
    innovation_squared: float = 0.0
    position_correction: float = 0.0


class EKFSLAM:
    """An initially empty, dynamically augmented joint Gaussian SLAM filter.

    Inputs are only local odometry increments/covariance and tag range/bearing
    measurements/covariance. The full state is [x,y,yaw,l0x,l0y,...]. The initial
    pose defines the map gauge; no global position sensor is used after launch.
    """

    def __init__(self, initial_pose=None, initial_covariance=None):
        self.mean = np.array(initial_pose if initial_pose is not None else
                             [0.0, 0.0, 0.0], dtype=float)
        self.covariance = np.array(
            initial_covariance if initial_covariance is not None else
            np.diag([0.002**2, 0.002**2, np.deg2rad(0.15)**2]), dtype=float
        )
        self.landmark_indices: dict[int, int] = {}
        self.initializations = 0
        self.updates = 0
        self.rejected_updates = 0

    def predict(self, body_delta: np.ndarray, process_covariance: np.ndarray):
        """Predict robot and robot-map cross covariance from relative odometry."""
        u = np.asarray(body_delta, dtype=float)
        q = np.asarray(process_covariance, dtype=float)
        yaw = self.mean[2]
        c, s = np.cos(yaw), np.sin(yaw)
        f = np.eye(len(self.mean))
        f[0, 2] = -s * u[0] - c * u[1]
        f[1, 2] = c * u[0] - s * u[1]
        g = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.mean[:3] = motion(self.mean[:3], u)
        self.covariance = f @ self.covariance @ f.T
        self.covariance[:3, :3] += g @ q @ g.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def _initialize(self, identifier: int, z: np.ndarray, r: np.ndarray):
        distance, bearing = z
        theta = self.mean[2] + bearing
        c, s = np.cos(theta), np.sin(theta)
        landmark = self.mean[:2] + distance * np.array([c, s])
        # Inverse-observation Jacobians with respect to robot and measurement.
        jx = np.array([[1.0, 0.0, -distance * s],
                       [0.0, 1.0, distance * c]])
        jz = np.array([[c, -distance * s], [s, distance * c]])
        cross = jx @ self.covariance[:3, :]
        pmm = jx @ self.covariance[:3, :3] @ jx.T + jz @ r @ jz.T
        n = len(self.mean)
        augmented = np.empty((n + 2, n + 2))
        augmented[:n, :n] = self.covariance
        augmented[n:, :n] = cross
        augmented[:n, n:] = cross.T
        augmented[n:, n:] = pmm
        self.landmark_indices[int(identifier)] = n
        self.mean = np.concatenate([self.mean, landmark])
        self.covariance = 0.5 * (augmented + augmented.T)
        self.initializations += 1

    def observe(self, identifier: int, measurement: np.ndarray,
                measurement_covariance: np.ndarray) -> ObservationResult:
        z = np.asarray(measurement, dtype=float)
        r = np.asarray(measurement_covariance, dtype=float)
        if z.shape != (2,) or z[0] <= 0 or not np.all(np.isfinite(z)):
            raise ValueError("An observation needs a positive range and finite bearing")
        if identifier not in self.landmark_indices:
            self._initialize(identifier, z, r)
            # Do not use the initialization measurement again as an update.
            return ObservationResult(initialized=True, accepted=True)
        index = self.landmark_indices[identifier]
        dx, dy = self.mean[index:index + 2] - self.mean[:2]
        squared_distance = dx * dx + dy * dy
        if squared_distance < 1e-10:
            self.rejected_updates += 1
            return ObservationResult(False, False)
        distance = np.sqrt(squared_distance)
        expected = np.array([distance, wrap_angle(np.arctan2(dy, dx) - self.mean[2])])
        innovation = z - expected
        innovation[1] = wrap_angle(innovation[1])
        h = np.zeros((2, len(self.mean)))
        h[:, :3] = [[-dx / distance, -dy / distance, 0.0],
                    [dy / squared_distance, -dx / squared_distance, -1.0]]
        h[:, index:index + 2] = [[dx / distance, dy / distance],
                                [-dy / squared_distance, dx / squared_distance]]
        innovation_covariance = h @ self.covariance @ h.T + r
        nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        # 99.99%-level chi-square(2) gate. IDs are known tags, not nearest-neighbor.
        if nis > 18.420681:
            self.rejected_updates += 1
            return ObservationResult(False, False, nis)
        gain = np.linalg.solve(innovation_covariance, h @ self.covariance).T
        correction = gain @ innovation
        self.mean += correction
        self.mean[2] = wrap_angle(self.mean[2])
        # Joseph form keeps all map cross correlations and numerical symmetry.
        ikh = np.eye(len(self.mean)) - gain @ h
        self.covariance = ikh @ self.covariance @ ikh.T + gain @ r @ gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.updates += 1
        return ObservationResult(False, True, nis, float(np.linalg.norm(correction[:2])))

    def map_estimate(self) -> dict[int, np.ndarray]:
        return {identifier: self.mean[index:index + 2].copy()
                for identifier, index in self.landmark_indices.items()}


class SLAMExperiment:
    """Sensor frontend, loop command and evaluator around the independent filter.

    Relative odometry emulates noisy onboard incremental translation/gyro,
    including a modest calibration bias. It is synthesized from simulator pose
    differences; it is not claimed to be image-derived optical flow or actual
    hardware encoder data. The estimator never receives the absolute pose.
    Fiducial observations have finite range, finite FOV, dropout and PyBullet
    ray-test occlusion. The ground-truth arrays below are evaluation only.
    """

    def __init__(self, world, seed: int, duration: float = 12.0, dt: float = 1 / 30):
        if not 0 < dt <= duration <= 30:
            raise ValueError("Require 0 < dt <= duration <= 30 seconds")
        self.world = world
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.duration = float(duration)
        self.dt = float(dt)
        self.kind = str(world.kind)
        self.is_drone = self.kind in {"drone", "crazyflie", "cf2x", "cf2p"}
        self.filter = EKFSLAM()
        self.dead_reckoning = np.zeros(3)
        self.previous_sensor_pose: np.ndarray | None = None
        self.previous_time: float | None = None
        self.next_observation_time = 0.0
        self.observation_period = 0.1
        self.range_limit = 3.6
        self.half_fov = np.deg2rad(70.0)
        self.measurement_covariance = np.diag([0.035**2, np.deg2rad(1.2)**2])
        self.last_seen: dict[int, float] = {}
        self.revisit_events: list[dict[str, Any]] = []
        self.sensor_observations: list[list[float]] = []
        self.odometry_log: list[list[float]] = []
        self.history: dict[str, list] = {key: [] for key in
            ["time", "truth", "estimate", "dead_reckoning", "pose_covariance",
             "joint_mean", "joint_covariance", "landmark_count", "observations"]}
        self.visible_identifiers: list[int] = []
        self.occlusion_checks = 0
        self.occluded_observations = 0
        self.sensor_dropouts = 0
        self.last_revisit_correction = 0.0

    def _pose_sensor_source(self) -> tuple[np.ndarray, dict]:
        state = self.world.state()
        position = np.asarray(state["position"])
        return np.array([position[0], position[1], state["yaw"]]), state

    def _generate_odometry(self, true_pose: np.ndarray, t: float):
        # This method is the sensor simulator. Only its noisy local output enters
        # EKFSLAM.predict. The scheduled high-level path is also independent of
        # this pose. World separately uses ideal simulator state feedback in its
        # low-level motor/attitude controller; navigation is not SLAM-controlled.
        if self.previous_sensor_pose is not None:
            elapsed = max(t - float(self.previous_time), 1e-8)
            delta = true_pose[:2] - self.previous_sensor_pose[:2]
            yaw0 = self.previous_sensor_pose[2]
            c, s = np.cos(yaw0), np.sin(yaw0)
            ideal = np.array([c * delta[0] + s * delta[1],
                              -s * delta[0] + c * delta[1],
                              wrap_angle(true_pose[2] - yaw0)])
            std = np.array([0.014, 0.014, 0.018]) * np.sqrt(elapsed)
            measured = ideal.copy()
            measured[:2] *= 1.045  # unmodelled incremental-odometry scale error
            measured[2] += 0.020 * elapsed  # gyro bias in radians per second
            measured += self.rng.normal(size=3) * std
            # Process uncertainty includes unmodelled slip/calibration variability.
            process_covariance = np.diag((np.array([0.025, 0.025, 0.030])**2) * elapsed)
            self.filter.predict(measured, process_covariance)
            self.dead_reckoning = motion(self.dead_reckoning, measured)
            self.odometry_log.append([t, *measured])
        self.previous_sensor_pose = true_pose.copy()
        self.previous_time = t

    def _line_of_sight(self, state: dict, landmark: np.ndarray, identifier: int) -> bool:
        position = np.asarray(state["position"], dtype=float)
        sensor_z = float(position[2] + 0.065) if self.is_drone else 0.72
        ray_start = [float(position[0]), float(position[1]), sensor_z]
        ray_end = [float(landmark[0]), float(landmark[1]), min(sensor_z, 1.25)]
        self.occlusion_checks += 1
        hit = self.world.p.rayTest(ray_start, ray_end,
                                   physicsClientId=self.world.client_id)[0][0]
        landmark_bodies = getattr(self.world, "landmark_ids", [])
        target_body = landmark_bodies[identifier] if identifier < len(landmark_bodies) else None
        visible = hit == -1 or hit == target_body
        if not visible:
            self.occluded_observations += 1
        return visible

    def _generate_observations(self, true_pose: np.ndarray, state: dict, t: float):
        self.visible_identifiers = []
        if t + 1e-9 < self.next_observation_time:
            return
        self.next_observation_time = t + self.observation_period
        for identifier, landmark in enumerate(np.asarray(self.world.landmarks)):
            delta = landmark - true_pose[:2]
            distance = np.linalg.norm(delta)
            bearing = wrap_angle(np.arctan2(delta[1], delta[0]) - true_pose[2])
            if distance > self.range_limit or distance < 0.15 or abs(bearing) > self.half_fov:
                continue
            if not self._line_of_sight(state, landmark, identifier):
                continue
            if self.rng.uniform() < 0.04:
                self.sensor_dropouts += 1
                continue
            z = np.array([distance, bearing]) + self.rng.multivariate_normal(
                np.zeros(2), self.measurement_covariance)
            z[0] = max(0.01, z[0])
            z[1] = wrap_angle(z[1])
            previous_seen = self.last_seen.get(identifier)
            before_pose = self.filter.mean[:3].copy()
            before_covariance_trace = float(np.trace(self.filter.covariance[:3, :3]))
            result = self.filter.observe(identifier, z, self.measurement_covariance)
            self.sensor_observations.append([t, identifier, *z, float(result.accepted)])
            if result.accepted:
                self.visible_identifiers.append(identifier)
                self.last_seen[identifier] = t
            if result.accepted and not result.initialized and previous_seen is not None and t - previous_seen > 1.0:
                event = {"time_s": t, "landmark_id": identifier,
                         "absence_s": t - previous_seen,
                         "position_correction_m": result.position_correction,
                         "pose_covariance_trace_before": before_covariance_trace,
                         "pose_covariance_trace_after": float(np.trace(self.filter.covariance[:3, :3])),
                         "position_error_before_m": float(np.linalg.norm(before_pose[:2] - true_pose[:2])),
                         "position_error_after_m": float(np.linalg.norm(self.filter.mean[:2] - true_pose[:2]))}
                self.revisit_events.append(event)
                self.last_revisit_correction = result.position_correction

    def command(self, t: float) -> tuple[np.ndarray, float]:
        # A scheduled loop, independent of simulator truth and estimator/map.
        # The physics world's motor/attitude controller follows this reference.
        radius = 0.80
        travel_time = max(self.duration - 2.0, 6.0)
        phase = np.clip((t + 0.22) / travel_time, 0.0, 1.0) * 2.0 * np.pi
        target = np.array([radius * np.sin(phase), radius * (1.0 - np.cos(phase)),
                           1.0 if self.is_drone else 0.0])
        return target, float(wrap_angle(phase))

    def step(self, t: float) -> dict[str, Any]:
        true_pose, state = self._pose_sensor_source()
        self._generate_odometry(true_pose, float(t))
        self._generate_observations(true_pose, state, float(t))
        self.history["time"].append(float(t))
        self.history["truth"].append(true_pose.copy())
        self.history["estimate"].append(self.filter.mean[:3].copy())
        self.history["dead_reckoning"].append(self.dead_reckoning.copy())
        self.history["pose_covariance"].append(self.filter.covariance[:3, :3].copy())
        self.history["joint_mean"].append(self.filter.mean.copy())
        self.history["joint_covariance"].append(self.filter.covariance.copy())
        self.history["landmark_count"].append(len(self.filter.landmark_indices))
        self.history["observations"].append(len(self.visible_identifiers))
        target, yaw = self.command(t)
        return {"target": target, "yaw": yaw, "force": np.zeros(3), "panel": self.panel()}

    def panel(self) -> dict[str, Any]:
        truth = np.asarray(self.history["truth"])
        estimate = np.asarray(self.history["estimate"])
        dead = np.asarray(self.history["dead_reckoning"])
        map_estimate = self.filter.map_estimate()
        landmarks = np.asarray(list(map_estimate.values())).reshape(-1, 2)
        covariances = [self.filter.covariance[i:i + 2, i:i + 2].tolist()
                       for i in self.filter.landmark_indices.values()]
        notes = ["Unknown map + joint pose/map covariance; decoded tag IDs",
                 "Synthetic odometry/gyro + range/bearing; FOV and ray occlusion",
                 f"Mapped {len(map_estimate)}/{len(self.world.landmarks)} | revisits {len(self.revisit_events)}",
                 f"SLAM error {np.linalg.norm(estimate[-1, :2] - truth[-1, :2]):.3f}m; "
                 f"odometry {np.linalg.norm(dead[-1, :2] - truth[-1, :2]):.3f}m"]
        if self.is_drone:
            notes.append("Crazyflie: planar (x,y,yaw) SLAM, altitude controlled at 1m")
        if self.revisit_events:
            notes.append(f"Last tag revisit corrected pose by {self.last_revisit_correction:.3f}m")
        return {"title": "SLAM: learn the map while estimating pose (slide 58)",
                "xlabel": "x (m)", "ylabel": "y (m)",
                "curves": [{"x": truth[:, 0].tolist(), "y": truth[:, 1].tolist(), "label": "Physics truth"},
                           {"x": dead[:, 0].tolist(), "y": dead[:, 1].tolist(), "label": "Odometry only"},
                           {"x": estimate[:, 0].tolist(), "y": estimate[:, 1].tolist(), "label": "Joint EKF-SLAM"}],
                "notes": notes,
                "xlim": [-2.3, 2.4], "ylim": [-1.7, 3.4],
                "landmark_estimates": landmarks.tolist(),
                "landmark_truth": np.asarray(self.world.landmarks).tolist(),
                "landmark_covariances": covariances,
                "pose_covariance": self.filter.covariance[:2, :2].tolist(),
                "pose_estimate": self.filter.mean[:3].tolist(),
                "confidence_probability": 0.95,
                "pose_confidence_ellipse": covariance_ellipse(
                    self.filter.mean[:2], self.filter.covariance[:2, :2]).tolist(),
                "landmark_confidence_ellipses": [covariance_ellipse(mean, covariance).tolist()
                                                  for mean, covariance in zip(landmarks, covariances)]}

    def metrics(self) -> dict[str, Any]:
        if not self.history["truth"]:
            return {"samples": 0}
        truth = np.asarray(self.history["truth"])
        estimate = np.asarray(self.history["estimate"])
        dead = np.asarray(self.history["dead_reckoning"])
        error = np.linalg.norm(estimate[:, :2] - truth[:, :2], axis=1)
        dead_error = np.linalg.norm(dead[:, :2] - truth[:, :2], axis=1)
        map_error = [np.linalg.norm(value - self.world.landmarks[identifier])
                     for identifier, value in self.filter.map_estimate().items()]
        travelled = float(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1).sum())
        return_distance = float(np.linalg.norm(truth[-1, :2] - truth[0, :2]))
        max_distance = float(np.linalg.norm(truth[:, :2] - truth[0, :2], axis=1).max())
        return {"samples": len(truth), "seed": self.seed,
                "simulation_duration_s": float(self.history["time"][-1]),
                "estimator": "joint EKF-SLAM with initially unknown XY landmarks",
                "state_dimension": len(self.filter.mean),
                "planar_slice": True, "map_initially_empty": True,
                "pose_anchor": "known launch origin (0,0,0), no absolute position updates",
                "navigation_uses_slam_estimate": False,
                "controller_feedback": "scheduled loop reference; ideal simulator state feedback for physical motor/attitude control",
                "sensor_model": "synthetic noisy relative body odometry/gyro; noisy range/bearing to decoded fiducial IDs",
                "sensor_range_m": self.range_limit, "sensor_fov_degrees": float(np.rad2deg(2 * self.half_fov)),
                "measurement_rate_hz": 1 / self.observation_period,
                "mapped_landmarks": len(map_error), "total_landmarks": len(self.world.landmarks),
                "position_rmse_m": float(np.sqrt(np.mean(error**2))),
                "dead_reckoning_position_rmse_m": float(np.sqrt(np.mean(dead_error**2))),
                "final_position_error_m": float(error[-1]),
                "final_dead_reckoning_error_m": float(dead_error[-1]),
                "heading_rmse_degrees": float(np.rad2deg(np.sqrt(np.mean(wrap_angle(estimate[:, 2] - truth[:, 2])**2)))),
                "map_rmse_m": float(np.sqrt(np.mean(np.square(map_error)))) if map_error else None,
                "travelled_distance_m": travelled, "return_to_start_distance_m": return_distance,
                "physical_loop_completed": bool(travelled > 3.0 and max_distance > 1.0 and return_distance < 0.4),
                "revisit_count": len(self.revisit_events), "revisit_events": self.revisit_events,
                "accepted_measurement_updates": self.filter.updates,
                "rejected_measurement_updates": self.filter.rejected_updates,
                "ray_occlusion_checks": self.occlusion_checks,
                "occluded_observations": self.occluded_observations,
                "random_sensor_dropouts": self.sensor_dropouts,
                "covariance_minimum_eigenvalue": float(np.linalg.eigvalsh(self.filter.covariance).min()),
                "covariance_symmetry_error": float(np.abs(self.filter.covariance - self.filter.covariance.T).max())}

    def archive(self) -> dict[str, np.ndarray]:
        """Return numeric arrays suitable for np.savez_compressed (no pickle).

        Dynamic joint histories are NaN-padded until a landmark is initialized;
        final_landmark_ids/final_landmark_state_indices specify the state order.
        """
        n, dimension = len(self.history["time"]), len(self.filter.mean)
        means = np.full((n, dimension), np.nan)
        covariances = np.full((n, dimension, dimension), np.nan)
        for i, (mean, covariance) in enumerate(zip(self.history["joint_mean"], self.history["joint_covariance"])):
            means[i, :len(mean)] = mean
            covariances[i, :len(mean), :len(mean)] = covariance
        output = {key: np.asarray(self.history[key]) for key in
                  ["time", "truth", "estimate", "dead_reckoning", "pose_covariance",
                   "landmark_count", "observations"]}
        output.update({"joint_mean": means, "joint_covariance": covariances,
                       "final_landmark_ids": np.asarray(list(self.filter.landmark_indices), dtype=int),
                       "final_landmark_state_indices": np.asarray(list(self.filter.landmark_indices.values()), dtype=int),
                       "true_landmarks": np.asarray(self.world.landmarks),
                       "sensor_range_bearing": np.asarray(self.sensor_observations).reshape(-1, 5),
                       "sensor_body_odometry": np.asarray(self.odometry_log).reshape(-1, 4)})
        return output
