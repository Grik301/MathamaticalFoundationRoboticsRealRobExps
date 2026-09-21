"""Estimator checks independent of PyBullet and its ground-truth sensor driver."""

import unittest
from types import SimpleNamespace

import numpy as np

from lecture6_sim.slam import EKFSLAM, SLAMExperiment, motion, wrap_angle


class JointSLAMTests(unittest.TestCase):
    def test_sensor_ray_rejects_physical_occluder(self):
        import pybullet as p

        client = p.connect(p.DIRECT)
        try:
            world = SimpleNamespace(kind="mobile", p=p, client_id=client,
                                    landmarks=np.array([[2.0, 0.0]]), landmark_ids=[])
            experiment = SLAMExperiment(world, seed=1)
            state = {"position": [0.0, 0.0, 0.0]}
            self.assertTrue(experiment._line_of_sight(state, world.landmarks[0], 0))
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.2, 0.5], physicsClientId=client)
            obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                         basePosition=[1.0, 0.0, 0.5], physicsClientId=client)
            self.assertFalse(experiment._line_of_sight(state, world.landmarks[0], 0))
            self.assertEqual(experiment.occluded_observations, 1)
            p.removeBody(obstacle, physicsClientId=client)
        finally:
            p.disconnect(client)

    def test_landmark_augmentation_preserves_cross_covariances(self):
        initial = np.diag([0.2, 0.3, 0.05])
        estimator = EKFSLAM(initial_covariance=initial)
        measurement_covariance = np.diag([0.01, 0.002])
        first = estimator.observe(91, [2.0, 0.0], measurement_covariance)
        self.assertTrue(first.initialized)
        np.testing.assert_allclose(estimator.mean, [0, 0, 0, 2, 0])
        np.testing.assert_allclose(estimator.covariance[:3, :3], initial)
        self.assertGreater(np.linalg.norm(estimator.covariance[3:, :3]), 0)
        estimator.observe(12, [2.0, np.pi / 2], measurement_covariance)
        self.assertGreater(np.linalg.norm(estimator.covariance[5:, 3:5]), 0)
        self.assertGreater(np.linalg.eigvalsh(estimator.covariance).min(), -1e-12)

    def test_bearing_wrap_and_outlier_gate(self):
        estimator = EKFSLAM()
        measurement_covariance = np.diag([0.04, 0.01])
        estimator.observe(0, [2.0, np.pi - 0.001], measurement_covariance)
        update = estimator.observe(0, [2.0, -np.pi + 0.001], measurement_covariance)
        self.assertTrue(update.accepted)
        self.assertLess(update.innovation_squared, 0.01)
        outlier = estimator.observe(0, [20.0, 0.0], measurement_covariance)
        self.assertFalse(outlier.accepted)

    def test_joint_filter_improves_biased_odometry_on_a_loop(self):
        rng = np.random.default_rng(10)
        estimator = EKFSLAM()
        dead = np.zeros(3)
        truth = np.zeros(3)
        landmarks = np.array([[2, -1], [3, 1], [0, 3], [-2, 1]])
        measurement_covariance = np.diag([0.025**2, 0.012**2])
        actual_increment = np.array([0.015, 0.0, 2 * np.pi / 400])
        errors, dead_errors = [], []
        for k in range(401):
            if k:
                truth = motion(truth, actual_increment)
                measured = actual_increment + np.array([0.0003, 0, 0.0005])
                measured += rng.normal(size=3) * np.array([0.001, 0.001, 0.001])
                estimator.predict(measured, np.diag([0.002**2] * 3))
                dead = motion(dead, measured)
            if k % 4 == 0:
                for tag, landmark in enumerate(landmarks):
                    dx, dy = landmark - truth[:2]
                    observation = [np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - truth[2])]
                    observation += rng.multivariate_normal(np.zeros(2), measurement_covariance)
                    estimator.observe(tag, observation, measurement_covariance)
            errors.append(np.linalg.norm(estimator.mean[:2] - truth[:2]))
            dead_errors.append(np.linalg.norm(dead[:2] - truth[:2]))
        self.assertEqual(len(estimator.mean), 11)
        self.assertEqual(estimator.initializations, 4)
        self.assertGreater(estimator.updates, 350)
        self.assertLess(np.mean(np.square(errors)), np.mean(np.square(dead_errors)) * 0.25)
        self.assertLess(errors[-1], 0.1)
        self.assertLess(np.abs(estimator.covariance - estimator.covariance.T).max(), 1e-12)
        self.assertGreater(np.linalg.eigvalsh(estimator.covariance).min(), -1e-12)


if __name__ == "__main__":
    unittest.main()
