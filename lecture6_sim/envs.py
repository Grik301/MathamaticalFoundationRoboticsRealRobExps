"""Gymnasium environments with direct physical actuator actions.

Crazyflie actions are the four rotor thrust requests in newtons. Husky actions
are FL, FR, RL, RR wheel speeds in radians per second, with 12 Nm motor limits.
The motor and rigid-body dynamics run inside Bullet, eight steps per action.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .physics import World


class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, kind="drone", render_mode=None, horizon=6.):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Invalid render_mode")
        if not 0 < horizon <= 30:
            raise ValueError("horizon must be in (0, 30]")
        self.kind, self.render_mode, self.horizon = kind, render_mode, horizon
        self.world = World(kind, gui=render_mode == "human")
        self.action_space = (spaces.Box(0, self.world.max_rotor_force, (4,), np.float32)
                             if kind == "drone" else spaces.Box(-9, 9, (4,), np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), np.float32)
        self.target = np.array([0, 0, 1.] if kind == "drone" else [1, 0, 0.])

    def _observation(self):
        s = self.world.state()
        return np.r_[s["position"], s["velocity"], s["quaternion"], s["omega"], self.world.t].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options
        self.world.reset(options.get("position"), options.get("yaw", 0.))
        return self._observation(), {"backend": "pybullet"}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (4,) or not np.isfinite(a).all():
            raise ValueError("Expected four finite actuator requests")
        a = np.clip(a, self.action_space.low, self.action_space.high)
        for _ in range(8):
            self.world.step_actuators(a)
        obs = self._observation()
        distance = np.linalg.norm(obs[:3] - self.target)
        terminated = bool((self.kind == "drone" and obs[2] < .03) or np.linalg.norm(obs[:3]) > 20)
        truncated = bool(self.world.t >= self.horizon - 1e-9)
        return obs, -float(distance), terminated, truncated, {"backend": "pybullet", "time": self.world.t}

    def render(self):
        if self.render_mode == "rgb_array":
            return self.world.render()
        return None

    def close(self):
        self.world.close()


def legacy_gym(kind="drone", **kwargs):
    """A Gym 0.26 adapter; uses Gym's real Env/Box classes and 5-value step API."""
    import gym as old_gym
    inner = RobotEnv(kind, **kwargs)

    class Adapter(old_gym.Env):
        metadata = inner.metadata

        def __init__(self):
            self.action_space = old_gym.spaces.Box(inner.action_space.low, inner.action_space.high, dtype=np.float32)
            self.observation_space = old_gym.spaces.Box(-np.inf, np.inf, (14,), np.float32)
            self.render_mode = inner.render_mode

        def reset(self, *, seed=None, options=None):
            return inner.reset(seed=seed, options=options)

        def step(self, action):
            return inner.step(action)

        def render(self):
            return inner.render()

        def close(self):
            inner.close()

    return Adapter()


for identifier, kind in (("Lecture6-Crazyflie-v0", "drone"), ("Lecture6-Husky-v0", "mobile")):
    if identifier not in gym.registry:
        gym.register(identifier, entry_point="lecture6_sim.envs:RobotEnv", kwargs={"kind": kind})
