import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from lecture6_sim.envs import RobotEnv, legacy_gym
from lecture6_sim.physics import World


def test_drone_needs_thrust_to_fly():
    with World("drone") as world:
        start = world.state()["position"][2]
        for _ in range(100):
            world.step_actuators(np.zeros(4))
        assert world.state()["position"][2] < start - .4
        world.reset()
        for _ in range(60):
            world.advance([.3, 0, 1.])
        state = world.state()
        assert state["position"][0] > .15
        assert abs(state["position"][2] - 1) < .12
        assert world.physics_steps == 480


def test_wheels_and_ground_contacts_move_husky():
    with World("mobile") as world:
        for _ in range(480):
            world.step_actuators(np.ones(4) * 2)
        assert world.state()["position"][0] > .3
        assert len(world.p.getContactPoints(bodyA=world.robot_id, bodyB=world.floor_id,
                                            physicsClientId=world.client_id)) > 0
        image = world.render(160, 120)
        assert image.shape == (120, 160, 3)
        assert image.std() > 10


@pytest.mark.parametrize("kind", ["drone", "mobile"])
def test_gymnasium_contract(kind):
    env = RobotEnv(kind, render_mode="rgb_array", horizon=.1)
    try:
        check_env(env, skip_render_check=False)
        env.reset(seed=123)
        action = np.full(4, env.world.mass * 9.81 / 4) if kind == "drone" else np.zeros(4)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
        assert truncated
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_legacy_gym_adapter():
    env = legacy_gym("drone")
    try:
        obs, _ = env.reset(seed=2)
        assert env.observation_space.contains(obs)
        assert len(env.step(env.action_space.sample())) == 5
    finally:
        env.close()
