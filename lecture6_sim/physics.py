"""Force-driven Crazyflie and contact-driven Husky in genuine PyBullet.

The deterministic 240 Hz solver supports headless camera recording and GUI
playback. ``advance`` controls real actuators and never resets a moving pose.
State positions use each URDF's root frame: Crazyflie base_link and Husky
base_footprint. SI units, XYZW quaternions, world-frame velocities throughout.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


ASSETS = Path(__file__).resolve().parent / "assets"


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class World:
    def __init__(self, kind, gui=False, dt=1 / 240, landmarks=None):
        if kind not in ("drone", "mobile"):
            raise ValueError("kind must be 'drone' or 'mobile'")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.kind, self.gui, self.dt = kind, bool(gui), float(dt)
        self.p = p
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("PyBullet connection failed")
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER
        self.renderer_name = "ER_BULLET_HARDWARE_OPENGL (GUI)" if gui else "ER_TINY_RENDERER"
        self.egl_plugin = None
        # Register the renderer before loading shapes, so all original meshes
        # are uploaded. EGL also works with Mesa software OpenGL in headless
        # containers; on this machine it is about 20x faster than TinyRenderer.
        if not gui and os.environ.get("LECTURE6_RENDERER", "egl") != "tiny":
            # Keep Mesa's software renderer from oversubscribing CPU cores;
            # callers can override this before launch on other machines.
            os.environ.setdefault("LP_NUM_THREADS", "4")
            egl = importlib.util.find_spec("eglRenderer")
            if egl is not None:
                plugin = p.loadPlugin(egl.origin, "_eglRendererPlugin", physicsClientId=self.client_id)
                if plugin >= 0:
                    self.egl_plugin = plugin
                    self.renderer = p.ER_BULLET_HARDWARE_OPENGL
                    self.renderer_name = "ER_BULLET_HARDWARE_OPENGL (EGL)"
        self.landmarks = np.empty((0, 2)) if landmarks is None else np.asarray(landmarks, float).reshape(-1, 2)
        self.landmark_ids = []
        self.camera_distance = .32 if kind == "drone" else 2.5
        self.camera_yaw = 40
        self.camera_pitch = -27
        self.urdf = ASSETS / "prepared" / ("cf2x.urdf" if kind == "drone" else "husky.urdf")
        if not self.urdf.is_file():
            raise FileNotFoundError(f"Missing robot assets: run python -m lecture6_sim.download_assets ({self.urdf})")
        if kind == "drone":
            tree = ET.parse(self.urdf).getroot()
            props = tree.find("properties")
            inertial = tree.find("link[@name='base_link']/inertial")
            self.mass = float(inertial.find("mass").get("value"))
            self.inertia = np.diag([float(inertial.find("inertia").get(k)) for k in ("ixx", "iyy", "izz")])
            self.kf, self.km = float(props.get("kf")), float(props.get("km"))
            self.max_rotor_force = float(props.get("thrust2weight")) * self.mass * 9.81 / 4
            self.rotor_xy = np.array([[.028, -.028], [-.028, -.028], [-.028, .028], [.028, .028]])
            self.rotor_signs = np.array([-1., 1., -1., 1.])
            self.mixer = np.vstack((np.ones(4), self.rotor_xy[:, 1], -self.rotor_xy[:, 0], self.rotor_signs * self.km / self.kf))
            self.inverse_mixer = np.linalg.inv(self.mixer)
        else:
            self.wheel_radius, self.track = .17775, .5708
        self.reset()

    def reset(self, position=None, yaw=0):
        """Initialize a fresh episode; no pose reset occurs in ``advance``."""
        cid = self.client_id
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(self.dt, physicsClientId=cid)
        p.setRealTimeSimulation(0, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=80, deterministicOverlappingPairs=1,
                                   physicsClientId=cid)
        floor_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
        floor_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[8, 8, .015],
                                          visualFramePosition=[0, 0, -.015],
                                          rgbaColor=[.84, .87, .90, 1], physicsClientId=cid)
        self.floor_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_shape,
                                         baseVisualShapeIndex=floor_visual,
                                         physicsClientId=cid)
        for value in range(-6, 7):
            for along_x in (False, True):
                half = [6, .005, .0007] if along_x else [.005, 6, .0007]
                center = [0, value, .001] if along_x else [value, 0, .001]
                grid = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                          rgbaColor=[.60, .67, .73, 1], physicsClientId=cid)
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=grid,
                                  basePosition=center, physicsClientId=cid)
        initial = [0, 0, 1.] if self.kind == "drone" else [0, 0, .02]
        if position is not None:
            initial = np.asarray(position, float).tolist()
            if len(initial) == 2:
                initial.append(1. if self.kind == "drone" else .02)
        self.robot_id = p.loadURDF(str(self.urdf), basePosition=initial,
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, float(yaw)]),
                                   useFixedBase=False,
                                   flags=p.URDF_USE_INERTIA_FROM_FILE,
                                   physicsClientId=cid)
        p.changeDynamics(self.robot_id, -1, linearDamping=0, angularDamping=0, physicsClientId=cid)
        dynamics = p.getDynamicsInfo(self.robot_id, -1, physicsClientId=cid)
        self.inertial_position, self.inertial_orientation = dynamics[3], dynamics[4]
        self.inverse_inertial = p.invertTransform(self.inertial_position, self.inertial_orientation)
        self.wheel_ids = []
        if self.kind == "mobile":
            joints = {p.getJointInfo(self.robot_id, i, physicsClientId=cid)[1].decode(): i
                      for i in range(p.getNumJoints(self.robot_id, physicsClientId=cid))}
            self.wheel_ids = [joints[name + "_wheel"] for name in
                              ("front_left", "front_right", "rear_left", "rear_right")]
            for wheel in self.wheel_ids:
                p.setJointMotorControl2(self.robot_id, wheel, p.VELOCITY_CONTROL,
                                       targetVelocity=0, force=12, physicsClientId=cid)
                p.changeDynamics(self.robot_id, wheel, lateralFriction=.9,
                                 anisotropicFriction=[1, .30, 1],
                                 rollingFriction=.001, spinningFriction=.001,
                                 physicsClientId=cid)
            # Initial settling uses the solver, not base-position assignment.
            for _ in range(round(.3 / self.dt)):
                p.stepSimulation(physicsClientId=cid)
        else:
            self.rotor_forces = np.full(4, self.mass * 9.81 / 4)
        self.landmark_ids = []
        colors = [(0.07, .55, .67, 1), (.95, .48, .16, 1), (.53, .35, .78, 1), (.2, .65, .40, 1)]
        for i, xy in enumerate(self.landmarks):
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=.065, height=1.4, physicsClientId=cid)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.065, length=1.4,
                                         rgbaColor=colors[i % len(colors)], physicsClientId=cid)
            body = p.createMultiBody(0, collision, visual, [float(xy[0]), float(xy[1]), .7], physicsClientId=cid)
            self.landmark_ids.append(body)
        self.t = 0.
        self.physics_steps = 0
        self.last_action = np.zeros(4)
        self.command = np.zeros(2)
        return self.state()

    def state(self):
        com_pos, com_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        com_vel, omega = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        position, quaternion = p.multiplyTransforms(com_pos, com_quat, *self.inverse_inertial)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        omega = np.asarray(omega)
        velocity = np.asarray(com_vel) - np.cross(omega, rotation @ np.asarray(self.inertial_position))
        rates = np.array([p.getJointState(self.robot_id, j, physicsClientId=self.client_id)[1] for j in self.wheel_ids])
        return {"position": np.asarray(position), "velocity": velocity,
                "quaternion": np.asarray(quaternion), "yaw": float(math.atan2(rotation[1, 0], rotation[0, 0])),
                "omega": omega, "wheel_rates": rates, "time": self.t,
                "rotation": rotation, "physics_steps": self.physics_steps}

    def _drone_action(self, state, target, yaw):
        rotation = state["rotation"]
        acceleration = np.array([3., 3., 5.]) * (target - state["position"]) - np.array([2.8, 2.8, 3.4]) * state["velocity"]
        acceleration = np.clip(acceleration, [-2.5, -2.5, -4], [2.5, 2.5, 4])
        force = self.mass * (acceleration + [0, 0, 9.81]) + .005 * state["velocity"]
        z_axis = force / np.linalg.norm(force)
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])
        y_axis = np.cross(z_axis, heading)
        y_axis /= np.linalg.norm(y_axis)
        desired = np.column_stack((np.cross(y_axis, z_axis), y_axis, z_axis))
        quaternion_error = Rotation.from_matrix(desired.T @ rotation).as_quat()
        if quaternion_error[3] < 0:
            quaternion_error *= -1
        error = 2 * quaternion_error[:3]
        omega_body = rotation.T @ state["omega"]
        torque = -np.array([.0011, .0011, .0006]) * error - np.array([.0002, .0002, .00016]) * omega_body
        torque += np.cross(omega_body, self.inertia @ omega_body)
        thrust = max(0., float(force @ rotation[:, 2]))
        return np.clip(self.inverse_mixer @ np.r_[thrust, torque], 0, self.max_rotor_force)

    def _mobile_action(self, state, target, yaw):
        delta = target[:2] - state["position"][:2]
        distance = np.linalg.norm(delta)
        if distance > .08:
            desired_heading = math.atan2(delta[1], delta[0])
            error = float(wrap_angle(desired_heading - state["yaw"]))
            speed = min(.6, 1.5 * distance) * max(0, np.cos(error))
            angular_speed = np.clip(2.8 * error, -1.5, 1.5)
        else:
            speed = 0.
            angular_speed = 0. if yaw is None else np.clip(2.8 * float(wrap_angle(yaw - state["yaw"])), -1.5, 1.5)
        # A low-level yaw-rate servo compensates skid friction using actual
        # Bullet angular velocity; wheel motors remain force limited.
        angular_drive = angular_speed + .65 * (angular_speed - state["omega"][2])
        left = (speed - self.track * angular_drive / 2) / self.wheel_radius
        right = (speed + self.track * angular_drive / 2) / self.wheel_radius
        self.command = np.array([speed, angular_speed])
        return np.clip([left, right, left, right], -9, 9)

    def step_actuators(self, action, external_force=None):
        """One physical step: rotor forces [N] or wheel velocities [rad/s]."""
        force = np.zeros(3) if external_force is None else np.asarray(external_force, float)
        action = np.asarray(action, float)
        if action.shape != (4,) or force.shape != (3,):
            raise ValueError("action must have four entries and force must have three")
        cid = self.client_id
        if self.kind == "drone":
            desired = np.clip(action, 0, self.max_rotor_force)
            self.rotor_forces += (1 - np.exp(-self.dt / .025)) * (desired - self.rotor_forces)
            for thrust, xy in zip(self.rotor_forces, self.rotor_xy):
                p.applyExternalForce(self.robot_id, -1, [0, 0, float(thrust)],
                                     [float(xy[0]), float(xy[1]), 0], p.LINK_FRAME, physicsClientId=cid)
            yaw_torque = float(self.rotor_signs @ self.rotor_forces) * self.km / self.kf
            p.applyExternalTorque(self.robot_id, -1, [0, 0, yaw_torque], p.LINK_FRAME, physicsClientId=cid)
            velocity = p.getBaseVelocity(self.robot_id, physicsClientId=cid)[0]
            force = force - .005 * np.asarray(velocity)
        else:
            p.setJointMotorControlArray(self.robot_id, self.wheel_ids, p.VELOCITY_CONTROL,
                                       targetVelocities=np.clip(action, -9, 9).tolist(),
                                       forces=[12.] * 4, physicsClientId=cid)
        position = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=cid)[0]
        p.applyExternalForce(self.robot_id, -1, force.tolist(), position, p.WORLD_FRAME, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        self.t += self.dt
        self.physics_steps += 1
        self.last_action = action.copy()

    def advance(self, target_xyz, yaw_target=0, external_force=None, duration=1 / 30):
        """Track a waypoint by rotor forces or motor joints for ``duration``.

        Mobile yaw is used at a reached waypoint; while translating the robot
        points toward its waypoint. ``None`` leaves terminal heading free.
        Duration must be a whole number of fixed solver steps.
        """
        target = np.asarray(target_xyz, float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be a finite length-three vector")
        steps = round(float(duration) / self.dt)
        if steps < 1 or not np.isclose(steps * self.dt, duration, atol=1e-9, rtol=1e-7):
            raise ValueError("duration must be a positive whole number of physics steps")
        for _ in range(steps):
            state = self.state()
            action = self._drone_action(state, target, state["yaw"] if yaw_target is None else yaw_target) if self.kind == "drone" else self._mobile_action(state, target, yaw_target)
            self.step_actuators(action, external_force)
        result = self.state()
        if not np.all(np.isfinite(np.r_[result["position"], result["velocity"], result["quaternion"]])):
            raise FloatingPointError("Nonfinite PyBullet robot state")
        return result

    def render(self, width=480, height=360):
        position = self.state()["position"].copy()
        if self.kind == "mobile":
            position[2] += .24
        view = p.computeViewMatrixFromYawPitchRoll(position.tolist(), self.camera_distance,
                                                  self.camera_yaw, self.camera_pitch, 0, 2)
        projection = p.computeProjectionMatrixFOV(48, width / height, .01, 50)
        camera = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=projection,
                                  renderer=self.renderer, shadow=1 if self.egl_plugin is not None or self.gui else 0,
                                  lightDirection=[-3, -4, 8], flags=p.ER_NO_SEGMENTATION_MASK,
                                  physicsClientId=self.client_id)
        # Builds without NumPy support return a flat tuple of Python integers.
        # bytes() copies that packed uint8 data ~9x faster than np.asarray;
        # NumPy-enabled Bullet builds can use their existing array directly.
        rgba = (np.frombuffer(bytes(camera[2]), dtype=np.uint8)
                if isinstance(camera[2], tuple) else np.asarray(camera[2], dtype=np.uint8))
        return rgba.reshape(height, width, 4)[:, :, :3].copy()

    def close(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
