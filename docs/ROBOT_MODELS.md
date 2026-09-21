# Robot models and simulation assumptions

The downloader retrieves existing descriptions of the Bitcraze Crazyflie 2.x and Clearpath Husky, including their original visual meshes. These files are generated locally under `lecture6_sim/assets/` and are not included in the source repository.

| Model | Pinned upstream source | License |
|---|---|---|
| Crazyflie `cf2x.urdf`, `cf2.dae` | [gym-pybullet-drones, commit 7ebad1ec](https://github.com/learnsyslab/gym-pybullet-drones/tree/7ebad1ecabd28a7000add2d05f888aa2e837c2cc/gym_pybullet_drones/assets) | MIT |
| Husky URDF and five STL meshes | [Bullet, commit 63c4d67e](https://github.com/bulletphysics/bullet3/tree/63c4d67e337017f9d8b298c900e9aabdb69296e7/examples/pybullet/gym/pybullet_data/husky) | Clearpath BSD notice in the original URDF; Bullet zlib notice also retained |

Run `python -m lecture6_sim.download_assets` from the repository root after installation. It stores byte-for-byte originals in `lecture6_sim/assets/upstream/`, prepares runnable URDFs in `lecture6_sim/assets/prepared/`, and writes source URLs, pinned commits, SHA256 hashes, and byte sizes to `lecture6_sim/assets/manifest.json`.

Downloaded notices are kept at `lecture6_sim/assets/upstream/crazyflie/LICENSE`, in the original Husky URDF, and at `lecture6_sim/assets/upstream/husky/BULLET_LICENSE.txt`. The project's MIT license does not replace these upstream licenses.

## URDF preparation

- Prepared mesh paths point to the original downloaded meshes.
- Husky's two unresolved Xacro IMU `optenv` entries are replaced with the literal defaults already present in its upstream file.
- Visual/sensor links lacking inertia receive explicit zero-mass inertial elements, avoiding Bullet's implicit 1 kg default for those decorative links.
- Husky's chassis inertial element is expressed on the fixed-connected `base_footprint` root frame by adding its 0.14493 m vertical joint offset to the inertial origin. The total mass, inertia tensor, and center of mass in physical space are preserved. This keeps the root dynamic: Bullet otherwise treats a zero-mass root as static. No fixed-link merging is used. The base retains a mass of 33.455 kg, and each wheel retains its 2.637 kg mass.
- Original visual meshes are retained. Original collision primitives also remain unchanged. Bullet loads Crazyflie's COLLADA mesh directly with its original visual transform.

## Dynamics and control

Crazyflie's 0.027 kg mass, diagonal inertia `(1.4e-5, 1.4e-5, 2.17e-5)` kg m², rotor positions, motor coefficients `kf=3.16e-10` and `km=7.94e-12`, and maximum thrust-to-weight ratio 2.25 come from the downloaded model. Four separate rotor forces and their reaction torque act on the rigid body. Position and quaternion attitude feedback determine motor commands. A 25 ms motor response and 0.005 N s/m drag are educational simulation choices, not measured hardware parameters.

Husky's 0.17775 m wheel radius, 0.5708 m track, masses, inertia tensors, and joint transforms come from its downloaded model. Four joint velocity motors, each limited to 12 N m, drive its wheels. Bullet solves gravity, contact, friction, and slip. The 0.6 m/s controller speed limit, 9 rad/s motor limit, lateral friction 0.9, anisotropic friction `[1, 0.30, 1]`, rolling/spinning friction 0.001, and feedback gains are simulation settings, not identified drivetrain or terrain parameters.

Physics advances in fixed 1/240 s increments with 80 solver iterations. Headless recording uses PyBullet `DIRECT`, preferably with its EGL/OpenGL renderer; `GUI` supports live viewing. TinyRenderer is the fallback and can be selected with `LECTURE6_RENDERER=tiny`. `World.renderer_name` records the actual backend. Software-renderer thread count defaults to four and can be overridden with `LP_NUM_THREADS`. Rendering speed depends on the machine and driver.

Robot pose is set during initialization; subsequent motion is driven by forces and motors. Husky receives a 0.3 s gravity/contact settling period during initialization. Returned positions and velocities refer to the URDF root frame after correcting the inertial-frame pose and center-of-mass velocity. SLAM landmarks are cylinders with collision geometry, 0.065 m radius and 1.4 m height.

These experiments exercise mathematical models and estimators in rigid-body simulation. They do not connect to physical robots or claim hardware validation. The SLAM sensor and estimator limitations are described in the [README](../README.md#physics-and-slam).
