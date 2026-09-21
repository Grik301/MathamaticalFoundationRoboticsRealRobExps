"""Download pinned, licensed upstream robot models; keep originals intact.

The repository includes originals, prepared URDFs, licenses and a manifest.
Run ``python -m lecture6_sim.download_assets`` to refresh the pinned copies;
ordinary experiments load the bundled files without downloading them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET


ASSETS = Path(__file__).resolve().parent / "assets"
UPSTREAM = ASSETS / "upstream"
DRONE_COMMIT = "7ebad1ecabd28a7000add2d05f888aa2e837c2cc"
BULLET_COMMIT = "63c4d67e337017f9d8b298c900e9aabdb69296e7"


def sources():
    drone = f"https://raw.githubusercontent.com/learnsyslab/gym-pybullet-drones/{DRONE_COMMIT}"
    bullet = f"https://raw.githubusercontent.com/bulletphysics/bullet3/{BULLET_COMMIT}"
    out = [(f"crazyflie/{name}", f"{drone}/gym_pybullet_drones/assets/{name}")
           for name in ("cf2x.urdf", "cf2.dae")]
    out += [("crazyflie/LICENSE", f"{drone}/LICENSE")]
    out += [(f"husky/{name}", f"{bullet}/examples/pybullet/gym/pybullet_data/husky/{name}")
            for name in ("husky.urdf", "meshes/base_link.stl", "meshes/wheel.stl",
                         "meshes/top_plate.stl", "meshes/user_rail.stl", "meshes/bumper.stl")]
    return out + [("husky/BULLET_LICENSE.txt", f"{bullet}/LICENSE.txt")]


def prepare_models():
    """Adapt only unresolved Xacro defaults and missing massless-link inertias.

    The Husky upstream file leaves Xacro optenv strings in the IMU origin.
    Bullet assigns a 1 kg mass to a link without an inertial element. Add
    explicit zero inertias to visual/sensor fixed links to avoid those phantom
    masses. Express the chassis inertia on the fixed-connected root footprint
    frame because Bullet treats a zero-mass URDF root as static before merging.
    Existing total mass/inertia, meshes and joint frames are kept.
    """
    text = (UPSTREAM / "husky/husky.urdf").read_text()
    text = re.sub(r"\$\(optenv HUSKY_IMU_RPY [^)]*\)", "0 -1.5708 3.1416", text)
    text = re.sub(r"\$\(optenv HUSKY_IMU_XYZ [^)]*\)", "0.19 0 0.149", text)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(text, parser=parser)
    chassis = root.find("link[@name='base_link']")
    footprint = root.find("link[@name='base_footprint']")
    inertia = chassis.find("inertial")
    chassis.remove(inertia)
    origin = inertia.find("origin")
    xyz = [float(value) for value in origin.get("xyz").split()]
    xyz[2] += .14493
    origin.set("xyz", " ".join(str(value) for value in xyz))
    footprint.append(inertia)
    for link in root.findall("link"):
        if link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0")
            ET.SubElement(inertial, "inertia", ixx="0", iyy="0", izz="0",
                          ixy="0", ixz="0", iyz="0")
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", "../upstream/husky/" + mesh.get("filename"))
    prepared = ASSETS / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(prepared / "husky.urdf", encoding="utf-8", xml_declaration=True)
    # Retain the original DAE mesh, all original visual transforms and the
    # original motor parameters. PyBullet supports COLLADA directly.
    drone = ET.parse(UPSTREAM / "crazyflie/cf2x.urdf")
    for mesh in drone.findall(".//mesh"):
        mesh.set("filename", "../upstream/crazyflie/" + Path(mesh.get("filename")).name)
    ET.indent(drone)
    drone.write(prepared / "cf2x.urdf", encoding="utf-8", xml_declaration=True)
    manifest = {"repositories": {
        "crazyflie": {"repository": "https://github.com/learnsyslab/gym-pybullet-drones", "commit": DRONE_COMMIT},
        "husky": {"repository": "https://github.com/bulletphysics/bullet3", "commit": BULLET_COMMIT}},
        "files": [{"path": f"upstream/{name}", "url": url,
                   "sha256": hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest(),
                   "bytes": (UPSTREAM / name).stat().st_size} for name, url in sources()]}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    for name, url in sources():
        path = UPSTREAM / name
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Lecture6-PyBullet-Simulations"})
        data = urllib.request.urlopen(request, timeout=60).read()
        path.write_bytes(data)
        print(f"Downloaded {name}: {len(data):,} bytes")
    prepare_models()


if __name__ == "__main__":
    main()
