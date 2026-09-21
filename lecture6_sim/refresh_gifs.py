"""Rebuild GIF typography from unchanged recorded camera frames and logs.

This is presentation postprocessing, not a new physics experiment. Original
simulation timing and results remain intact. Every camera pixel, frame duration
and raw numerical log is checked before the refreshed GIF replaces its source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .equations import LATEX_EQUATIONS
from .mathematics import TopicExperiment
from .rendering import Dashboard, RENDERING_VERSION, save_gif
from .run import ROOT, make_gallery, parse_pages
from .slam import SLAMExperiment
from .topics import TOPICS

CAMERA_BOX = (26, 126, 477, 418)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_slam_panel(experiment, data, metrics, k):
    for key in ("truth", "estimate", "dead_reckoning"):
        experiment.history[key] = data["model_"+key][:k+1]
    dimension = 3+2*int(data["model_landmark_count"][k])
    experiment.filter.mean = data["model_joint_mean"][k, :dimension].copy()
    experiment.filter.covariance = data["model_joint_covariance"][k, :dimension, :dimension].copy()
    experiment.filter.landmark_indices = {
        int(tag): int(index) for tag, index in zip(data["model_final_landmark_ids"], data["model_final_landmark_state_indices"])
        if int(index) < dimension
    }
    experiment.revisit_events = [event for event in metrics["revisit_events"] if event["time_s"] <= data["model_time"][k]+1e-9]
    experiment.last_revisit_correction = (experiment.revisit_events[-1]["position_correction_m"]
                                          if experiment.revisit_events else 0.)
    return experiment.panel()


def refresh_one(record_path):
    started = time.perf_counter()
    record = json.loads(record_path.read_text())
    output = record_path.parent.parent
    source = output/record["gif"]
    raw_paths = [output/record["log"], record_path.with_suffix(".csv")]
    raw_hashes = {str(path): digest(path) for path in raw_paths}
    source_hash = digest(source)
    kind, page = record["robot"], record["page"]
    with np.load(output/record["log"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if page == 59:
        experiment = SLAMExperiment(SimpleNamespace(kind=kind, landmarks=data["model_true_landmarks"]),
                                    seed=record["seed"], duration=record["simulation_seconds"])
    else:
        experiment = TopicExperiment(page, kind, record["seed"], duration=record["simulation_seconds"], dt=1/30)
    dashboard = Dashboard(TOPICS[page], kind, record["simulation_seconds"])
    frames, camera_hashes, durations = [], [], []
    with Image.open(source) as original:
        for i in range(original.n_frames):
            original.seek(i)
            durations.append(original.info.get("duration", 0))
            camera = original.convert("RGB").crop(CAMERA_BOX)
            camera_hashes.append(hashlib.sha256(camera.tobytes()).hexdigest())
            k = (i+1)*3-1
            state = data["states"][k]
            quaternion = state[6:10]
            x, y, z, w = quaternion
            yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            physical = {"position": state[:3], "velocity": state[3:6], "yaw": float(yaw)}
            if page == 59:
                command = data["commands"][k]
                payload = {"target": command[:3], "yaw": command[3], "force": data["external_forces_N"][k],
                           "panel": replay_slam_panel(experiment, data, record["mathematics_or_slam"], k)}
            else:
                payload = experiment.step(k/30)
                np.testing.assert_allclose(payload["target"], data["commands"][k, :3], atol=1e-12, rtol=0)
                np.testing.assert_allclose(payload["force"], data["external_forces_N"][k], atol=1e-12, rtol=0)
            frames.append(dashboard.frame(np.asarray(camera), physical, payload, float(data["times"][k])))
    if len(frames) != record["frames"] or set(durations) != {round(1000/record["fps"])}:
        raise ValueError(f"Unexpected source animation timing: {source}")
    temporary = source.with_suffix(".refresh.gif")
    save_gif(frames, temporary, record["fps"], preserve_palette_colors=True)
    with Image.open(temporary) as refreshed:
        if refreshed.n_frames != len(frames):
            raise ValueError("GIF frame count changed")
        for i in range(refreshed.n_frames):
            refreshed.seek(i)
            camera = refreshed.convert("RGB").crop(CAMERA_BOX)
            if hashlib.sha256(camera.tobytes()).hexdigest() != camera_hashes[i]:
                raise ValueError(f"Camera pixels changed in {source.name}, frame {i}")
            if refreshed.info.get("duration", 0) != durations[i]:
                raise ValueError("GIF playback timing changed")
    if any(digest(path) != raw_hashes[str(path)] for path in raw_paths):
        raise ValueError("A raw simulation log changed")
    temporary.replace(source)
    frames[len(frames)//2].save(output/record["preview"])
    record["presentation_version"] = RENDERING_VERSION
    record["color_legends"] = dashboard.color_legends
    record["legend_frames_checked"] = dashboard.legend_frames_checked
    record["presentation_refresh"] = {
        "method": "Reconstructed dashboard; original camera pixels and raw simulation logs preserved",
        "seconds": time.perf_counter()-started,
        "source_gif_sha256": source_hash,
        "latex_equations": list(LATEX_EQUATIONS[page]),
        "camera_frames_pixel_identical": True,
        "frame_durations_unchanged": True,
        "raw_logs_sha256": {Path(path).name: sha for path, sha in raw_hashes.items()},
        "dashboard_pixels": [dashboard.width, dashboard.height],
    }
    record_path.write_text(json.dumps(record, indent=2)+"\n")
    print(f"{record['name']}: equations and color legends updated; {len(frames)} camera frames unchanged; {time.perf_counter()-started:.2f}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--pages", type=parse_pages, default=parse_pages("26-61"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in sorted((args.output/"logs").glob("s??_*.json")):
        record = json.loads(path.read_text())
        if record["page"] not in args.pages:
            continue
        if not args.force and record.get("presentation_version") == RENDERING_VERSION:
            continue
        refresh_one(path)
    make_gallery(args.output)


if __name__ == "__main__":
    main()
