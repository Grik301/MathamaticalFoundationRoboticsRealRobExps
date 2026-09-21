"""Run the complete slide-to-robot matrix, recording real-time Bullet frames."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import time

import numpy as np

from .mathematics import TopicExperiment
from .physics import World
from .rendering import Dashboard, save_gif, RENDERING_VERSION
from .slam import SLAMExperiment, SLAM_LANDMARKS
from .topics import TOPICS

ROOT = Path(__file__).resolve().parents[1]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def run_one(page, kind, output, *, duration=6., seed=123, realtime=True, gui=False, fps=10,
            camera_width=360):
    """Timing includes construction, rendering, logging and GIF encoding."""
    if not (0 < duration <= 24):
        raise ValueError("Choose 0 < duration <= 24 seconds (leaves time for GIF encoding).")
    if fps != 10:
        raise ValueError("Use 10 fps: exact 100 ms GIF frames and three controller steps per frame.")
    if not np.isclose(duration * fps, round(duration * fps)):
        raise ValueError("Duration must be a multiple of 0.1 seconds.")
    if not 160 <= camera_width <= 960:
        raise ValueError("camera_width must be between 160 and 960 pixels.")
    camera_height = round(camera_width * 292 / 451)
    start = time.perf_counter()
    topic = TOPICS[page]
    for child in ("gifs", "logs", "previews"):
        (output / child).mkdir(parents=True, exist_ok=True)
    stem = f"s{page:02d}_{kind}"
    world = World(kind, gui=gui, dt=1/240, landmarks=SLAM_LANDMARKS if page == 59 else None)
    frames, states, commands, forces, timestamps, actuators = [], [], [], [], [], []
    max_lag = 0.
    try:
        experiment = (SLAMExperiment(world, seed=seed, duration=duration, dt=1/30) if page == 59
                      else TopicExperiment(page, kind, seed, duration=duration, dt=1/30))
        dashboard = Dashboard(topic, kind, duration)
        # Compile camera shaders during measured setup, before the pacing clock.
        world.render(width=camera_width, height=camera_height)
        simulation_start = time.perf_counter()
        steps = round(duration * 30)
        for k in range(steps):
            payload = (experiment.step(k / 30) if page == 59 else
                       experiment.step(k / 30, include_panel=(k + 1) % 3 == 0))
            world.advance(payload["target"], yaw_target=payload.get("yaw"),
                          external_force=payload.get("force", np.zeros(3)), duration=1/30)
            s = world.state()
            vector = np.r_[s["position"], s["velocity"], s["quaternion"], s["omega"]]
            if not np.isfinite(vector).all():
                raise FloatingPointError(f"Nonfinite physics state in {stem}")
            states.append(vector)
            commands.append(np.r_[payload["target"], np.nan if payload.get("yaw") is None else payload["yaw"]])
            forces.append(payload.get("force", np.zeros(3)))
            actuators.append(world.last_action.copy())
            timestamps.append((k + 1) / 30)
            if (k + 1) % 3 == 0:
                frames.append(dashboard.frame(world.render(width=camera_width, height=camera_height), s, payload, (k+1)/30))
            if realtime:
                deadline = simulation_start + (k + 1) / 30
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    max_lag = max(max_lag, -remaining)
        simulation_wall = time.perf_counter() - simulation_start
        data = {"times": np.array(timestamps), "states": np.array(states),
                "commands": np.array(commands), "external_forces_N": np.array(forces),
                "actuators": np.array(actuators)}
        if hasattr(experiment, "archive"):
            data.update({"model_" + key: value for key, value in experiment.archive().items()})
        np.savez_compressed(output / "logs" / f"{stem}.npz", **data)
        columns = np.column_stack([data["times"], data["states"], data["commands"], data["external_forces_N"]])
        np.savetxt(output / "logs" / f"{stem}.csv", columns, delimiter=",", comments="",
                   header="t,x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,target_x,target_y,target_z,target_yaw,force_x,force_y,force_z")
        frames[len(frames)//2].save(output / "previews" / f"{stem}.png")
        save_gif(frames, output / "gifs" / f"{stem}.gif", fps)
        final = np.array(states)
        tracking_dimensions = 3 if kind == "drone" else 2
        physical = {
            "finite": bool(np.isfinite(final).all()),
            "distance_travelled_m": float(np.linalg.norm(np.diff(final[:, :3], axis=0), axis=1).sum()),
            "max_speed_m_s": float(np.linalg.norm(final[:, 3:6], axis=1).max()),
            "min_height_m": float(final[:, 2].min()),
            "max_height_m": float(final[:, 2].max()),
            "max_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(final[:, 6:10], axis=1) - 1))),
            "tracking_rmse_m": float(np.sqrt(np.mean(np.sum((final[:, :tracking_dimensions]-data['commands'][:, :tracking_dimensions])**2, axis=1)))),
            "tracking_dimensions": "xyz" if kind == "drone" else "xy",
            "physics_steps": world.physics_steps,
            "actuator_units": "rotor thrust N" if kind == "drone" else "wheel angular velocity rad/s",
        }
        math_metrics = experiment.metrics()
        world.close()
        elapsed = time.perf_counter() - start
        record = {
            "name": stem, "page": page, "printed_slide": page-1, "robot": kind,
            "title": topic.title, "category": topic.category,
            "seed": seed, "simulation_seconds": duration, "gif_seconds": len(frames)/fps,
            "wall_seconds_including_gif": elapsed, "simulation_wall_seconds": simulation_wall,
            "paced_realtime": realtime, "maximum_frame_deadline_lag_seconds": max_lag,
            "under_30_seconds": elapsed < 30 and duration <= 30,
            "fps": fps, "frames": len(frames), "physics_hz": 240, "controller_hz": 240, "reference_hz": 30,
            "backend": "pybullet", "renderer": getattr(world, "renderer_name", "ER_TINY_RENDERER"),
            "camera_pixels": [camera_width, camera_height],
            "presentation_version": RENDERING_VERSION,
            "color_legends": dashboard.color_legends,
            "legend_frames_checked": dashboard.legend_frames_checked,
            "pose_teleportation_during_episode": False,
            "hardware_experiment": False,
            "scope": "Actuated URDF rigid-body/contact simulation; mathematical ensembles are separate numerical models. Controller uses simulated state feedback.",
            "physical": physical, "mathematics_or_slam": math_metrics,
            "gif": f"gifs/{stem}.gif", "preview": f"previews/{stem}.png", "log": f"logs/{stem}.npz",
        }
        (output / "logs" / f"{stem}.json").write_text(json.dumps(jsonable(record), indent=2) + "\n")
        print(f"{stem}: {duration:g}s simulated, {elapsed:.2f}s including GIF; {len(frames)} frames", flush=True)
        return record
    finally:
        world.close()


def make_gallery(output):
    records = [json.loads(p.read_text()) for p in sorted((output / "logs").glob("s??_*.json"))]
    manifest = {"source_pdf": "lecture6-mbzuai.pdf", "pdf_pages": [26, 61],
                "expected_experiments": 72, "recorded_experiments": len(records),
                "all_under_30_seconds": all(r["under_30_seconds"] for r in records),
                "experiments": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cards = []
    rows = ["| PDF page (printed) | Topic | Crazyflie | Husky |", "|---|---|---|---|"]
    for page, topic in TOPICS.items():
        links = []
        for kind in ("drone", "mobile"):
            found = next((r for r in records if r["page"] == page and r["robot"] == kind), None)
            links.append(f"[GIF]({found['gif']}) · [metrics](logs/{found['name']}.json)" if found else "pending")
        rows.append(f"| {page} ({page-1}) | {topic.title} | {' | '.join(links)} |")
    (output / "SLIDE_COVERAGE.md").write_text("# All requested slides\n\nEach GIF uses actual PyBullet camera frames. The right panel contains a separately labelled numerical model or sensor estimate.\n\n" + "\n".join(rows) + "\n")
    for r in records:
        robot = "Crazyflie" if r["robot"] == "drone" else "Husky"
        cards.append(f'''<article data-robot="{r['robot']}" data-page="{r['page']}">
<a href="{r['gif']}"><img loading="lazy" src="{r['preview']}" alt="Page {r['page']}: {html.escape(r['title'])}"></a>
<div class="body"><small>PDF {r['page']} / printed {r['printed_slide']} · {robot}</small>
<h2>{html.escape(r['title'])}</h2><p>{r['simulation_seconds']:g}s playback · {r['wall_seconds_including_gif']:.2f}s total runtime</p>
<a href="{r['gif']}">Play GIF ↗</a> · <a href="logs/{r['name']}.json">Metrics</a> · <a href="{r['log']}">Trajectory</a></div></article>''')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lecture 6 · PyBullet robot experiments</title><style>
body{margin:0;font:16px system-ui,sans-serif;background:#edf2f5;color:#203348}header{padding:40px max(24px,5vw);background:#203348;color:white}h1{margin:12px 0;font-size:34px}header p{max-width:940px;line-height:1.6;color:#d4e7ee}.eyebrow{color:#9de1e2;font-size:13px;letter-spacing:2px}nav{padding:22px 5vw;position:sticky;top:0;background:#edf2f5ee}button{border:1px solid #9ab3c3;background:white;padding:9px 17px;margin-right:8px;border-radius:6px;cursor:pointer;color:#203348}main{padding:0 5vw 40px;display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px}article{background:white;border-radius:9px;overflow:hidden;border:1px solid #d4e1e8}article img{width:100%;display:block}.body{padding:18px}small{color:#477383}h2{font-size:19px}p{font-size:14px}a{color:#007e91}header a{color:#9de1e2}footer{padding:20px 5vw;line-height:1.6}article[hidden]{display:none}
</style><header><div class="eyebrow">MATHEMATICS FOR ROBOTICS / LECTURE 6</div><h1>Probability in motion</h1>
<p>72 experiments across PDF pages 26–61, on a Crazyflie 2.0 and a Clearpath Husky. Downloaded robot URDFs run under gravity, rotor forces or wheel motors in PyBullet. Every GIF plays at the recorded simulation speed. Open a preview to play its GIF.</p>
<p>The plots show numerical probability models alongside physical robot tracking. SLAM jointly estimates pose and an initially unknown landmark map using noisy simulated sensors. These are physics simulations of real robot models, not hardware measurements.</p>
<a href="SLIDE_COVERAGE.md">Slide coverage</a> · <a href="manifest.json">All timings and results</a> · <a href="../README.md">Run instructions and assumptions</a></header>
<nav><button onclick="filter('all')">All robots</button><button onclick="filter('drone')">Crazyflie</button><button onclick="filter('mobile')">Husky</button><button onclick="filter('slam')">SLAM</button></nav><main>'''
    document += "\n".join(cards)
    document += '''</main><footer>Real-time pacing uses deterministic 240 Hz fixed physics steps and wall-clock scheduling. Each recorded experiment includes measured runtime, GIF duration, raw state and actuator-reference logs. See the README for model and sensor assumptions.</footer><script>function filter(kind){document.querySelectorAll('article').forEach(x=>x.hidden=!(kind==='all'||x.dataset.robot===kind||(kind==='slam'&&x.dataset.page==='59')))}</script></html>'''
    (output / "index.html").write_text(document)
    return manifest


def parse_pages(text):
    pages = set()
    for chunk in text.split(","):
        if "-" in chunk:
            a, b = map(int, chunk.split("-"))
            pages.update(range(a, b+1))
        else:
            pages.add(int(chunk))
    if not pages or not pages <= set(range(26, 62)):
        raise argparse.ArgumentTypeError("Pages must be within 26–61.")
    return sorted(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=parse_pages, default=parse_pages("26-61"))
    parser.add_argument("--robot", choices=("drone", "mobile", "both"), default="both")
    parser.add_argument("--duration", type=float, default=6.)
    parser.add_argument("--slam-duration", type=float, default=18.)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--camera-width", type=int, default=360, help="Native camera width; dashboard remains 1000×970.")
    parser.add_argument("--fast", action="store_true", help="Disable wall pacing; GIF playback is still 1×.")
    parser.add_argument("--gui", action="store_true", help="Open the live PyBullet desktop viewer.")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true", help="Skip previously successful GIFs with matching settings.")
    args = parser.parse_args()
    kinds = ("drone", "mobile") if args.robot == "both" else (args.robot,)
    for page in args.pages:
        for kind in kinds:
            duration = args.slam_duration if page == 59 else args.duration
            seed = args.seed + page * 10 + int(kind == "mobile")
            log = args.output / "logs" / f"s{page:02d}_{kind}.json"
            if args.resume and log.exists():
                record = json.loads(log.read_text())
                if (record["under_30_seconds"] and record["simulation_seconds"] == duration
                        and record["seed"] == seed and record["paced_realtime"] == (not args.fast)
                        and record.get("camera_pixels", [451, 292])[0] == args.camera_width
                        and record.get("presentation_version") == RENDERING_VERSION
                        and (args.output / record["gif"]).exists()):
                    continue
            run_one(page, kind, args.output, duration=duration, seed=seed, realtime=not args.fast,
                    gui=args.gui, camera_width=args.camera_width)
            make_gallery(args.output)
    manifest = make_gallery(args.output)
    print(f"Gallery: {args.output / 'index.html'} ({manifest['recorded_experiments']} experiments)")
    if not manifest["all_under_30_seconds"]:
        raise SystemExit("One or more recorded experiments exceeded 30 seconds. See manifest.json.")


if __name__ == "__main__":
    main()
