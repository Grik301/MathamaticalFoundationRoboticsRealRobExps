"""Audit actual GIF duration/content and saved simulation/estimation results."""
from __future__ import annotations

import json
from pathlib import Path
import argparse
import hashlib

import numpy as np
from PIL import Image, ImageChops

from .typeset import ASSET_DIR
from .rendering import LEGEND_BOXES, RENDERING_VERSION


def verify(output: Path):
    manifest = json.loads((output / "manifest.json").read_text())
    records = manifest["experiments"]
    errors, gif_checks = [], []
    typography_count = 0
    legend_count = 0
    expected = {(page, kind) for page in range(26, 62) for kind in ("drone", "mobile")}
    actual = {(r["page"], r["robot"]) for r in records}
    if actual != expected:
        errors.append(f"Missing/extra page-robot pairs: {expected ^ actual}")
    for r in records:
        name = r["name"]
        path = output / r["gif"]
        with Image.open(path) as gif:
            duration_ms = 0
            first = gif.convert("RGB").copy()
            equation_pixels = first.crop((519, 446, 975, 564)).tobytes()
            equation_changed = False
            legends_visible = True
            for i in range(gif.n_frames):
                gif.seek(i)
                duration_ms += gif.info.get("duration", 0)
                if r.get("presentation_version") == RENDERING_VERSION:
                    equation_changed |= gif.convert("RGB").crop((519, 446, 975, 564)).tobytes() != equation_pixels
                    for box in LEGEND_BOXES:
                        pixels = np.asarray(gif.convert("RGB").crop(box))
                        legends_visible &= np.count_nonzero(pixels.min(axis=2) < 150) > 500
            last = gif.convert("RGB")
            differs = ImageChops.difference(first, last).getbbox() is not None
            frames = gif.n_frames
        if r.get("presentation_version") == RENDERING_VERSION:
            typography_count += 1
            legends = r.get("color_legends", {})
            if (legends_visible and r.get("legend_frames_checked") == frames
                    and len(legends.get("physical", [])) >= 2 and legends.get("model")
                    and all(e.get("color_name") and e.get("label") for group in legends.values() for e in group)):
                legend_count += 1
                for group in legends.values():
                    for entry in group:
                        pixels = np.asarray(last.crop(entry["box"]))
                        if np.count_nonzero(pixels.min(axis=2) < 150) < 40:
                            errors.append(f"{name}: missing visible legend entry {entry['text']}")
            else:
                errors.append(f"{name}: incomplete or missing color legends")
            if equation_changed or not (ASSET_DIR/f"s{r['page']}.svg").exists():
                errors.append(f"{name}: inconsistent or missing LaTeX equation artwork")
            equation = np.asarray(first.crop((519, 446, 975, 564)))
            if np.count_nonzero(equation.min(axis=2) < 150) < 150:
                errors.append(f"{name}: equation block appears blank")
            refresh = r.get("presentation_refresh")
            if refresh:
                if not refresh["camera_frames_pixel_identical"] or not refresh["frame_durations_unchanged"]:
                    errors.append(f"{name}: presentation refresh changed the recording")
                for filename, expected_sha in refresh["raw_logs_sha256"].items():
                    if hashlib.sha256((output/"logs"/filename).read_bytes()).hexdigest() != expected_sha:
                        errors.append(f"{name}: raw simulation log changed after refresh")
        if not differs or frames != r["frames"]:
            errors.append(f"{name}: frame count or animation content mismatch")
        if not np.isclose(duration_ms / 1000, r["simulation_seconds"]) or duration_ms > 30000:
            errors.append(f"{name}: GIF duration mismatch/limit")
        if r["wall_seconds_including_gif"] >= 30 or not r["paced_realtime"]:
            errors.append(f"{name}: real-time runtime constraint failed")
        if abs(r["simulation_wall_seconds"] - r["simulation_seconds"]) > .5:
            errors.append(f"{name}: wall pacing lag exceeds 0.5s")
        with np.load(output / r["log"], allow_pickle=False) as log:
            if not np.isfinite(log["states"]).all() or not np.isfinite(log["actuators"]).all():
                errors.append(f"{name}: nonfinite robot data")
            if len(log["times"]) != round(30*r["simulation_seconds"]):
                errors.append(f"{name}: physical trajectory length mismatch")
        if r["physical"]["physics_steps"] != round(r["simulation_seconds"]*240):
            errors.append(f"{name}: wrong solver step count")
        if r["robot"] == "drone" and r["physical"]["min_height_m"] < .2:
            errors.append(f"{name}: drone touched/approached floor unexpectedly")
        math = r["mathematics_or_slam"]
        if r["page"] == 59:
            if not (math["mapped_landmarks"] == 7 and math["physical_loop_completed"] and math["revisit_count"] > 0):
                errors.append(f"{name}: SLAM map/loop incomplete")
            if math["position_rmse_m"] >= math["dead_reckoning_position_rmse_m"]:
                errors.append(f"{name}: SLAM did not improve odometry")
            if math["covariance_minimum_eigenvalue"] < -1e-10:
                errors.append(f"{name}: SLAM covariance not PSD")
        elif not math["all_numerical_checks_passed"]:
            errors.append(f"{name}: numerical checks failed: {math['checks']}")
        gif_checks.append({"name": name, "decoded_gif_seconds": duration_ms/1000,
                           "decoded_frames": frames, "animated": differs,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if typography_count != len(records):
        errors.append(f"Only {typography_count}/{len(records)} GIFs have the current LaTeX presentation")
    report = {"passed": not errors, "experiments_checked": len(records), "latex_gifs_checked": typography_count,
              "color_legend_gifs_checked": legend_count, "errors": errors,
              "max_wall_seconds_including_gif": max((r["wall_seconds_including_gif"] for r in records), default=0),
              "max_gif_seconds": max((r["gif_seconds"] for r in records), default=0),
              "gif_audit": gif_checks}
    (output / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    status = "PASS" if report["passed"] else "FAIL"
    checks = sum(len(r["mathematics_or_slam"].get("checks", {})) for r in records)
    summary = ["# Recorded experiment results", "", f"Audit: **{status}**. {len(records)}/72 experiments recorded.", "",
               f"- Decoded GIFs: {len(gif_checks)}; each has changing frames and the expected playback duration.",
               f"- Longest GIF: {report['max_gif_seconds']:.1f} seconds.",
               f"- Longest measured run, including GIF creation: {report['max_wall_seconds_including_gif']:.2f} seconds.",
               f"- Numerical checks across the delivered runs: {checks}.",
               f"- GIFs with typeset LaTeX equations, checked on every frame: {typography_count}.",
               f"- GIFs with explicit color names and full curve/marker legends on every frame: {legend_count}.",
               "- Fixed 240 Hz PyBullet dynamics; original downloaded Crazyflie and Husky visual meshes.", "",
               "## Joint EKF-SLAM", "", "| Robot | Pose RMSE | Odometry RMSE | Map RMSE | Landmarks | Revisits |",
               "|---|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["page"] == 59:
            m = r["mathematics_or_slam"]
            summary.append(f"| {r['robot']} | {m['position_rmse_m']:.4f} m | {m['dead_reckoning_position_rmse_m']:.4f} m | {m['map_rmse_m']:.4f} m | {m['mapped_landmarks']} | {m['revisit_count']} |")
    summary += ["", "These are simulations of real robot models. Numerical ensembles are separate reduced models; SLAM uses synthetic noisy sensors and a planar joint pose/map estimate.", "",
                "[GIF gallery](index.html) · [Slide coverage](SLIDE_COVERAGE.md) · [Full metrics](manifest.json) · [GIF audit and hashes](verification.json)", ""]
    if errors:
        summary += ["## Audit failures", ""] + [f"- {message}" for message in errors]
    (output / "REPORT.md").write_text("\n".join(summary))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path(__file__).resolve().parents[1]/"results")
    report = verify(parser.parse_args().output)
    print(json.dumps({k:v for k,v in report.items() if k != "gif_audit"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
