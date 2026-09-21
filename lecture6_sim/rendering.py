"""Lightweight dashboard: Bullet camera pixels plus explicitly numerical plots."""
from __future__ import annotations

import math
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .typeset import equation_image, RENDERING_VERSION as EQUATION_VERSION

COLORS = ["#007e91", "#e07430", "#765bb5", "#cc4255", "#348452", "#658399"]
COLOR_NAMES = ["Teal (blue-green)", "Orange", "Purple", "Red", "Green", "Slate blue"]
SAMPLE_COLOR = "#8da8b7"
TRUTH_COLOR = "#8b9298"
RENDERING_VERSION = EQUATION_VERSION + "-color-legends-v1"
LEGEND_BOXES = ((14, 688, 489, 931), (503, 688, 986, 931))
INK = "#203348"
MUTED = "#586d7c"
BG = "#edf2f5"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size, layout_engine=ImageFont.Layout.BASIC)
    except OSError:
        return ImageFont.truetype(name, size, layout_engine=ImageFont.Layout.BASIC)


FONTS = {s: font(s) for s in (11, 12, 13, 14, 15, 17, 20)}
TITLE_FONT = font(22, True)


def short(text, max_chars=80):
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def curve_color(curve, index):
    return curve.get("color", COLORS[index % len(COLORS)])


def legend_entry(color, label, marker="line", color_name=None):
    names = dict(zip(COLORS, COLOR_NAMES)) | {SAMPLE_COLOR: "Blue-gray", TRUTH_COLOR: "Gray"}
    if not label:
        raise ValueError("Every plotted item needs an explicit legend label")
    return {"color": color, "color_name": color_name or names[color],
            "label": label, "marker": marker}


def panel_legend(panel):
    entries = [legend_entry(curve_color(curve, i), curve.get("label"))
               for i, curve in enumerate(panel.get("curves", []))]
    if len(panel.get("samples", [])):
        entries.append(legend_entry(SAMPLE_COLOR, panel["samples_label"], "dot"))
    if "landmark_estimates" in panel:
        entries.extend([
            legend_entry(TRUTH_COLOR, "Crosses: true landmark positions", "cross"),
            legend_entry(COLORS[3], "Circles: estimated landmark positions", "circle"),
            legend_entry(COLORS[3], "Ellipses: 95% pose and landmark uncertainty", "ellipse"),
        ])
    return entries


def wrap_pixels(draw, text, max_width, face):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=face) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_legend(draw, box, title, entries):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill="white")
    draw.text((x0 + 14, y0 + 12), title, fill=INK, font=FONTS[14])
    y = y0 + 39
    rendered = []
    for entry in entries:
        text = f"{entry['color_name']}: {entry['label']}"
        lines = wrap_pixels(draw, text, x1 - x0 - 62, FONTS[13])
        height = max(24, 17 * len(lines) + 6)
        if y + height > y1 - 8:
            raise ValueError(f"Legend does not fit: {title}: {text}")
        color, marker = entry["color"], entry["marker"]
        x, cy = x0 + 14, y + 8
        if marker == "dot":
            draw.ellipse((x + 8, cy - 3, x + 14, cy + 3), fill=color)
        elif marker == "cross":
            draw.line((x + 6, cy - 5, x + 16, cy + 5), fill=color, width=2)
            draw.line((x + 6, cy + 5, x + 16, cy - 5), fill=color, width=2)
        elif marker in ("circle", "ellipse"):
            half = 5 if marker == "circle" else 11
            draw.ellipse((x + 11 - half, cy - 5, x + 11 + half, cy + 5), outline=color, width=2)
        else:
            draw.line((x, cy, x + 24, cy), fill=color, width=3)
        for j, line in enumerate(lines):
            draw.text((x0 + 48, y + 17*j), line, fill=INK, font=FONTS[13])
        rendered.append(dict(entry, text=text, box=[x0 + 12, y, x1 - 10, y + height]))
        y += height
    return rendered


def plot(draw, box, panel, small=False):
    """Plot supplied numerical arrays; never infer a robot pose from a plot."""
    x0, y0, x1, y1 = box
    curves = panel.get("curves", [])
    samples = np.asarray(panel.get("samples", []), dtype=float).reshape(-1, 2)
    datasets = []
    for c in curves:
        x, y = np.asarray(c["x"], float), np.asarray(c["y"], float)
        if len(x) == len(y) and len(x):
            datasets.append(np.column_stack([x, y]))
    if len(samples):
        datasets.append(samples)
    data = np.concatenate(datasets) if datasets else np.array([[-1., -1.], [1., 1.]])
    data = data[np.isfinite(data).all(axis=1)]
    if len(data) == 0:
        data = np.array([[-1., -1.], [1., 1.]])

    def limits(axis, name):
        if name in panel:
            lo, hi = panel[name]
        else:
            lo, hi = np.min(data[:, axis]), np.max(data[:, axis])
            pad = max((hi - lo) * .08, .05)
            lo, hi = lo - pad, hi + pad
        if hi <= lo:
            hi = lo + 1
        return float(lo), float(hi)

    xmin, xmax = limits(0, "xlim")
    ymin, ymax = limits(1, "ylim")
    left, top, right, bottom = x0 + 46, y0 + 30, x1 - 14, y1 - (28 if small else 45)
    draw.text((x0 + 12, y0 + 5), short(panel.get("title", ""), 54), fill=INK, font=FONTS[13])

    def xy(x, y):
        return (left + min(1., max(0., (x - xmin) / (xmax - xmin))) * (right - left),
                bottom - min(1., max(0., (y - ymin) / (ymax - ymin))) * (bottom - top))

    for a in np.linspace(0, 1, 5):
        xx, yy = left + a * (right - left), bottom - a * (bottom - top)
        draw.line((xx, top, xx, bottom), fill="#e1e7eb")
        draw.line((left, yy, right, yy), fill="#e1e7eb")
        if not small or a in (0, .5, 1):
            draw.text((xx - 12, bottom + 3), f"{xmin + a*(xmax-xmin):.2g}", fill=MUTED, font=FONTS[11])
            draw.text((x0 + 3, yy - 6), f"{ymin + a*(ymax-ymin):.2g}", fill=MUTED, font=FONTS[11])
    draw.rectangle((left, top, right, bottom), outline="#afbdc7")
    for x, y in samples[:1600]:
        if np.isfinite(x + y) and xmin <= x <= xmax and ymin <= y <= ymax:
            xx, yy = xy(x, y)
            draw.ellipse((xx - 1, yy - 1, xx + 1, yy + 1), fill=SAMPLE_COLOR)
    for i, curve in enumerate(curves):
        points = [(float(x), float(y)) for x, y in zip(curve["x"], curve["y"]) if np.isfinite(x + y)]
        color = curve_color(curve, i)
        mapped = [xy(x, y) for x, y in points]
        if len(mapped) > 1:
            draw.line(mapped, fill=color, width=2)
        if mapped and (small or curve.get("markers")):
            xx, yy = mapped[-1]
            draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=color)
    if "landmark_estimates" in panel:
        # Truth markers are evaluation overlays; they are never estimator input.
        for x, y in panel.get("landmark_truth", []):
            xx, yy = xy(x, y)
            draw.line((xx-4, yy-4, xx+4, yy+4), fill=TRUTH_COLOR, width=2)
            draw.line((xx-4, yy+4, xx+4, yy-4), fill=TRUTH_COLOR, width=2)
        estimates = panel["landmark_estimates"]
        covariances = panel.get("landmark_covariances", [])
        pairs = list(zip(estimates, covariances))
        if "pose_estimate" in panel:
            pairs.append((panel["pose_estimate"][:2], panel["pose_covariance"]))
        for center, covariance in pairs:
            values, vectors = np.linalg.eigh(np.asarray(covariance))
            angles = np.linspace(0, 2*np.pi, 45)
            # 95% marginal confidence ellipse in two dimensions: chi2(2)=5.991.
            shape = vectors @ np.diag(np.sqrt(5.991 * np.maximum(values, 0)))
            ellipse = np.asarray(center)[:, None] + shape @ np.array([np.cos(angles), np.sin(angles)])
            draw.line([xy(x, y) for x, y in ellipse.T], fill=COLORS[3], width=1)
        for x, y in estimates:
            xx, yy = xy(x, y)
            draw.ellipse((xx-3, yy-3, xx+3, yy+3), outline=COLORS[3], width=2)
    if not small:
        draw.text((right - 140, bottom + 16), short(panel.get("xlabel", ""), 24), fill=MUTED, font=FONTS[11])
        draw.text((left + 5, top + 3), short(panel.get("ylabel", ""), 28), fill=MUTED, font=FONTS[11])
        draw.text((x0 + 12, y1 - 14), "Full color legend below", fill=MUTED, font=FONTS[11])


class Dashboard:
    width, height = 1000, 970

    def __init__(self, topic, kind, duration):
        self.topic, self.kind, self.duration = topic, kind, duration
        self.path, self.targets = [], []
        self.equation_art = equation_image(topic.page)
        self.legend_frames_checked = 0
        self.color_legends = {}

    def frame(self, rgb, state, payload, t):
        im = Image.new("RGB", (self.width, self.height), BG)
        d = ImageDraw.Draw(im)
        name = "CRAZYFLIE 2.0" if self.kind == "drone" else "CLEARPATH HUSKY"
        d.rectangle((0, 0, 1000, 80), fill=INK)
        d.text((20, 10), f"LECTURE 6  /  PDF PAGE {self.topic.page}  /  {name}", fill="#9de1e2", font=FONTS[13])
        d.text((20, 34), short(self.topic.title, 70), fill="white", font=TITLE_FONT)
        d.rounded_rectangle((14, 91, 489, 675), radius=8, fill="white")
        d.rounded_rectangle((503, 91, 986, 675), radius=8, fill="white")
        d.text((26, 101), "PYBULLET CAMERA  ·  physical URDF dynamics", fill=INK, font=FONTS[13])
        im.paste(Image.fromarray(rgb).resize((451, 292)), (26, 126))
        p = np.asarray(state["position"])
        v = np.asarray(state["velocity"])
        yaw = float(state["yaw"])
        d.text((27, 424), f"t = {t:4.1f} / {self.duration:g} s    |    speed = {np.linalg.norm(v):.2f} m/s", fill=INK, font=FONTS[13])
        d.text((27, 444), f"position ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f}) m   yaw {math.degrees(yaw):+.0f}°", fill=MUTED, font=FONTS[12])
        self.path.append(p[:2].copy())
        self.targets.append(np.asarray(payload["target"])[:2].copy())
        path, targets = np.array(self.path), np.array(self.targets)
        panel = {"title": "XY trajectory (m); dots = latest positions", "curves": [
            {"x": path[:, 0], "y": path[:, 1], "label": "Actual robot trajectory"},
            {"x": targets[:, 0], "y": targets[:, 1], "label": "Commanded target trajectory"}]}
        plot(d, (26, 462, 477, 670), panel, small=True)
        numerical = payload.get("panel", {})
        plot(d, (511, 102, 979, 423), numerical)
        d.text((521, 428), "LECTURE MODEL / SENSOR ESTIMATE", fill=COLORS[0], font=FONTS[12])
        im.paste(self.equation_art, (519, 446), self.equation_art)
        notes = numerical.get("notes", [])
        if not notes:
            notes = [self.topic.interpretation]
        lines = textwrap.wrap(" · ".join(str(n) for n in notes), width=66)
        for j, line in enumerate(lines[:4]):
            d.text((521, 581 + 17*j), line, fill=MUTED, font=FONTS[12])
        physical_legend = panel_legend(panel)
        if self.topic.page == 59:
            for index, tag_ids in enumerate(("0, 4", "1, 5", "2, 6", "3")):
                # The PyBullet scene cycles these four colors over tag IDs.
                color_index = (0, 1, 2, 4)[index]
                physical_legend.append(legend_entry(COLORS[color_index],
                    f"Camera columns: landmark IDs {tag_ids}"))
        self.color_legends = {
            "physical": draw_legend(d, LEGEND_BOXES[0], "COLOR LEGEND · ROBOT XY PATH / CAMERA", physical_legend),
            "model": draw_legend(d, LEGEND_BOXES[1], "COLOR LEGEND · MODEL / SLAM PLOT", panel_legend(numerical)),
        }
        self.legend_frames_checked += 1
        d.text((20, 949), "240 Hz physics  •  1× playback  •  Downloaded robot geometry  •  Numerical ensembles shown separately", fill=MUTED, font=FONTS[12])
        return im


def save_gif(frames, path: Path, fps=10, *, preserve_palette_colors=False):
    palette = []
    for r in range(0, 256, 51):
        for g in range(0, 256, 51):
            for b in range(0, 256, 51):
                palette.extend((r, g, b))
    for level in np.linspace(0, 255, 40).astype(int):
        palette.extend((level, level, level))
    ref = Image.new("P", (1, 1))
    ref.putpalette(palette)
    encoded = [im.quantize(palette=ref, dither=Image.Dither.NONE) for im in frames]
    if preserve_palette_colors:
        # Pillow's fast quantizer can move an already quantized color to a
        # neighboring palette entry. Preserve exact existing colors during
        # presentation-only refreshes so the camera pixels remain identical.
        lookup = np.full(1 << 24, -1, dtype=np.int16)
        for i, (r, g, b) in enumerate(np.asarray(palette).reshape(-1, 3)):
            lookup[(int(r) << 16) | (int(g) << 8) | int(b)] = i
        for i, im in enumerate(frames):
            rgb = np.asarray(im, dtype=np.uint32)
            exact = lookup[(rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]]
            indices = np.asarray(encoded[i]).copy()
            mask = exact >= 0
            indices[mask] = exact[mask]
            encoded[i] = Image.fromarray(indices)
            encoded[i].putpalette(palette)
    encoded[0].save(path, save_all=True, append_images=encoded[1:], duration=round(1000/fps), loop=0, optimize=False, disposal=1)
