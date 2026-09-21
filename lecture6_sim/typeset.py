"""Build local LaTeX equation artwork once; GIF rendering just pastes it.

MathJax lays out true TeX fractions, AMS matrices, derivatives and symbols.
The SVG contains glyph paths, so no browser or TeX installation is needed to
display it. CairoSVG rasterizes at 2x resolution for antialiased GIF lettering.
"""
from __future__ import annotations

from functools import lru_cache
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image

from .equations import LATEX_EQUATIONS

ASSET_DIR = Path(__file__).resolve().parent / "math_assets"
EQUATION_SIZE = (456, 118)
RENDERING_VERSION = "latex-v1"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def build_equations():
    import cairosvg
    script = Path(__file__).with_name("typeset_math.cjs")
    result = subprocess.run(["node", str(script)], input=json.dumps(LATEX_EQUATIONS),
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"LaTeX typesetting failed:\n{result.stderr}")
    compiled = json.loads(result.stdout)
    ASSET_DIR.mkdir(exist_ok=True)
    manifest = {}
    for page, svgs in compiled.items():
        roots = [ET.fromstring(svg.replace("currentColor", "#203348")) for svg in svgs]
        boxes = [list(map(float, root.attrib["viewBox"].split())) for root in roots]
        # MathJax's em is 1000 SVG units. Start at a readable 24-pixel em.
        em = min(24., (EQUATION_SIZE[0]-8) * 1000 / max(box[2] for box in boxes))
        gap = 8.
        em = min(em, (EQUATION_SIZE[1]-8-gap*(len(roots)-1)) * 1000 / sum(box[3] for box in boxes))
        if em < 15:
            raise ValueError(f"Page {page}: formula too long for readable type ({em:.1f}px); split its LaTeX lines")
        height = sum(box[3]*em/1000 for box in boxes) + gap*(len(roots)-1)
        combined = ET.Element(f"{{{NS}}}svg", width=str(EQUATION_SIZE[0]), height=str(EQUATION_SIZE[1]),
                              viewBox=f"0 0 {EQUATION_SIZE[0]} {EQUATION_SIZE[1]}")
        y = (EQUATION_SIZE[1]-height)/2
        for root, box in zip(roots, boxes):
            w, h = box[2]*em/1000, box[3]*em/1000
            root.attrib.update(x=str((EQUATION_SIZE[0]-w)/2), y=str(y), width=str(w), height=str(h))
            root.attrib.pop("style", None)
            combined.append(root)
            y += h + gap
        svg = ET.tostring(combined, encoding="utf-8", xml_declaration=True)
        (ASSET_DIR/f"s{page}.svg").write_bytes(svg)
        raster = cairosvg.svg2png(bytestring=svg, output_width=EQUATION_SIZE[0]*2, output_height=EQUATION_SIZE[1]*2)
        image = Image.open(io.BytesIO(raster)).convert("RGBA").resize(EQUATION_SIZE, Image.Resampling.LANCZOS)
        image.save(ASSET_DIR/f"s{page}.png")
        manifest[page] = {"latex": LATEX_EQUATIONS[int(page)], "em_pixels": em,
                          "size": list(EQUATION_SIZE), "renderer": "MathJax 3.2.2 SVG / CairoSVG",
                          "version": RENDERING_VERSION}
    (ASSET_DIR/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Typeset {len(manifest)} equation blocks; minimum type size {min(m['em_pixels'] for m in manifest.values()):.1f}px")
    equation_image.cache_clear()


@lru_cache(maxsize=36)
def equation_image(page):
    path = ASSET_DIR/f"s{page}.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing typeset equation {path}; run .venv/bin/python -m lecture6_sim.typeset")
    with Image.open(path) as image:
        return image.convert("RGBA")


if __name__ == "__main__":
    build_equations()
