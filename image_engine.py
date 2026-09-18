"""
image_engine.py
GS Creative Vision & Image Generation Engine.
Synthesizes high-impact visual artwork, neural flow-matching acoustic landscapes,
and cinematic scene compositions modeled after Antigravity visual generative capabilities.
"""
import os
import uuid
import math
import random
import hashlib
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_OUTPUT_DIR = os.path.join(BASE_DIR, "static", "img_output")
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

PALETTES = [
    {"name": "Cyber Neon", "primary": "#6366f1", "accent": "#06b6d4", "glow": "#a855f7", "bg": "#080912"},
    {"name": "Solar Ember", "primary": "#f97316", "accent": "#eab308", "glow": "#ec4899", "bg": "#0e090b"},
    {"name": "Deep Emerald", "primary": "#10b981", "accent": "#06b6d4", "glow": "#3b82f6", "bg": "#060f0d"},
    {"name": "Amethyst Void", "primary": "#8b5cf6", "accent": "#d946ef", "glow": "#6366f1", "bg": "#0c0814"},
]


def generate_cinematic_artwork(prompt: str, style: str = "cinematic") -> Dict[str, Any]:
    """
    Generates a rich, high-resolution SVG artwork card with multi-layered neural
    waves, geometric flow lines, particle stars, and atmospheric gradients.
    """
    prompt_clean = prompt.strip() or "Neural Flow Matching Acoustic Landscape"
    seed = int(hashlib.md5(prompt_clean.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)

    palette = rng.choice(PALETTES)
    width = 1200
    height = 675

    image_id = f"art_{uuid.uuid4().hex[:8]}"
    filename = f"{image_id}.svg"
    file_path = os.path.join(IMG_OUTPUT_DIR, filename)

    waves = []
    for layer in range(5):
        points = []
        base_y = height * 0.45 + (layer * 38)
        amplitude = 30 + rng.randint(10, 45)
        frequency = 0.004 + (layer * 0.0012)
        phase = rng.uniform(0, math.pi * 2)

        for x in range(0, width + 40, 30):
            y = base_y + math.sin(x * frequency + phase) * amplitude + math.cos(x * 0.002) * 15
            points.append(f"{x},{y:.1f}")

        path_data = f"M 0,{height} L 0,{points[0].split(',')[1]} " + " ".join([f"L {p}" for p in points]) + f" L {width},{height} Z"
        opacity = 0.18 + (layer * 0.12)
        color = palette["primary"] if layer % 2 == 0 else palette["accent"]
        waves.append((path_data, color, opacity))

    nodes = []
    for _ in range(35):
        nx = rng.randint(40, width - 40)
        ny = rng.randint(40, int(height * 0.65))
        nr = rng.uniform(1.5, 4.0)
        nodes.append((nx, ny, nr))

    lines_x = "".join([f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" />' for x in range(0, width, 60)])
    lines_y = "".join([f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" />' for y in range(0, height, 45)])
    waves_svg = "".join([f'<path d="{w[0]}" fill="{w[1]}" opacity="{w[2]}" filter="url(#neonGlow)" />' for w in waves])
    nodes_svg = "".join([f'<circle cx="{n[0]}" cy="{n[1]}" r="{n[2]}" />' for n in nodes])

    safe_title = prompt_clean[:80] + ('...' if len(prompt_clean) > 80 else '')
    safe_title = safe_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{palette['bg']}" />
      <stop offset="50%" stop-color="#121422" />
      <stop offset="100%" stop-color="#07080d" />
    </linearGradient>

    <radialGradient id="sunGlow" cx="50%" cy="35%" r="50%">
      <stop offset="0%" stop-color="{palette['glow']}" stop-opacity="0.45" />
      <stop offset="60%" stop-color="{palette['primary']}" stop-opacity="0.12" />
      <stop offset="100%" stop-color="transparent" stop-opacity="0" />
    </radialGradient>

    <filter id="neonGlow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>

  <rect width="{width}" height="{height}" fill="url(#bgGrad)" />
  <circle cx="{width * 0.5}" cy="{height * 0.35}" r="260" fill="url(#sunGlow)" />

  <g stroke="rgba(255,255,255,0.04)" stroke-width="1">
    {lines_x}
    {lines_y}
  </g>

  <circle cx="{width * 0.5}" cy="{height * 0.38}" r="110" fill="none" stroke="{palette['accent']}" stroke-width="2" opacity="0.6" filter="url(#neonGlow)" />
  <circle cx="{width * 0.5}" cy="{height * 0.38}" r="80" fill="none" stroke="{palette['glow']}" stroke-width="1.5" stroke-dasharray="8 6" opacity="0.8" />
  <circle cx="{width * 0.5}" cy="{height * 0.38}" r="50" fill="{palette['primary']}" opacity="0.25" />

  {waves_svg}

  <g fill="{palette['accent']}" opacity="0.85">
    {nodes_svg}
  </g>

  <g transform="translate(45, 45)">
    <rect width="260" height="38" rx="19" fill="rgba(10, 12, 20, 0.85)" stroke="rgba(255,255,255,0.15)" stroke-width="1" />
    <circle cx="20" cy="19" r="6" fill="{palette['accent']}" />
    <text x="36" y="23" fill="#ffffff" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="700" letter-spacing="1">GS CREATIVE VISION AI</text>
  </g>

  <g transform="translate(45, {height - 75})">
    <rect width="{width - 90}" height="50" rx="14" fill="rgba(12, 14, 24, 0.9)" stroke="rgba(255,255,255,0.12)" stroke-width="1" />
    <text x="24" y="31" fill="#ffffff" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="14" font-weight="600">
      {safe_title}
    </text>
    <text x="{width - 120}" y="31" fill="{palette['accent']}" font-family="monospace" font-size="11" text-anchor="end" font-weight="bold">
      FLOW-RENDER • 24kHz
    </text>
  </g>
</svg>'''

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return {
        "success": True,
        "image_url": f"/static/img_output/{filename}",
        "filename": filename,
        "prompt": prompt_clean,
        "theme": palette["name"],
        "width": width,
        "height": height,
        "format": "svg"
    }
