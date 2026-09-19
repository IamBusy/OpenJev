"""Original procedural observation renderer. No hidden-state access."""

import numpy as np
from PIL import Image, ImageDraw

HEIGHT = 64
WIDTH = 192
PALETTES = {
    "default": ((44, 100, 199), (217, 65, 58)),
    "shift": ((33, 135, 184), (190, 49, 104)),
}


def render_scene(observed, seed=0, style="default"):
    obs = np.asarray(observed)
    if obs.shape != (6,) or not np.isin(obs, (-1, 0, 1)).all():
        raise ValueError("Need six observed values in {-1,0,1}")
    if style not in PALETTES:
        raise ValueError("Unknown rendering style")
    for slot in range(3):
        pair = obs[slot * 2 : slot * 2 + 2]
        if (pair == -1).any() and not (pair == -1).all():
            raise ValueError("The v0.1 renderer hides whole objects")
    rng = np.random.default_rng(seed)
    base = int(rng.integers(232, 248))
    image = Image.new("RGB", (WIDTH, HEIGHT), (base, base, min(base + 3, 255)))
    draw = ImageDraw.Draw(image)
    for i in range(3):
        # Draw-independent RNG consumption keeps appearance identical across views.
        x = 32 + i * 64 + int(rng.integers(-4, 5))
        y = 34 + int(rng.integers(-3, 4))
        radius = int(rng.integers(13, 19))
        shade = int(rng.integers(-9, 10))
        jitter = rng.integers(-8, 9, 3)
        draw.rounded_rectangle((i * 64 + 3, 3, i * 64 + 60, 60), radius=6, fill=(255, 255, 255))
        if obs[2 * i] == -1:
            color = (150 + shade, 157 + shade, 169 + shade)
            draw.rounded_rectangle((i * 64 + 9, 10, i * 64 + 54, 55), radius=4, fill=color)
            for row in range(16, 52, 8):
                draw.line((i * 64 + 13, row, i * 64 + 50, row), fill=(183, 189, 200))
        else:
            fill = tuple(
                int(x) for x in np.clip(np.array(PALETTES[style][int(obs[2 * i])]) + jitter, 0, 255)
            )
            bounds = (x - radius, y - radius, x + radius, y + radius)
            if obs[2 * i + 1] == 0:
                draw.ellipse(bounds, fill=fill)
            else:
                draw.rectangle(bounds, fill=fill)
    # Independent sensor texture makes repeated blank views distinguishable without
    # encoding a split ID, latent truth, or prior into the pixels.
    noise = rng.integers(-2, 3, size=(HEIGHT, WIDTH, 1))
    return Image.fromarray(
        np.clip(np.asarray(image).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    )
