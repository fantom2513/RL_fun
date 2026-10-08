"""Anti-aliased car sprites: drawn once at high resolution, smoothed down, then rotated and cached.

Orientation: in an unrotated sprite the nose points to +x (right) and the car centre is the
centre of the surface. `length_px` is the on-screen length of the car from nose to tail; the
surface is a little larger to leave room for the wings' end plates and the soft shadow.
"""

from __future__ import annotations

import math

import pygame

from rl_fun.racing.style import Color

CAR_VARIANTS = ("alive", "dead", "leader")
SPRITE_SUPERSAMPLE = 4  # sprites are drawn this much larger, then smoothscaled down
ANGLE_STEP = 3  # degrees; headings are quantised to this for the rotation cache
MAX_CACHED_ROTATIONS = 4096
_CAR_LENGTH_M = 4.5

_BACKGROUND: Color = (40, 46, 58)  # RGB of the transparent pixels, so smoothing fringes stay dark

_PALETTES: dict[str, dict[str, Color]] = {
    "alive": {
        "top": (234, 238, 243),
        "side": (160, 170, 186),
        "wing": (88, 96, 110),
        "wing_light": (156, 164, 176),
        "wheel": (20, 22, 28),
        "rim": (96, 104, 118),
        "cockpit": (34, 38, 50),
        "outline": (40, 46, 58),
        "helmet": (226, 66, 70),
        "stripe": (200, 208, 218),
    },
    "leader": {
        "top": (255, 255, 255),
        "side": (206, 214, 226),
        "wing": (84, 92, 106),
        "wing_light": (170, 178, 190),
        "wheel": (16, 18, 24),
        "rim": (110, 118, 132),
        "cockpit": (30, 34, 46),
        "outline": (34, 40, 52),
        "helmet": (240, 70, 74),
        "stripe": (226, 232, 240),
    },
    "dead": {
        "top": (168, 178, 190),
        "side": (124, 136, 150),
        "wing": (80, 92, 106),
        "wing_light": (104, 116, 130),
        "wheel": (54, 64, 76),
        "rim": (96, 108, 120),
        "cockpit": (72, 84, 98),
        "outline": (70, 82, 96),
        "helmet": (132, 120, 126),
        "stripe": (150, 160, 172),
    },
}
_DEAD_OPACITY = 0.62

# Half outline of the body in the car frame (metres, nose +x, y to the left), nose to tail.
_BODY_HALF = (
    (2.25, 0.0),
    (2.00, 0.05),
    (1.70, 0.10),
    (1.20, 0.15),
    (0.95, 0.19),
    (0.85, 0.44),
    (0.70, 0.60),
    (0.05, 0.63),
    (-0.30, 0.48),
    (-0.60, 0.30),
    (-1.15, 0.25),
    (-1.85, 0.28),
)

_sprites: dict[tuple[str, int], pygame.Surface] = {}
_rotated: dict[tuple[str, int, int], pygame.Surface] = {}


def _even(value: float) -> int:
    return int(math.ceil(value / 2.0)) * 2


def sprite_size(length_px: int) -> tuple[int, int]:
    """Size of the (unrotated) sprite surface for a car `length_px` long."""
    return _even(length_px * 1.2 + 2), _even(length_px * 0.66 + 2)


def build_car_sprite(variant: str, length_px: int) -> pygame.Surface:
    """SRCALPHA sprite of one car variant ("alive", "dead" or "leader"), cached per length."""
    if variant not in CAR_VARIANTS:
        raise ValueError(f"unknown car variant {variant!r}")
    length_px = max(6, int(length_px))
    key = (variant, length_px)
    sprite = _sprites.get(key)
    if sprite is None:
        sprite = _render_sprite(variant, length_px)
        _sprites[key] = sprite
    return sprite


def rotated_car_sprite(variant: str, length_px: int, heading: float) -> pygame.Surface:
    """The sprite rotated to `heading` (radians, counter-clockwise on screen), cached per 3 deg."""
    bucket = round(math.degrees(heading) / ANGLE_STEP) % (360 // ANGLE_STEP)
    key = (variant, max(6, int(length_px)), bucket)
    rotated = _rotated.get(key)
    if rotated is None:
        if len(_rotated) >= MAX_CACHED_ROTATIONS:
            _rotated.clear()
        base = build_car_sprite(variant, length_px)
        rotated = pygame.transform.rotozoom(base, bucket * ANGLE_STEP, 1.0)
        _rotated[key] = rotated
    return rotated


def _render_sprite(variant: str, length_px: int) -> pygame.Surface:
    palette = _PALETTES[variant]
    width, height = sprite_size(length_px)
    k = SPRITE_SUPERSAMPLE
    big = pygame.Surface((width * k, height * k), pygame.SRCALPHA)
    big.fill((*_BACKGROUND, 0))
    scale = k * length_px / _CAR_LENGTH_M  # big pixels per metre
    cx, cy = width * k / 2, height * k / 2

    def point(x: float, y: float) -> tuple[float, float]:
        return cx + x * scale, cy - y * scale

    def polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [point(x, y) for x, y in points]

    def box(x0: float, x1: float, y0: float, y1: float) -> pygame.Rect:
        left, top = point(x0, y1)
        right, bottom = point(x1, y0)
        return pygame.Rect(round(left), round(top), round(right - left), round(bottom - top))

    outline_width = max(1, round(0.045 * scale))
    outline = palette["outline"]

    def rounded(rect: pygame.Rect, color: Color, radius: float = 0.07) -> None:
        pygame.draw.rect(big, color, rect, border_radius=max(1, round(radius * scale)))

    _draw_shadow(big, box)

    # Suspension arms, then wheels (tyre, lighter rim highlight, dark outline).
    for x_axle in (1.12, -1.35):
        for side in (1, -1):
            pygame.draw.line(
                big, palette["wing"], point(x_axle, 0.25 * side), point(x_axle, 0.70 * side),
                max(1, round(0.05 * scale)),
            )
    for x0, x1 in ((0.80, 1.45), (-1.75, -0.95)):
        for sign in (1, -1):
            y0, y1 = sorted((0.62 * sign, 1.02 * sign))
            if x0 < 0:
                y0, y1 = sorted((0.64 * sign, 1.06 * sign))
            rect = box(x0, x1, y0, y1)
            rounded(rect, palette["wheel"], 0.10)
            inset = rect.inflate(-round(0.14 * scale), -round(0.14 * scale))
            pygame.draw.rect(
                big, palette["rim"], inset, max(1, round(0.035 * scale)),
                border_radius=max(1, round(0.05 * scale)),
            )
            pygame.draw.rect(
                big, outline, rect, outline_width, border_radius=max(1, round(0.10 * scale))
            )

    # Rear wing: main plane, upper flap highlight, end plates.
    rounded(box(-2.25, -1.78, -0.90, 0.90), palette["wing"], 0.05)
    pygame.draw.rect(big, palette["wing_light"], box(-2.18, -2.08, -0.84, 0.84))
    for sign in (1, -1):
        y0, y1 = sorted((0.86 * sign, 0.97 * sign))
        pygame.draw.rect(big, outline, box(-2.28, -1.62, y0, y1))

    # Front wing: main plane, flap highlight, end plates.
    rounded(box(1.58, 2.04, -1.04, 1.04), palette["wing"], 0.05)
    pygame.draw.rect(big, palette["wing_light"], box(1.66, 1.74, -0.98, 0.98))
    for sign in (1, -1):
        y0, y1 = sorted((1.00 * sign, 1.11 * sign))
        pygame.draw.rect(big, outline, box(1.54, 2.08, y0, y1))

    # Body: darker flanks as the base, a lighter narrower top on it, then outline.
    flank = [(x, y) for x, y in _BODY_HALF] + [(x, -y) for x, y in reversed(_BODY_HALF) if y]
    pygame.draw.polygon(big, palette["side"], polygon(flank))
    top_half = [(x * 0.99, y * 0.64) for x, y in _BODY_HALF[1:]]
    top = [*top_half, *[(x, -y) for x, y in reversed(top_half)]]
    pygame.draw.polygon(big, palette["top"], polygon(top))
    pygame.draw.polygon(big, outline, polygon(flank), outline_width)
    # Engine cover stripe and nose highlight.
    pygame.draw.polygon(
        big,
        palette["stripe"],
        polygon([(-0.40, 0.07), (-1.70, 0.08), (-1.70, -0.08), (-0.40, -0.07)]),
    )
    pygame.draw.polygon(
        big, palette["stripe"], polygon([(2.05, 0.0), (1.30, 0.045), (1.30, -0.045)])
    )

    # Cockpit opening, helmet and the halo arch.
    pygame.draw.ellipse(big, palette["cockpit"], box(-0.32, 0.58, -0.23, 0.23))
    pygame.draw.circle(big, palette["helmet"], point(0.08, 0.0), max(2, 0.13 * scale))
    helmet_radius = max(2, 0.13 * scale)
    helmet_edge = max(1, outline_width // 2)
    pygame.draw.circle(big, outline, point(0.08, 0.0), helmet_radius, helmet_edge)
    halo = [point(0.18, 0.25), point(0.52, 0.12), point(0.52, -0.12), point(0.18, -0.25)]
    pygame.draw.lines(big, palette["cockpit"], False, halo, max(2, round(0.07 * scale)))

    sprite = pygame.transform.smoothscale(big, (width, height))
    if variant == "dead":
        fade = (255, 255, 255, round(255 * _DEAD_OPACITY))
        sprite.fill(fade, special_flags=pygame.BLEND_RGBA_MULT)
    return sprite


def _draw_shadow(big: pygame.Surface, box) -> None:  # noqa: ANN001
    """Soft drop shadow: nested ellipses with growing opacity, hugging the car (alpha <= 30)."""
    for grow, alpha in ((0.20, 8), (0.13, 14), (0.06, 22), (0.0, 30)):
        rect = box(-1.95 - grow, 1.95 + grow, -0.80 - grow, 0.80 + grow)
        rect.move_ip(0, round(0.05 * big.get_height() / 20))
        pygame.draw.ellipse(big, (0, 0, 0, alpha), rect)