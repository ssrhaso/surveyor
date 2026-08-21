"""Render the animated SURVEYOR overview used in the README.

Four acts: a title card, an animated run of the mechanism (the drafted plan
drawn as a dashed line inside a shaded tolerance tube of half-width tau;
reality traces a solid path inside it, waypoints are served free while the
path stays in the tube, and the one excursion beyond tau triggers a
re-draft anchored where reality actually is), a calibration/routing card,
and a results card. Colors are the paper's figure palette so the README and
the paper read as one hand.

Run from the repository root:

    python docs/media/make_surveyor_overview.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

from PIL import Image, ImageColor, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 675
OUT_W, OUT_H = 1080, 570  # README shows the GIF at width=900
SCALE = 2
SLOWDOWN = 1.45  # global tempo: multiply every frame duration

# ---- the paper's figure palette (fig:method provenance coding) ----
PAPER = "#FFFFFF"
INK = "#2B2D33"        # echInk
MUTED = "#8A8F98"      # echLat: drafter / neutral
BORDER = "#BFC2C4"
BLUE = "#3F6FB0"       # echEnc: frozen planner stack / reality
TEAL = "#4C907E"       # echHead: accept
AMBER = "#B07D10"      # survAmber: draft / contribution (text-weight variant)
AMBER_FILL = "#D9A441" # echVQ: draft fills
CORAL = "#C06A5B"      # echDec: divergence / re-draft


def s(value: float) -> int:
    return round(value * SCALE)


_FONT_DIR = Path(__file__).with_name("fonts")


def _font_candidates(bold: bool) -> list[str]:
    # TeX Gyre Heros (vendored beside this script) is the free Helvetica:
    # URW Nimbus Sans, metrically Helvetica-compatible, GUST license. The
    # system faces below are fallbacks only.
    return (
        [
            str(_FONT_DIR / "texgyreheros-bold.otf"),
            "C:/Windows/Fonts/helvetica-bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            str(_FONT_DIR / "texgyreheros-regular.otf"),
            "C:/Windows/Fonts/helvetica.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )


_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def get_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _FONT_CACHE:
        for candidate in _font_candidates(bold):
            if Path(candidate).exists():
                _FONT_CACHE[key] = ImageFont.truetype(candidate, s(size))
                break
        else:
            raise FileNotFoundError("No suitable font found")
    return _FONT_CACHE[key]


def rgba(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    return (*ImageColor.getrgb(color), round(255 * max(0.0, min(1.0, alpha))))


def blend(color: str, toward: str, t: float) -> str:
    a, b = ImageColor.getrgb(color), ImageColor.getrgb(toward)
    return "#{:02X}{:02X}{:02X}".format(
        *(round(x + (y - x) * t) for x, y in zip(a, b))
    )


def tint(color: str, t: float) -> str:
    """Paper-mixed tint, like TikZ's color!<pct>."""
    return blend(PAPER, color, t)


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    color: str = INK,
    *,
    bold: bool = False,
    anchor: str = "la",
    alpha: float = 1.0,
) -> None:
    if alpha <= 0:
        return
    draw.text(
        (s(xy[0]), s(xy[1])),
        value,
        font=get_font(size, bold=bold),
        fill=rgba(color, alpha),
        anchor=anchor,
    )


def alpha_for(visible: float, item: int) -> float:
    return max(0.0, min(1.0, visible - item))


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def canvas(section: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (WIDTH * SCALE, HEIGHT * SCALE), rgba(PAPER))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (s(27), s(18), s(WIDTH - 27), s(HEIGHT - 18)),
        radius=s(11),
        outline=rgba(BORDER),
        width=s(1.3),
    )
    text(draw, (58, 50), f"SURVEYOR  /  {section}", 14, MUTED, anchor="lm")
    return image, draw


def heading(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    text(draw, (WIDTH / 2, 104), title, 34, INK, bold=True, anchor="mm")
    text(draw, (WIDTH / 2, 146), subtitle, 19, MUTED, anchor="mm")


def finish(image: Image.Image) -> Image.Image:
    return image.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS).convert("RGB")


def dashed_path(draw, pts, color, width=1.6, dash=9.0, gap=6.5, alpha=1.0):
    """Dashed polyline in unscaled coordinates."""
    dist_carry = 0.0
    pen_down = True
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        d = 0.0
        while d < seg:
            span = (dash if pen_down else gap) - dist_carry
            step = min(span, seg - d)
            if pen_down:
                draw.line(
                    (s(x0 + ux * d), s(y0 + uy * d),
                     s(x0 + ux * (d + step)), s(y0 + uy * (d + step))),
                    fill=rgba(color, alpha), width=s(width),
                )
            d += step
            dist_carry += step
            if dist_carry >= (dash if pen_down else gap) - 1e-6:
                dist_carry = 0.0
                pen_down = not pen_down


# ======================================================================
# Act 1 / title card
# ======================================================================

def title_slide(visible: float) -> Image.Image:
    image, draw = canvas("OVERVIEW")
    text(draw, (WIDTH / 2, 195), "SURVEYOR", 62, INK, bold=True, anchor="mm")
    text(
        draw,
        (WIDTH / 2, 257),
        "Certified Speculative Plan Consumption for Latent World-Model Planning",
        23,
        MUTED,
        anchor="mm",
        alpha=alpha_for(visible, 0),
    )
    steps = [
        ("draft once", AMBER),
        ("verify against reality", TEAL),
        ("re-draft on divergence", CORAL),
    ]
    sizes = [get_font(26, bold=True).getlength(label) / SCALE
             for label, _ in steps]
    gap = 72
    total_w = sum(sizes) + gap * 2
    x = WIDTH / 2 - total_w / 2
    for index, ((label, color), w) in enumerate(zip(steps, sizes)):
        a = alpha_for(visible, index + 1)
        text(draw, (x, 380), label, 26, color, bold=True, anchor="lm", alpha=a)
        if index < 2:
            text(draw, (x + w + gap / 2, 380), "\u2192", 26, MUTED,
                 anchor="mm", alpha=alpha_for(visible, index + 2))
        x += w + gap
    text(
        draw,
        (WIDTH / 2, 470),
        "zero learned parameters  \u00b7  constants measured, not tuned",
        21,
        MUTED,
        anchor="mm",
        alpha=alpha_for(visible, 4),
    )
    return image


# ======================================================================
# Act 2 / the mechanism: a tolerance tube around the drafted plan
# ======================================================================

PLAN_Y = 385          # nominal plan height
TUBE = 50             # tau, in pixels: tube half-width
X_START = 150
GOAL_X = 1128
BXS = [300, 442, 584, 726, 868, 1010]   # boundaries b1..b6
REJOIN_X = 700        # after the re-draft, plan eases back to PLAN_Y by here
DIVE_X0, DIVE_X1 = 470, 584             # reality dives out of the tube here
DIVE = 96             # how far below PLAN_Y the excursion lands (> TUBE)


def plan1_y(x: float) -> float:
    return PLAN_Y


def plan2_y(x: float) -> float:
    """Second plan: anchored at reality's excursion, easing back up."""
    if x <= BXS[2]:
        return PLAN_Y + DIVE
    if x >= REJOIN_X:
        return PLAN_Y
    t = ease((x - BXS[2]) / (REJOIN_X - BXS[2]))
    return PLAN_Y + DIVE * (1 - t)


def wobble(x: float) -> float:
    return 10.0 * math.sin(x / 34.0) + 4.5 * math.sin(x / 13.0)


def real_y(x: float) -> float:
    """Reality's path across the whole episode."""
    # settle onto the axis over the last stretch so the dot lands
    # dead-center on the goal marker
    settle = ease(min(1.0, max(0.0, (GOAL_X - x) / 80.0)))
    if x < DIVE_X0:
        return PLAN_Y + wobble(x) * settle
    if x <= DIVE_X1:
        t = ease((x - DIVE_X0) / (DIVE_X1 - DIVE_X0))
        return PLAN_Y + wobble(x) * (1 - t) + DIVE * t
    return plan2_y(x) + wobble(x) * min(1.0, (x - DIVE_X1) / 90.0) * settle


def tube_polygon(draw, y_of, x0, x1, alpha=1.0):
    xs = [x0 + i * (x1 - x0) / 40 for i in range(41)]
    top = [(s(x), s(y_of(x) - TUBE)) for x in xs]
    bot = [(s(x), s(y_of(x) + TUBE)) for x in reversed(xs)]
    draw.polygon(top + bot, fill=rgba(tint(TEAL, 0.10), alpha))
    for pts in (top, list(reversed(bot))):
        draw.line(pts, fill=rgba(tint(TEAL, 0.45), alpha), width=s(1.1))


def _diamond(draw, cx, cy, r, color, fill, width=1.6, alpha=1.0):
    pts = [(s(cx), s(cy - r)), (s(cx + r), s(cy)),
           (s(cx), s(cy + r)), (s(cx - r), s(cy))]
    draw.polygon(pts, fill=rgba(fill, alpha), outline=rgba(color, alpha),
                 width=s(width))


def _circle(draw, cx, cy, r, outline, fill=None, width=1.6, alpha=1.0):
    box = (s(cx - r), s(cy - r), s(cx + r), s(cy + r))
    draw.ellipse(box, fill=rgba(fill, alpha) if fill else None,
                 outline=rgba(outline, alpha), width=s(width))


class MechState:
    def __init__(self) -> None:
        self.pills: list[str] = []          # "draft" | "free", one per serve
        self.consumed: dict[int, bool] = {} # boundary idx -> served free
        self.plan = 0                       # 0 none, 1 first, 2 second
        self.caption = ""
        self.caption_color = MUTED
        self.show_tau = False


DOT_R, DOT_PITCH = 9, 26


def draw_pills(draw, st: MechState) -> None:
    """Per-serve cost tally, top right: amber dot = paid draft, teal = free."""
    x1 = WIDTH - 74
    y = 112
    text(draw, (x1, 82), "cost per served target", 14, MUTED, anchor="rm")
    n = len(st.pills)
    for i, kind in enumerate(st.pills):
        cx = x1 - DOT_R - (n - 1 - i) * DOT_PITCH
        color = AMBER_FILL if kind == "draft" else TEAL
        _circle(draw, cx, y, DOT_R, blend(color, INK, 0.15),
                tint(color, 0.55 if kind == "draft" else 0.4), 1.6)
    drafts = st.pills.count("draft")
    if n:
        text(draw, (x1, y + 26), f"{drafts} paid · {n - drafts} free",
             15, MUTED, anchor="rm")


def mech_frame(
    st: MechState,
    dot_x: float,
    *,
    trail_to: float | None = None,
    badge_at: int | None = None,
    badge_ok: bool = True,
    badge_pulse: float = 0.0,
    draft_pulse: float = 0.0,
    plan2_alpha: float = 1.0,
    goal_reached: bool = False,
    footer_alpha: float = 0.0,
) -> Image.Image:
    image, draw = canvas("MECHANISM")
    heading(draw, "Draft once. Verify against reality.",
            "The drafted plan carries a measured tolerance \u03c4; "
            "reality is verified against it at every replan boundary.")
    draw_pills(draw, st)

    # episode axis: a quiet full-width ground line from start to goal,
    # drawn first so everything else sits on one shared reference
    draw.line((s(X_START - 20), s(PLAN_Y), s(GOAL_X), s(PLAN_Y)),
              fill=rgba(tint(INK, 0.16)), width=s(1.1))

    # goal, on the axis, label beside it
    goal_col = TEAL if goal_reached else tint(INK, 0.45)
    _circle(draw, GOAL_X, PLAN_Y, 10, goal_col,
            tint(TEAL, 0.18) if goal_reached else PAPER, 2.0)
    text(draw, (GOAL_X + 20, PLAN_Y + 1), "goal", 16,
         TEAL if goal_reached else MUTED, anchor="lm", bold=goal_reached)

    # tubes + dashed plan lines
    if st.plan >= 1:
        tube_polygon(draw, plan1_y, X_START, BXS[2])
        dashed_path(draw, [(X_START, PLAN_Y), (BXS[2], PLAN_Y)],
                    blend(AMBER_FILL, INK, 0.1), 1.7)
    if st.plan >= 2:
        a = plan2_alpha
        # start a few px inside the first tube so the two bands join
        # without a white seam
        tube_polygon(draw, plan2_y, BXS[2] - 8, BXS[5] + 40, alpha=a)
        pts = [(BXS[2] + i * (BXS[5] + 40 - BXS[2]) / 30, 0) for i in range(31)]
        pts = [(x, plan2_y(x)) for x, _ in pts]
        dashed_path(draw, pts, blend(AMBER_FILL, INK, 0.1), 1.7, alpha=a)

    # tau bracket, outside the tube on the left so it crosses nothing
    if st.show_tau and st.plan >= 1:
        bx = X_START - 20
        ink = blend(TEAL, INK, 0.2)
        draw.line((s(bx), s(PLAN_Y), s(bx), s(PLAN_Y - TUBE)),
                  fill=rgba(ink), width=s(1.6))
        for yy in (PLAN_Y, PLAN_Y - TUBE):
            draw.line((s(bx - 5), s(yy), s(bx + 5), s(yy)),
                      fill=rgba(ink), width=s(1.6))
        text(draw, (bx - 12, PLAN_Y - TUBE / 2), "\u03c4", 20, ink,
             bold=True, anchor="rm")

    # waypoints on their plan
    for i, bx in enumerate(BXS):
        if st.plan >= 2 and i >= 3:
            wy = plan2_y(bx)
            a = plan2_alpha
        elif st.plan >= 1 and i <= 2:
            wy = PLAN_Y
            a = 1.0
        else:
            continue
        if st.consumed.get(i):
            _diamond(draw, bx, wy, 9, TEAL, tint(TEAL, 0.35), 1.8, alpha=a)
        elif i == 2 and st.plan >= 2:
            _diamond(draw, bx, PLAN_Y, 9, blend(CORAL, INK, 0.1),
                     tint(CORAL, 0.25), 1.8)  # the rejected waypoint stays put
        else:
            _diamond(draw, bx, wy, 9, blend(AMBER_FILL, INK, 0.2),
                     tint(AMBER_FILL, 0.4), 1.8, alpha=a)

    # reality trail
    end = trail_to if trail_to is not None else dot_x
    if end > X_START + 2:
        xs = [X_START + i * (end - X_START) / 64 for i in range(65)]
        draw.line([(s(x), s(real_y(x))) for x in xs],
                  fill=rgba(blend(BLUE, INK, 0.05), 0.9), width=s(2.6))

    # draft pulse ring
    if draft_pulse > 0:
        rr = 14 + 30 * ease(draft_pulse)
        _circle(draw, dot_x, real_y(dot_x), rr, AMBER_FILL, None, 2.2,
                alpha=1.0 - draft_pulse * 0.85)

    # the achieved latent
    dy = real_y(dot_x)
    _circle(draw, dot_x, dy, 10.5, blend(BLUE, INK, 0.15), tint(BLUE, 0.85), 2.0)
    text(draw, (dot_x, dy + 1), "z", 15, PAPER, bold=True, anchor="mm")

    # verify badge above the boundary; check/cross drawn by hand so the
    # glyphs render identically on every platform's Arial substitute
    if badge_at is not None and badge_pulse > 0:
        bx = BXS[badge_at]
        by = min(plan1_y(bx) if badge_at <= 2 else plan2_y(bx),
                 PLAN_Y) - TUBE - 42
        pop = 0.8 + 0.2 * ease(min(1.0, badge_pulse * 2))
        color = TEAL if badge_ok else CORAL
        label = "serve free" if badge_ok else "rel > \u03c4"
        lw = get_font(16, bold=True).getlength(label) / SCALE
        w = lw + 48
        draw.rounded_rectangle(
            (s(bx - w / 2 * pop), s(by - 15 * pop),
             s(bx + w / 2 * pop), s(by + 15 * pop)),
            radius=s(14), fill=rgba(tint(color, 0.13)),
            outline=rgba(blend(color, INK, 0.1)), width=s(1.4),
        )
        gx = bx - w / 2 + 16  # glyph center
        ink = blend(color, INK, 0.2)
        if badge_ok:
            draw.line((s(gx - 5), s(by + 1), s(gx - 1.5), s(by + 5)),
                      fill=rgba(ink), width=s(2.2))
            draw.line((s(gx - 1.5), s(by + 5), s(gx + 6), s(by - 4)),
                      fill=rgba(ink), width=s(2.2))
        else:
            draw.line((s(gx - 5), s(by - 5), s(gx + 5), s(by + 5)),
                      fill=rgba(ink), width=s(2.2))
            draw.line((s(gx - 5), s(by + 5), s(gx + 5), s(by - 5)),
                      fill=rgba(ink), width=s(2.2))
        text(draw, (gx + 12, by + 1), label, 16, ink, bold=True, anchor="lm")

    # caption band
    if st.caption:
        text(draw, (WIDTH / 2, 565), st.caption, 22, st.caption_color,
             anchor="mm", bold=True)
    if footer_alpha > 0:
        text(draw, (WIDTH / 2, 612),
             "7 targets served \u00b7 2 drafter calls \u00b7 "
             "every-step drafting would have paid 7",
             20, INK, anchor="mm", alpha=footer_alpha)
    return image


def mechanism_frames() -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    st = MechState()

    def emit(img: Image.Image, ms: int) -> None:
        frames.append(finish(img))
        durations.append(round(ms * SLOWDOWN))

    def dissolve_to(img: Image.Image, steps: int = 4, ms: int = 70) -> None:
        """Soft pixel dissolve from the last emitted frame into `img`."""
        target = finish(img)
        if frames:
            base = frames[-1]
            for i in range(1, steps):
                frames.append(Image.blend(base, target, ease(i / steps)))
                durations.append(round(ms * SLOWDOWN))
        frames.append(target)
        durations.append(round(ms * SLOWDOWN))

    # opening: bare stage, dot at start
    emit(mech_frame(st, X_START), 900)

    # draft #1: the tube and waypoints dissolve in, then the pulse rings
    st.plan = 1
    st.pills.append("draft")
    st.show_tau = True
    st.caption = "draft a block of subgoals once: one paid call, " \
                 "a plan with a measured tolerance \u03c4"
    st.caption_color = AMBER
    dissolve_to(mech_frame(st, X_START), steps=6, ms=80)
    for f in range(6):
        emit(mech_frame(st, X_START, draft_pulse=(f + 1) / 6), 95)
    emit(mech_frame(st, X_START), 1100)

    def travel(x_from: float, x_to: float, n: int = 10) -> None:
        for f in range(1, n + 1):
            t = ease(f / n)
            emit(mech_frame(st, x_from + (x_to - x_from) * t), 80)

    def verify(i: int, ok: bool) -> None:
        # badge blooms in over a dissolve, grows, holds, then dissolves
        # away before the dot moves on
        dissolve_to(mech_frame(st, BXS[i], badge_at=i, badge_ok=ok,
                               badge_pulse=0.35), steps=3, ms=70)
        emit(mech_frame(st, BXS[i], badge_at=i, badge_ok=ok,
                        badge_pulse=0.4), 105)
        # the caption swaps under its own small dissolve
        if ok:
            st.caption = ("inside the tube \u2192 next waypoint "
                          "served free, no drafter call")
            st.caption_color = TEAL
        else:
            st.caption = ("outside the tube \u2192 the plan no "
                          "longer matches reality")
            st.caption_color = CORAL
        dissolve_to(mech_frame(st, BXS[i], badge_at=i, badge_ok=ok,
                               badge_pulse=0.6), steps=3, ms=70)
        for f in (4, 5):
            emit(mech_frame(st, BXS[i], badge_at=i, badge_ok=ok,
                            badge_pulse=f / 5), 105)
        if ok:
            st.consumed[i] = True
            st.pills.append("free")
        emit(mech_frame(st, BXS[i], badge_at=i, badge_ok=ok,
                        badge_pulse=1.0), 1050)
        dissolve_to(mech_frame(st, BXS[i]), steps=4, ms=70)

    st.caption = "the executor acts; reality (the solid path) stays " \
                 "inside the tube"
    st.caption_color = MUTED
    dissolve_to(mech_frame(st, X_START), steps=3, ms=70)
    travel(X_START, BXS[0])
    verify(0, True)
    travel(BXS[0], BXS[1])
    verify(1, True)

    # divergence into b3
    st.caption = "then reality drifts: the executor cannot hold the plan"
    st.caption_color = CORAL
    dissolve_to(mech_frame(st, BXS[1]), steps=3, ms=70)
    travel(BXS[1], BXS[2], n=13)
    verify(2, False)

    # re-draft: second tube grows from reality's actual position
    st.plan = 2
    st.pills.append("draft")
    st.caption = "re-draft from the achieved state: the new plan is " \
                 "anchored to reality, one more call"
    st.caption_color = AMBER
    dissolve_to(mech_frame(st, BXS[2], plan2_alpha=0.0), steps=3, ms=70)
    for f in range(8):
        emit(mech_frame(st, BXS[2], draft_pulse=(f + 1) / 8,
                        plan2_alpha=ease((f + 1) / 8)), 95)
    emit(mech_frame(st, BXS[2]), 1100)

    # b4..b6 accepts
    travel(BXS[2], BXS[3])
    verify(3, True)
    travel(BXS[3], BXS[4])
    verify(4, True)
    travel(BXS[4], BXS[5])
    verify(5, True)

    # home to the goal
    st.caption = ""
    dissolve_to(mech_frame(st, BXS[5]), steps=3, ms=70)
    travel(BXS[5], GOAL_X, n=9)
    for f in range(1, 9):
        emit(mech_frame(st, GOAL_X, goal_reached=True,
                        footer_alpha=ease(f / 8)), 100)
    emit(mech_frame(st, GOAL_X, goal_reached=True, footer_alpha=1.0), 3400)
    return frames, durations


# ======================================================================
# Act 3 / measured constants + the router
# ======================================================================

def constants_slide(visible: float) -> Image.Image:
    image, draw = canvas("CALIBRATION  +  ROUTING")
    heading(draw, "Measured, not tuned",
            "Every constant is read off the frozen stack offline; the planner routes itself.")

    lx, rx = 175, 680
    text(draw, (lx, 262), "\u03c4", 30, AMBER, bold=True,
         alpha=alpha_for(visible, 0))
    text(draw, (lx + 48, 268), "the criterion floor: the latent distance the task", 21,
         INK, alpha=alpha_for(visible, 0))
    text(draw, (lx + 48, 298), "itself cannot distinguish", 21, INK,
         alpha=alpha_for(visible, 0))
    text(draw, (rx, 262), "k", 30, AMBER, bold=True,
         alpha=alpha_for(visible, 1))
    text(draw, (rx + 48, 268), "the sampler's measured convergence", 21, INK,
         alpha=alpha_for(visible, 1))
    text(draw, (rx + 48, 298), "point: smallest budget past it", 21, INK,
         alpha=alpha_for(visible, 1))

    draw.line((s(120), s(372), s(WIDTH - 120), s(372)),
              fill=rgba(tint(INK, 0.18)), width=s(1))

    text(draw, (WIDTH / 2, 420),
         "and whether to draft at all is the planner's own call:", 21, MUTED,
         anchor="mm", alpha=alpha_for(visible, 2))
    text(draw, (WIDTH / 2, 468), "c*  \u2264  \u03c4  ?", 30, INK, bold=True,
         anchor="mm", alpha=alpha_for(visible, 2))
    text(draw, (390, 530), "yes \u2192 plan flat, no drafter", 22, BLUE,
         anchor="mm", alpha=alpha_for(visible, 3))
    text(draw, (880, 530), "no \u2192 draft and verify", 22, AMBER,
         anchor="mm", alpha=alpha_for(visible, 3))
    text(draw, (WIDTH / 2, 590),
         "same \u03c4, no new constant \u00b7 first crossing retires the drafter", 19,
         MUTED, anchor="mm", alpha=alpha_for(visible, 4))
    return image


# ======================================================================
# Act 4 / results card
# ======================================================================

def results_slide(visible: float) -> Image.Image:
    image, draw = canvas("RESULTS")
    heading(draw, "Measured outcomes",
            "Lower drafter cost without sacrificing success.")

    rows = [
        ("1.8 vs 50", AMBER, "drafter calls per replan, at matched success"),
        ("10 / 10", TEAL, "pre-registered cells meet or beat the strongest flat baseline"),
        ("65.6 \u2192 93.0", TEAL, "transplant repairs an amortized executor's long-horizon collapse"),
        ("0", TEAL, "learned parameters added"),
    ]
    for index, (stat, color, label) in enumerate(rows):
        y = 268 + index * 78
        a = alpha_for(visible, index)
        text(draw, (330, y), stat, 29, color, bold=True, anchor="rm", alpha=a)
        text(draw, (395, y), label, 22, INK, anchor="lm", alpha=a)

    text(draw, (WIDTH / 2, 600), "SURVEYOR", 24, INK, bold=True, anchor="mm",
         alpha=alpha_for(visible, 4))
    return image


# ======================================================================
# assembly
# ======================================================================

SlideRenderer = Callable[[float], Image.Image]


def add_slide(
    frames: list[Image.Image],
    durations: list[int],
    renderer: SlideRenderer,
    items: int,
    *,
    opening_hold: int = 600,
    reveal_hold: int = 340,
    final_hold: int = 2200,
) -> None:
    frames.append(finish(renderer(0)))
    durations.append(round(opening_hold * SLOWDOWN))
    for item in range(items):
        for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
            frames.append(finish(renderer(item + ease(fraction))))
            durations.append(round(75 * SLOWDOWN))
        durations[-1] += round(reveal_hold * SLOWDOWN)
    durations[-1] += round(final_hold * SLOWDOWN)


def crossfade(
    frames: list[Image.Image],
    durations: list[int],
    into: Image.Image,
    steps: int = 6,
) -> None:
    """Blend the last emitted frame into `into` (already finished)."""
    if not frames:
        return
    base = frames[-1]
    for i in range(1, steps + 1):
        frames.append(Image.blend(base, into, ease(i / steps)))
        durations.append(round(80 * SLOWDOWN))


def render_frames() -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    add_slide(frames, durations, title_slide, 5, final_hold=2000)

    mech_f, mech_d = mechanism_frames()
    crossfade(frames, durations, mech_f[0])
    frames.extend(mech_f)
    durations.extend(mech_d)

    const_first = finish(constants_slide(0))
    crossfade(frames, durations, const_first)
    add_slide(frames, durations, constants_slide, 5, final_hold=2400)

    results_first = finish(results_slide(0))
    crossfade(frames, durations, results_first)
    add_slide(frames, durations, results_slide, 5, final_hold=5000)
    return frames, durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("surveyor_overview.gif"),
        help="GIF destination (default: beside this script)",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(__file__).with_name("surveyor_overview_preview.png"),
        help="static final-screen preview destination",
    )
    parser.add_argument(
        "--dump-frames",
        type=int,
        nargs="*",
        help="also save these frame indices as PNGs beside the GIF",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames, durations = render_frames()
    palette = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    paletted = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    paletted[0].save(
        args.output,
        save_all=True,
        append_images=paletted[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    frames[-1].save(args.preview, optimize=True)
    if args.dump_frames:
        for index in args.dump_frames:
            frames[index % len(frames)].save(
                args.output.with_name(f"frame_{index:04d}.png")
            )
    print(f"frames: {len(frames)}")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)")
    print(f"duration: {sum(durations) / 1000:.1f}s")


if __name__ == "__main__":
    main()
