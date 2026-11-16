#!/usr/bin/env python3
"""asaply profile banner: a New York subway-tile wall that texts back.

Errands pile up on the wall as pixel-art mosaics, then the tiles flip in waves to
spell "text it." and "we run it." while a text thread plays out on top. It ends
on a station-style name tablet.

Needs Pillow, numpy, ffmpeg on PATH, and macOS (for Apple Color Emoji).
    python3 source/subway.py                    # writes assets/asaply-subway.gif + .png
    python3 source/subway.py --preview 3.6 8.5 --out /tmp  # stills at those times
"""
import argparse
import math
import subprocess
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
INTER = ROOT / "source/fonts/inter-tight.woff2"
EMOJI = "/System/Library/Fonts/Apple Color Emoji.ttc"

W, H, S, FPS = 1200, 630, 2, 25     # output size, supersample factor, frame rate
BAND = 70                           # signage band height
TW, TH = 20, 10                     # subway tile pitch (2:1)
COLS, ROWS = W // TW, (H - BAND) // TH
WALL_H = ROWS * TH
FLIP = 0.28                         # seconds for one tile to flip
DURATION = 13.4


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def clamp(x):
    return max(0.0, min(1.0, x))


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    return 1 - (1 - clamp(x)) ** 3


def ease_in(x):
    return clamp(x) ** 3


def ease_back(x, k=1.7):
    x = clamp(x) - 1
    return 1 + (k + 1) * x ** 3 + k * x ** 2


# Tile colours. The first five are the brand set; the rest are for the icons.
COLORS = ["#F2F0EA", "#151515", "#1086FF", "#A6A6AC", "#4A4A50",
          "#7A4A2A", "#D9A066", "#E5484D", "#FFC83D", "#3E9E55",
          "#FF8A3D", "#F4A6C4", "#8FCBFF", "#FFF1D0", "#8E6CD8"]
WHITE, INK, BLUE, GRAY, CHAR = range(5)
CREAM = 13
PAL = [rgb(c) for c in COLORS]
GROUT, HOLE = rgb("#CFCABF"), rgb("#A39D91")
BLACK, PAPER = rgb("#111111"), (255, 255, 255)
IOS_BLUE, BUBBLE_GRAY = rgb("#1086FF"), rgb("#E9E9EB")

RR, CC = np.mgrid[0:ROWS, 0:COLS]
rng = np.random.default_rng(7)


def inter(size, wght=400):
    f = ImageFont.truetype(str(INTER), round(size))
    f.set_variation_by_axes([wght])
    return f


# ---------------------------------------------------------------- tiles

LEVELS = [1, .86, .7, .54, .38, .24, .12, .04]


def make_sprites():
    """Pre-render every tile colour at every flip height, grout included."""
    cw, ch = TW * S, TH * S
    sprites = {}
    for ci, base in enumerate(PAL):
        for li, s in enumerate(LEVELS):
            cell = Image.new("RGB", (cw, ch), GROUT if li == 0 else HOLE)
            d = ImageDraw.Draw(cell)
            col = tuple(round(v * (.62 + .38 * s)) for v in base)
            th = max(2.0, (ch - 2 * S) * s)
            y0, x0, x1 = (ch - th) / 2, S, cw - S - 1
            y1 = y0 + th - 1
            r = min(4, max(0, th / 2 - 1))
            d.rounded_rectangle([x0, y0, x1, y1], r, fill=mix(col, (0, 0, 0), .18))
            if s > .5:
                d.rounded_rectangle([x0, y0, x1, y1 - 2], r, fill=col)
                d.line([(x0 + 5, y0 + 2), (x1 - 5, y0 + 2)], fill=mix(col, PAPER, .4), width=2)
            sprites[ci, li] = cell
    return sprites


def level(s):
    return min(range(len(LEVELS)), key=lambda i: abs(LEVELS[i] - s))


class Wall:
    """Each tile keeps a list of (time, colour) flips, replayed in time order."""

    def __init__(self):
        self.cur = np.full((ROWS, COLS), WHITE)
        self.events = [[] for _ in range(ROWS * COLS)]

    def paint(self, target, t0, delay):
        """Flip every tile whose colour differs from `target` (-1 = leave alone)."""
        for r, c in np.argwhere((target >= 0) & (target != self.cur)):
            self.events[r * COLS + c].append((t0 + float(delay[r, c]), int(target[r, c])))
            self.cur[r, c] = target[r, c]

    def state(self, i, t):
        base, out = WHITE, (WHITE, 0)
        for te, v in sorted(self.events[i]):
            if te > t:
                break
            p = (t - te) / FLIP
            if p >= 1:
                out = (v, 0)
            else:
                q = ease_io(p)
                out = (base if q < .5 else v, level(abs(math.cos(math.pi * q))))
            base = v
        return out


def jitter(amount):
    return rng.random((ROWS, COLS)) * amount


def from_point(r0, c0, sec_per_px):
    return np.hypot((RR - r0) * TH, (CC - c0) * TW) * sec_per_px


def word_mask(text, size, wght, track, cy, cx=W / 2, max_w=1090, thresh=.42):
    """Rasterise text onto the tile grid, nudged so strokes land on whole tiles."""
    f = inter(size, wght)
    width = sum(f.getlength(ch) for ch in text) + track * size * (len(text) - 1)
    if width > max_w:
        return word_mask(text, size * max_w / width, wght, track, cy, cx, max_w, thresh)
    tmp = Image.new("L", (W * 2, int(size * 2)), 0)
    d = ImageDraw.Draw(tmp)
    x = 20
    for ch in text:
        d.text((x, size * .2), ch, font=f, fill=255)
        x += f.getlength(ch) + track * size
    glyphs = tmp.crop(tmp.getbbox())
    best, best_score = None, -1
    for dx in range(0, TW, 3):
        for dy in range(0, TH, 3):
            im = Image.new("L", (W, WALL_H), 0)
            im.paste(glyphs, (round(cx - glyphs.width / 2) + dx - TW // 2,
                              round(cy - glyphs.height / 2) + dy - TH // 2))
            cov = np.asarray(im, float).reshape(ROWS, TH, COLS, TW).mean(axis=(1, 3)) / 255
            score = np.abs(cov - .5)[cov > .02].sum()
            if score > best_score:
                best, best_score = cov, score
    return best > thresh


EMOJI_FONT = ImageFont.truetype(EMOJI, 160)


def emoji_image(ch):
    im = Image.new("RGBA", (200, 200))
    ImageDraw.Draw(im).text((10, 10), ch, font=EMOJI_FONT, embedded_color=True)
    return im.crop(im.getbbox())


# Pixel art for the errands. One character = one column x two rows of tiles,
# so each "pixel" is square. "." leaves the wall alone.
INKS = {"K": INK, "B": BLUE, "g": GRAY, "D": CHAR, "N": 5, "T": 6, "R": 7, "Y": 8,
        "G": 9, "O": 10, "P": 11, "S": 12, "c": CREAM, "V": 14, "w": WHITE}
ART = {
    "coffee": """
        ....g..g....
        ...g..g.....
        ..KKKKKKKK..
        .KDDDDDDDDK.
        .KKKKKKKKKK.
        ..KccccccK..
        ..KBBBBBBK..
        ..KBBBBBBK..
        ..KBBBBBBK..
        ..KccccccK..
        ...KccccK...
        ...KKKKKK...""",
    "shirt": """
        ......K.....
        .....K.K....
        .......K....
        ......K.....
        .KKKK..KKKK.
        KBBBBKKBBBBK
        KBBBBBBBBBBK
        KKKBBBBBBKKK
        ..KBBBBBBK..
        ..KBBBBBBK..
        ..KBBBBBBK..
        ..KKKKKKKK..""",
    "ticket": """
        KKKKKKKKKKKKKK
        KRRRRRRRRKYYYK
        .KRRccRRRKYYK.
        .KRRccRRRKYYK.
        KRRRRRRRRKYYYK
        KKKKKKKKKKKKKK""",
    "flowers": """
        .PP....RR...
        PPPP..RRRR..
        PPYP..RRYR..
        PPPP.YY.RR..
        .PP.YYYY....
        ..G.YOYY.G..
        ..G..YY..G..
        ...G.G..G...
        ...KcGGcK...
        ...KccccK...
        ....KccK....
        ....KccK....
        .....KK.....""",
    "taxi": """
        ......KYYK......
        .....KKKKKK.....
        ....KSSKKSSK....
        ...KSSSKKSSSK...
        .KKKKKKKKKKKKKK.
        KYYYYYYYYYYYYYYK
        KYKYKYKYKYKYKYYK
        KYYYYYYYYYYYYYYK
        KKKKKKKKKKKKKKKK
        ..KDDK....KDDK..
        ...KK......KK...""",
    "pizza": """
        KKKKKKKKKKKK
        KTTTTTTTTTTK
        KYYYYYYYYYYK
        .KYRYYYYRYK.
        .KYYYYRYYYK.
        ..KYRYYYYK..
        ..KYYYYRYK..
        ...KYYRYK...
        ...KYYYYK...
        ....KYYK....
        ....KRYK....
        .....KK.....""",
    "bagel": """
        .KKKKKK.
        KTTTTTTK
        KTTKKTTK
        KTTTTTTK
        .KKKKKK.""",
    "box": """
        KKKKKKKK
        KTTTNTTK
        KKKKNKKK
        KTTTNTTK
        KTTTNTTK
        KKKKKKKK""",
    "bag": """
        ..KKKK..
        ..K..K..
        KKKKKKKK
        KRRRRRRK
        KRRRRRRK
        KRRRRRRK
        KKKKKKKK""",
    "milk": """
        ..KK..
        .KccK.
        KccccK
        KKKKKK
        KccccK
        KBBBBK
        KBccBK
        KBBBBK
        KKKKKK""",
    "pill": """
        .KKKKKK.
        KRRRKccK
        KRRRKccK
        .KKKKKK.""",
}


def icon(name):
    lines = textwrap.dedent(ART[name]).strip("\n").splitlines()
    grid = np.full((len(lines) * 2, max(map(len, lines))), -1)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch != ".":
                grid[2 * y:2 * y + 2, x] = INKS[ch]
    return grid


def stamp(wall, grid, r0, c0, t0):
    target = np.full((ROWS, COLS), -1)
    gr, gc = grid.shape
    rs, cs = slice(max(r0, 0), min(r0 + gr, ROWS)), slice(max(c0, 0), min(c0 + gc, COLS))
    target[rs, cs] = grid[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0]
    wall.paint(target, t0, from_point(r0 + gr / 2, c0 + gc / 2, .0011) + jitter(.05))


# ---------------------------------------------------------------- text bubbles

BUBBLE_H, PAD_X, TAIL, SHADOW = 46, 17, 8, 10


def bubble_mask(w, side):
    """iMessage-style bubble mask at S scale, drawn at 4x and downsampled.

    The tail sits bottom right for "you" and bottom left for "them"."""
    h, k = BUBBLE_H, 4 * S
    mask = Image.new("L", ((w + TAIL + 2) * k, h * k), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, w * k, h * k - 1], 21 * k, fill=255)
    d.rounded_rectangle([(w - 14) * k, (h - 24) * k, (w + TAIL) * k, h * k - 1], 15 * k,
                        fill=255, corners=(False, False, False, True))
    d.rounded_rectangle([w * k, (h - 26) * k, (w + 28) * k, h * k - 1], 10 * k,
                        fill=0, corners=(False, False, False, True))
    mask = mask.resize(((w + TAIL + 2) * S, h * S), Image.LANCZOS)
    return mask.transpose(Image.FLIP_LEFT_RIGHT) if side == "them" else mask


def bubble_sprite(side, emoji, text):
    """Return (RGBA sprite at S scale, anchor x, anchor y) — anchor is the tail tip."""
    k = S
    f = inter(25 * k, 500)
    ew = 0
    if emoji:
        e = emoji_image(emoji)
        eh = 24 * k
        e = e.resize((round(e.width * eh / e.height), eh), Image.LANCZOS)
        ew = e.width + 7 * k
    w = round(2 * PAD_X + (ew + f.getlength(text)) / k)
    h = BUBBLE_H
    mask = bubble_mask(w, side)
    fill, ink = (IOS_BLUE, PAPER) if side == "you" else (BUBBLE_GRAY, BLACK)

    m = SHADOW * k
    sprite = Image.new("RGBA", (mask.width + 2 * m, mask.height + 2 * m), (0, 0, 0, 0))
    shadow = Image.new("L", sprite.size, 0)
    shadow.paste(mask.point(lambda v: v * .28), (m, m + 3 * k))
    sprite.putalpha(shadow.filter(ImageFilter.GaussianBlur(5 * k)))
    body = Image.new("RGBA", mask.size, fill + (255,))
    body.putalpha(mask)
    sprite.alpha_composite(body, (m, m))

    d = ImageDraw.Draw(sprite)
    x = m + (PAD_X + (TAIL + 2 if side == "them" else 0)) * k
    if emoji:
        sprite.alpha_composite(e, (round(x), m + (h * k - e.height) // 2))
        x += ew
    d.text((x, m + h * k / 2), text, font=f, fill=ink, anchor="lm")
    ax = m + (w + TAIL) * k if side == "you" else m + 2 * k
    return sprite, ax, m + h * k


def typing_sprite(t):
    """Grey bubble with three bouncing dots."""
    h, k = BUBBLE_H, S
    mask = bubble_mask(74, "them")
    m = SHADOW * k
    sprite = Image.new("RGBA", (mask.width + 2 * m, mask.height + 2 * m), (0, 0, 0, 0))
    body = Image.new("RGBA", mask.size, BUBBLE_GRAY + (255,))
    body.putalpha(mask)
    sprite.alpha_composite(body, (m, m))
    d = ImageDraw.Draw(sprite)
    for i in range(3):
        ph = (math.sin(t * 2 * math.pi * 1.6 - i * .9) + 1) / 2
        cx = m + (TAIL + 2 + 22 + i * 15) * k
        cy = m + (h / 2 - 3 * ph) * k
        r = 5 * k
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=mix((150, 150, 156), (95, 95, 100), ph))
    return sprite, m + 2 * k, m + h * k


class Bubble:
    def __init__(self, side, t_in, emoji=None, text=None, typing_until=None, slide=True):
        self.side, self.t_in, self.slide = side, t_in, slide
        self.typing_until = typing_until
        self.sprite = None if typing_until else bubble_sprite(side, emoji, text)


class Thread:
    """One scene's worth of bubbles. Later bubbles push earlier ones up."""
    GAP = 8

    def __init__(self, bubbles, t_exit, top):
        self.bubbles, self.t_exit, self.top = bubbles, t_exit, top

    def visible(self, b, t):
        return t >= b.t_in and (b.typing_until is None or t < b.typing_until)

    def draw(self, frame, t):
        if t > self.t_exit + .4:
            return
        drop = ease_in((t - self.t_exit) / .3)
        for i, b in enumerate(self.bubbles):
            if not self.visible(b, t):
                continue
            lift = sum((BUBBLE_H + self.GAP) * (ease_out((t - n.t_in) / .22) if n.slide else 1)
                       for n in self.bubbles[i + 1:] if self.visible(n, t))
            sprite, ax, ay = typing_sprite(t) if b.typing_until else b.sprite
            x = (W - 26) * S if b.side == "you" else 26 * S
            y = (H - 22 - lift + drop * 60) * S
            pop = (t - b.t_in) / .34
            scale = .55 + .45 * ease_back(pop) if b.slide else 1
            alpha = clamp(pop * 4 if b.slide else 1) * (1 - drop)
            alpha *= clamp((y / S - BUBBLE_H - self.top) / 50)
            paste(frame, sprite, ax, ay, x, y, scale, alpha)


def paste(frame, sprite, ax, ay, x, y, scale, alpha):
    if alpha <= 0.01:
        return
    if abs(scale - 1) > .005:
        sprite = sprite.resize((max(1, round(sprite.width * scale)),
                                max(1, round(sprite.height * scale))), Image.BICUBIC)
        ax, ay = ax * scale, ay * scale
    if alpha < .995:
        sprite = sprite.copy()
        sprite.putalpha(sprite.getchannel("A").point(lambda v: v * alpha))
    frame.paste(sprite, (round(x - ax), round(y - ay)), sprite)


def chip_sprite(text):
    k = S
    f = inter(28 * k, 650)
    w, h = round(f.getlength(text) / k) + 52, 58
    m = SHADOW * k
    sprite = Image.new("RGBA", ((w + 2 * SHADOW) * k, (h + 2 * SHADOW) * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(sprite)
    d.rounded_rectangle([m, m, m + w * k, m + h * k], 14 * k, fill=BLACK)
    d.text((m + w * k / 2, m + h * k / 2), text, font=f, fill=PAPER, anchor="mm")
    return sprite, m + w * k / 2, m + h * k


class Chip:
    def __init__(self, text, t_in, t_exit):
        self.sprite, self.t_in, self.t_exit = chip_sprite(text), t_in, t_exit

    def draw(self, frame, t):
        if not self.t_in <= t <= self.t_exit + .4:
            return
        pop, drop = (t - self.t_in) / .4, ease_in((t - self.t_exit) / .3)
        sprite, ax, ay = self.sprite
        paste(frame, sprite, ax, ay, W / 2 * S, (H - 44 + drop * 40) * S,
              .6 + .4 * ease_back(pop), clamp(pop * 4) * (1 - drop))


# ---------------------------------------------------------------- signage band

def make_band():
    k = S
    im = Image.new("RGB", (W * k, BAND * k), BLACK)
    d = ImageDraw.Draw(im)
    cy = BAND / 2 - 2
    x = 30
    d.ellipse([x * k, (cy - 7) * k, (x + 14) * k, (cy + 7) * k], fill=PAPER)
    d.rounded_rectangle([(x + 21) * k, (cy - 4) * k, (x + 63) * k, (cy + 4) * k], k, fill=PAPER)
    d.text(((x + 78) * k, cy * k), "asaply", font=inter(31 * k, 750), fill=PAPER, anchor="lm")

    right = (W - 30) * k
    tail = "Co-founder & CEO  ·  New York City"
    f_tail, f_name = inter(19 * k, 450), inter(19 * k, 650)
    d.text((right, cy * k), tail, font=f_tail, fill=rgb("#A1A1A6"), anchor="rm")
    d.text((right - f_tail.getlength(tail) - 14 * k, cy * k), "Chaira Harder",
           font=f_name, fill=PAPER, anchor="rm")
    d.rectangle([0, (BAND - 4) * k, W * k, BAND * k], fill=IOS_BLUE)
    return im


# ---------------------------------------------------------------- the script

def build():
    wall = Wall()

    # 1. Errands pile up: each text drops a pixel-art icon on the wall.
    asks = [(.30, "coffee before my 9am", "coffee", 3, 3),
            (.90, "pick up dry cleaning", "shirt", 2, 21),
            (1.50, "2 tix for tonight", "ticket", 34, 3),
            (2.05, "flowers for mom", "flowers", 2, 41),
            (2.55, "a ride to JFK", "taxi", 31, 20),
            (3.00, "and lunch??", "pizza", 29, 37)]
    for t, _, art, r, c in asks:
        stamp(wall, icon(art), r, c, t + .1)
    extras = [("bagel", 25, 13), ("box", 12, 33), ("bag", 40, 50), ("milk", 25, 54),
              ("pill", 48, 4), ("bagel", 6, 53), ("box", 44, 40), ("milk", 1, 15)]
    for i, (art, r, c) in enumerate(extras):
        stamp(wall, icon(art), r, c, 3.08 + i * .07)
    n = 320
    for i in range(n):
        t = 3.0 + .75 * (i / n) ** .5
        r, c, length = rng.integers(ROWS), rng.integers(COLS - 1), rng.integers(2, 9)
        target = np.full((ROWS, COLS), -1)
        target[r, c:c + length] = rng.choice([INK, INK, BLUE, GRAY, CHAR])
        wall.paint(target, t, (CC - c) * .02)

    # 2. "text it." sweeps in left to right, then a cursor blinks.
    cy = 175
    text_it = word_mask("text it.", 260, 850, .05, cy)
    wall.paint(np.where(text_it, INK, WHITE), 4.0, CC * .014 + RR * .002 + jitter(.08))
    cols = np.where(text_it.any(axis=0))[0]
    rows = np.where(text_it.any(axis=1))[0]
    cursor = np.full((ROWS, COLS), -1)
    cursor[rows.min():rows.max() + 1, cols.max() + 2] = BLUE
    clear = np.where(cursor >= 0, WHITE, -1)
    for i, t in enumerate(np.arange(5.05, 5.95, .3)):
        wall.paint(cursor if i % 2 == 0 else clear, t, RR * .004)

    # 3. "we run it." floods out from asaply's reply onto a blue wall.
    we_run = word_mask("we run it.", 260, 850, .05, cy)
    wall.paint(np.where(we_run, WHITE, BLUE), 6.0, from_point(ROWS - 3, 4, .0009) + jitter(.06))

    # 4. Station name tablet, wiped in right to left like a passing train.
    name = word_mask("asaply", 280, 760, .10, 200, max_w=980, thresh=.45)
    rows = np.where(name.any(axis=1))[0]
    cols = np.where(name.any(axis=0))[0]
    r0, r1 = rows.min() - 6, rows.max() + 7
    c0, c1 = cols.min() - 4, cols.max() + 5
    tablet = np.full((ROWS, COLS), WHITE)
    tablet[r0:r1, c0:c1] = BLUE
    tablet[r0 + 2:r1 - 2, c0 + 1:c1 - 1] = CREAM
    tablet[name] = INK
    wall.paint(tablet, 9.4, (COLS - 1 - CC) * .011 + jitter(.05))

    # 5. Back to a clean wall for the loop.
    wall.paint(np.full((ROWS, COLS), WHITE), 12.4, RR * .012 + jitter(.06))

    threads = [
        Thread([Bubble("you", t, text=s) for t, s, *_ in asks], 3.95, top=250),
        Thread([Bubble("them", 4.55, typing_until=5.2),
                Bubble("them", 5.2, text="on it.", slide=False),
                Bubble("them", 6.45, "☕", "ready at 8:52"),
                Bubble("them", 6.95, "👔", "picked up"),
                Bubble("them", 7.45, "🎟️", "row F, 7pm"),
                Bubble("them", 7.95, "🚕", "6:15 pickup")], 9.3, top=355),
    ]
    chip = Chip("The future takes the subway.", 10.05, 12.3)
    return wall, threads, chip


def render(t, wall, threads, chip, sprites, band):
    frame = Image.new("RGB", (W * S, H * S), GROUT)
    for i in range(ROWS * COLS):
        r, c = divmod(i, COLS)
        frame.paste(sprites[wall.state(i, t)], (c * TW * S, (BAND + r * TH) * S))
    frame.paste(band, (0, 0))
    for th in threads:
        th.draw(frame, t)
    chip.draw(frame, t)
    return frame.reduce(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", nargs="*", type=float, help="render stills at these times")
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    args = ap.parse_args()

    sprites, band = make_sprites(), make_band()
    wall, threads, chip = build()

    def frame_at(t):
        return render(t, wall, threads, chip, sprites, band)

    if args.preview:
        for t in args.preview:
            frame_at(t).save(args.out / f"frame-{t:05.2f}.png")
        return

    gif = args.out / "asaply-subway.gif"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-filter_complex",
         "split[a][b];[a]palettegen=max_colors=255:stats_mode=full[p];"
         "[b][p]paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", str(gif)],
        stdin=subprocess.PIPE)
    for i in range(round(DURATION * FPS)):
        ff.stdin.write(frame_at(i / FPS).tobytes())
    ff.stdin.close()
    ff.wait()
    frame_at(11.6).save(args.out / "asaply-subway.png")
    print(f"{gif.name}: {gif.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
