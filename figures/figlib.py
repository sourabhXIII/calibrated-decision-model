"""Tiny SVG toolkit for the blog figures. Standard library only.

Every figure is written twice, NN-name.svg (light) and NN-name.dark.svg (dark),
each with a fixed theme. Colours are the validated 3-slot categorical palette
(blue / orange / aqua) plus neutral chrome; aqua sits below 3:1 on the light
surface, so every aqua mark gets a direct label.
"""

from __future__ import annotations

import math
from html import escape

LIGHT_VARS = ("--surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; "
              "--axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --ctx:#7d7b75; "
              "--block:#ecebe6; --wash1:rgba(42,120,214,0.10); --wash2:rgba(235,104,52,0.10); "
              "--wash3:rgba(27,175,122,0.12); --washc:rgba(125,123,117,0.12);")
DARK_VARS = ("--surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; "
             "--axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --ctx:#9a988f; "
             "--block:#2a2a28; --wash1:rgba(57,135,229,0.16); --wash2:rgba(217,89,38,0.16); "
             "--wash3:rgba(25,158,112,0.18); --washc:rgba(154,152,143,0.16);")
RULES = """
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink); }
  .bg { fill: var(--surface); }
  .t2 { fill: var(--ink2); } .tm { fill: var(--muted); }
  .title { font-size: 15px; font-weight: 600; } .sub { font-size: 12px; fill: var(--ink2); }
  .lab { font-size: 12px; } .small { font-size: 11px; } .tick { font-size: 11px; fill: var(--muted);
         font-variant-numeric: tabular-nums; }
  .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--axis); stroke-width: 1; }
  .f1 { fill: var(--s1); } .f2 { fill: var(--s2); } .f3 { fill: var(--s3); } .fc { fill: var(--ctx); }
  .fb { fill: var(--block); } .fw1 { fill: var(--wash1); } .fw2 { fill: var(--wash2); }
  .fw3 { fill: var(--wash3); } .fwc { fill: var(--washc); } .fs { fill: var(--surface); }
  .l1 { stroke: var(--s1); } .l2 { stroke: var(--s2); } .l3 { stroke: var(--s3); } .lc { stroke: var(--ctx); }
  .lm { stroke: var(--muted); } .lk { stroke: var(--ink2); } .ls { stroke: var(--surface); }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .thin { fill: none; stroke-width: 1.25; }
  .box { fill: none; stroke: var(--axis); stroke-width: 1.25; }
  .ring { stroke: var(--surface); stroke-width: 2; }
"""


def style(dark: bool) -> str:
    # Fixed theme per file. An SVG used as <img> follows the OS colour scheme, not the
    # page's, so a self-switching figure can clash with a light-only blog. We ship
    # NN-name.svg (light) and NN-name.dark.svg (dark) and let the page pick.
    return f"<style>\n  svg {{ {DARK_VARS if dark else LIGHT_VARS} }}{RULES}</style>"


class Fig:
    def __init__(self, w: int, h: int, title: str = "", desc: str = ""):
        self.w, self.h, self.parts = w, h, []
        self.title, self.desc = title, desc

    def add(self, s: str):
        self.parts.append(s)

    # primitives -----------------------------------------------------------
    def text(self, x, y, s, cls="lab", anchor="start", weight=None, rotate=None):
        extra = f' font-weight="{weight}"' if weight else ""
        if rotate is not None:
            extra += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>'
                 f'{escape(str(s))}</text>')

    def rect(self, x, y, w, h, cls, rx=0):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
                 f'rx="{rx}" class="{cls}"/>')

    def line(self, x1, y1, x2, y2, cls):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')

    def path(self, d, cls):
        self.add(f'<path d="{d}" class="{cls}"/>')

    def poly(self, pts, cls):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.path(d, cls)

    def dot(self, x, y, cls, r=4.5):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls} ring"/>')

    def arrow(self, x1, y1, x2, y2, cls="lk"):
        ang = math.atan2(y2 - y1, x2 - x1)
        a, L = 0.45, 7
        p1 = (x2 - L * math.cos(ang - a), y2 - L * math.sin(ang - a))
        p2 = (x2 - L * math.cos(ang + a), y2 - L * math.sin(ang + a))
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'class="thin {cls}"/>')
        self.add(f'<path d="M{p1[0]:.1f},{p1[1]:.1f} L{x2:.1f},{y2:.1f} L{p2[0]:.1f},{p2[1]:.1f}" '
                 f'class="thin {cls}"/>')

    def hbar(self, x0, y, length, thick, cls):
        """Horizontal bar from baseline x0, 4px rounded data-end, square at the baseline."""
        if length <= 0.5:
            return
        r = min(4, length, thick / 2)
        x1 = x0 + length
        d = (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
             f"V{y + thick - r:.1f} Q{x1:.1f},{y + thick:.1f} {x1 - r:.1f},{y + thick:.1f} "
             f"H{x0:.1f} Z")
        self.path(d, cls)

    def vbar(self, x, y_base, height, thick, cls):
        """Vertical column from baseline y_base upward, rounded top."""
        if height <= 0.5:
            return
        r = min(4, height, thick / 2)
        y1 = y_base - height
        d = (f"M{x:.1f},{y_base:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} "
             f"H{x + thick - r:.1f} Q{x + thick:.1f},{y1:.1f} {x + thick:.1f},{y1 + r:.1f} "
             f"V{y_base:.1f} Z")
        self.path(d, cls)

    def legend(self, x, y, items, gap=22):
        """items: [(cls, label)]; swatch + text, laid out left to right."""
        for cls, lab in items:
            self.rect(x, y - 9, 12, 12, cls, rx=2)
            self.text(x + 17, y + 1, lab, "small t2")
            x += 17 + 6.6 * len(lab) + gap

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="t d">'
                f'<title id="t">{escape(self.title)}</title><desc id="d">{escape(self.desc)}</desc>')
        body = "\n".join(self.parts)
        path = str(path)
        for dark, out in [(False, path), (True, path.replace(".svg", ".dark.svg"))]:
            with open(out, "w") as f:
                f.write(head + style(dark) + f'<rect width="{self.w}" height="{self.h}" class="bg"/>\n'
                        + body + "\n</svg>\n")


class Axes:
    """Linear or log axes inside a plot rectangle."""

    def __init__(self, fig, x, y, w, h, xdom, ydom, xlog=False, ylog=False):
        self.f, self.x, self.y, self.w, self.h = fig, x, y, w, h
        self.xdom, self.ydom, self.xlog, self.ylog = xdom, ydom, xlog, ylog

    def _n(self, v, dom, log):
        if log:
            return (math.log10(v) - math.log10(dom[0])) / (math.log10(dom[1]) - math.log10(dom[0]))
        return (v - dom[0]) / (dom[1] - dom[0])

    def px(self, v):
        return self.x + self._n(v, self.xdom, self.xlog) * self.w

    def py(self, v):
        return self.y + self.h - self._n(v, self.ydom, self.ylog) * self.h

    def grid_y(self, ticks, fmt=str, label=None):
        for t in ticks:
            yy = self.py(t)
            self.f.line(self.x, yy, self.x + self.w, yy, "grid")
            self.f.text(self.x - 8, yy + 4, fmt(t), "tick", "end")
        self.f.line(self.x, self.y + self.h, self.x + self.w, self.y + self.h, "axis")
        if label:
            self.f.text(self.x - 44, self.y + self.h / 2, label, "small t2", "middle", rotate=-90)

    def ticks_x(self, ticks, fmt=str, label=None):
        for t in ticks:
            self.f.text(self.px(t), self.y + self.h + 16, fmt(t), "tick", "middle")
        if label:
            self.f.text(self.x + self.w / 2, self.y + self.h + 34, label, "small t2", "middle")

    def series(self, pts, cls, dots=True):
        xy = [(self.px(a), self.py(b)) for a, b in pts]
        self.f.poly(xy, f"line {cls}")
        if dots:
            dcls = cls.replace("l", "f")
            for x, y in xy:
                self.f.dot(x, y, dcls, r=4)
        return xy
