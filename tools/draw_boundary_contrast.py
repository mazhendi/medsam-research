"""Regenerate code-aligned vector diagrams; optional dependency: cairosvg."""

import math
from html import escape
from pathlib import Path

P = Path(__file__).parent
s = []
ink = "#242424"
green = "#5e804d"
teal = "#337e86"
purple = "#8155a6"
gold = "#9e7536"
red = "#9c4e50"
blue = "#477ba8"
gray = "#737373"


def rect(x, y, w, h, fc="white", ec=ink, r=0, lw=1.5, dash=False):
    s.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fc}" stroke="{ec}" stroke-width="{lw}"'
        + (' stroke-dasharray="8 6"' if dash else "")
        + "/>"
    )


def txt(x, y, t, fs=24, c=ink, anchor="middle", bold=False):
    s.append(
        f'<text x="{x}" y="{y}" font-family="Liberation Serif, Times New Roman, serif" font-size="{fs}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{c}">{escape(t)}</text>'
    )


def lines(x, y, ts, fs=23, c=ink):
    for i, t in enumerate(ts):
        txt(x, y + i * (fs + 7), t, fs, c)


def box(x, y, w, h, ts, fc, ec, fs=23):
    rect(x, y, w, h, fc, ec, 12)
    lines(x + w / 2, y + h / 2 - (len(ts) - 1) * (fs + 7) / 2 + 8, ts, fs)


def path(d, c=ink, lw=1.8, fill="none", dash=False):
    s.append(
        f'<path d="{d}" fill="{fill}" stroke="{c}" stroke-width="{lw}"'
        + (' stroke-dasharray="7 5"' if dash else "")
        + "/>"
    )


def arrow(pts, c=ink, lw=2):
    path(" ".join(("M" if i == 0 else "L") + f"{x},{y}" for i, (x, y) in enumerate(pts)), c, lw)
    x, y = pts[-1]
    a, b = pts[-2]
    dx, dy = x - a, y - b
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / n, dy / n
    path(
        f"M{x},{y} L{x - 10 * ux + 4 * uy},{y - 10 * uy - 4 * ux} L{x - 10 * ux - 4 * uy},{y - 10 * uy + 4 * ux} Z",
        c,
        0,
        c,
    )


def circ(x, y, t, c=ink):
    s.append(f'<circle cx="{x}" cy="{y}" r="15" fill="white" stroke="{c}" stroke-width="1.6"/>')
    txt(x, y + 7, t, 22, c)


def token(x, y, c=purple, n=4):
    for k in range(n):
        rect(x + 21 * k, y, 15, 21, "#e8ddf3", c, 0, 1.3)


def feat(x, y, c=green, n=4):
    for k in range(n):
        rect(x, y + 26 * k, 18, 18, "#edf3e6" if c == green else "#e4edf7", c, 0, 1)


def shape(x, y, w=70, h=64, k=0):
    rect(x, y, w, h, "#f8f4fc", "#c5aedb", 1, 1)
    for a0 in (0.82, 1.05):
        pts = []
        for j in range(65):
            a = j * math.pi / 32
            r = 1 + 0.13 * math.sin(3 * a + 0.9 * k) + 0.07 * math.cos(5 * a)
            pts.append(
                (
                    x + w * 0.5 + w * 0.3 * math.cos(a) * r * a0 * (1 + 0.06 * k),
                    y + h * 0.5 + h * 0.32 * math.sin(a) * r * a0,
                )
            )
        path("M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts) + " Z", purple, 1.5)


def heat(x, y, w=80, h=75, kind="mean"):
    rect(x - 1, y - 1, w + 2, h + 2, "white", "#999999", 0, 1)
    n = 13
    for j in range(n):
        for i in range(n):
            xx = (i + 0.5) / n
            yy = (j + 0.5) / n
            r = ((xx - 0.5) / 0.33) ** 2 + ((yy - 0.5) / 0.38) ** 2
            v = math.exp(-1.7 * r) if kind == "mean" else math.exp(-(((r - 1) / 0.32) ** 2))
            aa = (54, 28, 90)
            bb = (225, 226, 65) if kind == "mean" else (68, 191, 180)
            c = "#" + "".join(f"{int(aa[z] + (bb[z] - aa[z]) * v):02x}" for z in range(3))
            rect(x + i * w / n, y + j * h / n, w / n + 0.1, h / n + 0.1, c, c, 0, 0)


def group(x, y, w, h, title, fill, c):
    rect(x, y, w, h, fill, c, 0, 1.2, True)
    txt(x + w / 2, y + h - 12, title, 23, c)


def frozen(x, y, c=ink):
    for a in (0, math.pi / 3, 2 * math.pi / 3):
        dx, dy = 10 * math.cos(a), 10 * math.sin(a)
        path(f"M{x - dx},{y - dy} L{x + dx},{y + dy}", c, 1.5)


ROOT = Path(__file__).resolve().parents[1] / "docs" / "figures"
ROOT.mkdir(parents=True, exist_ok=True)


def port(x, y, t, c=teal, w=110):
    rect(x, y, w, 32, "white", c, 4, 1.3)
    txt(x + w / 2, y + 24, t, 23, c)


def dot(x, y, c):
    s.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{c}"/>')


def start(w, h):
    global s
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    rect(0, 0, w, h, "white", "white", 0, 0)


def finish(name):
    import cairosvg

    s.append("</svg>")
    svg = ROOT / f"{name}.svg"
    svg.write_text("\n".join(s))
    cairosvg.svg2png(url=str(svg), write_to=str(ROOT / f"{name}.png"), scale=1.5)
    cairosvg.svg2pdf(url=str(svg), write_to=str(ROOT / f"{name}.pdf"))


def panel(x, y, w, h, title, c, fc):
    rect(x, y, w, h, fc, c, 0, 1.2, True)
    txt(x + 20, y + 35, title, 27, c, anchor="start", bold=True)


# Standalone module B. Names correspond to InnerOuterBoundary.forward.
start(1640, 900)
txt(32, 46, "B | Inner–Outer Contrast Boundary Module", 31, anchor="start", bold=True)
txt(
    32,
    83,
    "Input Fs = mean[Proj(F1), Upsample(Proj(F2))]; p0 from the learned coarse head.",
    23,
    gray,
    anchor="start",
)
panel(30, 111, 1580, 626, "Paired multi-distance sampling with a local fallback", teal, "#f7fbfb")
port(65, 337, "Fs", teal, 92)
path("M157,353 L190,353 L190,230", teal, 2)
arrow([(190, 230), (250, 230)], teal)
box(250, 187, 290, 86, ["DW Conv 3×3 → Conv 1×1", "GN → GELU"], "#e8f3f3", teal, 23)
arrow([(540, 230), (1085, 230), (1085, 347)], teal)
txt(800, 215, "Local detail L", 24, teal)
path("M190,353 L190,359", teal, 2)
dot(190, 353, teal)
arrow([(190, 359), (250, 359)], teal)
box(250, 316, 290, 86, ["Inner sampling", "Fs(x − r n)"], "#edf1fa", blue, 24)
path("M190,353 L190,488", teal, 2)
arrow([(190, 488), (250, 488)], teal)
box(250, 445, 290, 86, ["Outer sampling", "Fs(x + r n)"], "#f0e7f8", purple, 24)
port(306, 268, "n, r", purple, 110)
arrow([(361, 300), (361, 316)], purple)
# Second sampler uses the same named normal, no crossing wire.
port(306, 412, "n, r", purple, 110)
arrow([(361, 444), (361, 445)], purple)
arrow([(540, 359), (596, 359), (596, 408), (650, 408)], blue)
arrow([(540, 488), (596, 488), (596, 438), (650, 438)], purple)
box(
    650,
    355,
    330,
    132,
    ["Pair difference at r = 1,2,4", "Concat differences", "Conv 1×1 → GN → GELU"],
    "#f0e7f8",
    purple,
    23,
)
arrow([(980, 423), (1038, 423)], purple)
box(
    1038,
    347,
    302,
    154,
    ["Reliability-gated fusion", "a = rho · sigmoid(G)", "B = (1−a)L + aC"],
    "#e8f3f3",
    teal,
    24,
)
arrow([(1340, 423), (1430, 423)], teal)
port(1430, 407, "Boundary B", teal, 140)
port(65, 623, "p0", gold, 92)
arrow([(157, 639), (250, 639)], gold)
box(
    250,
    592,
    430,
    95,
    ["n = −grad(p0) / sqrt(|grad(p0)|² + eps)", "rho = gradient strength × valid margin"],
    "#fcf4e6",
    gold,
    23,
)
arrow([(680, 639), (752, 639)], gold)
port(752, 623, "n, rho", gold, 134)
port(1125, 559, "rho", gold, 92)
arrow([(1171, 559), (1171, 501)], gold)
txt(
    820,
    789,
    "Zero directional reliability ⇒ local-only output. Sampling radii are feature-grid pixels.",
    24,
    teal,
)
txt(
    32,
    858,
    "All directions come from predicted p0; no ground-truth boundary or normal enters forward().",
    22,
    gray,
    anchor="start",
)
finish("boundary_contrast_module_b")

# Standalone module C, exact constrained_pool affinity.
start(1640, 900)
txt(32, 46, "C | Boundary-Constrained Cross-Scale Fusion", 31, anchor="start", bold=True)
txt(
    32,
    83,
    "Applied twice: F3 → F2, then F2 → F1. D and S are computed in parallel.",
    23,
    gray,
    anchor="start",
)
panel(
    30,
    111,
    1580,
    626,
    "Parallel detail / semantic paths with complementary fusion",
    purple,
    "#faf7fd",
)
port(65, 234, "Fshallow", teal, 145)
arrow([(210, 250), (294, 250)], teal)
box(
    294,
    207,
    385,
    86,
    ["Semantic-guided detail", "D = Fshallow · sigmoid(Conv(U))"],
    "#e8f3f3",
    teal,
    24,
)
port(426, 147, "U", green, 102)
arrow([(477, 179), (477, 207)], green)
arrow([(679, 250), (1055, 250)], teal)
circ(1071, 250, "×", purple)
port(1021, 173, "g", purple, 100)
arrow([(1071, 205), (1071, 234)], purple)
port(65, 505, "Fdeep", green, 130)
arrow([(195, 521), (237, 521)], green)
box(237, 478, 212, 86, ["Upsample", "U = Resize(Fdeep)"], "#edf2e7", green, 23)
arrow([(449, 521), (496, 521)], green)
box(
    496,
    450,
    430,
    142,
    [
        "Boundary-constrained 3×3 pooling",
        "axy = exp(−lambda |px−py| − |ex−ey|)",
        "S = Conv-GN-GELU(normalized pool)",
    ],
    "#f0e7f8",
    purple,
    22,
)
port(571, 374, "p0, e", gold, 156)
arrow([(649, 406), (649, 450)], gold)
arrow([(926, 521), (1055, 521)], purple)
circ(1071, 521, "×", purple)
port(1001, 590, "1 − g", purple, 140)
arrow([(1071, 590), (1071, 537)], purple)
arrow([(1087, 250), (1237, 250), (1237, 369)], purple)
arrow([(1087, 521), (1237, 521), (1237, 401)], purple)
circ(1237, 385, "+", purple)
arrow([(1253, 385), (1410, 385)], purple)
port(1410, 369, "Fout", purple, 138)
# Supporting local computations use named ports, avoiding inter-path crossings.
box(69, 630, 432, 65, ["B → Conv1×1 → sigmoid → e"], "#fcf4e6", gold, 23)
box(
    595,
    630,
    827,
    65,
    ["g = sigmoid(Conv1×1([D, S, B]))    |    e and g are learned gates"],
    "#f0e7f8",
    purple,
    23,
)
txt(820, 789, "Fout = g · D + (1−g) · S", 29, purple)
txt(
    32,
    858,
    "Boundary affinity is continuous; p0 and B can receive gradients through the semantic branch.",
    22,
    gray,
    anchor="start",
)
finish("boundary_contrast_module_c")

# Corrected overall: baseline decoder retained, three independent feature bypasses.
start(2200, 1220)
txt(34, 49, "Baseline-Preserving Boundary Enhancement", 34, anchor="start", bold=True)
txt(2160, 47, "Corrected implementation · efficacy unvalidated", 22, gray, anchor="end")
panel(30, 80, 2140, 505, "A | Original trained SAM2 segmentation path", blue, "#fbfcfe")
box(52, 276, 100, 100, ["Input", "image"], "#f5f5f5", "#999", 25)
arrow([(152, 326), (200, 326)])
box(200, 145, 290, 365, ["Existing SAM2.1", "image encoder", "", "Frozen"], "#e6efdb", green, 29)
frozen(230, 171, green)
# Three DISTINCT output ports, original SAM decoder-interface channel/grid shapes.
for idx, y in enumerate((200, 320, 440), 1):
    arrow([(490, y), (535, y)], green)
    for k in reversed(range(3)):
        rect(540 + k * 6, y - 29 - k * 6, 30, 58, "#e6efdb", green, 0, 1)
    txt(623, y + 7, f"F{idx}", 28, green)
    arrow([(650, y), (1464, y)], green)
    # Delta features enter their own sum. Named ports remove long crossing wires.
    port(1400, y - 84, f"ΔF{idx}", purple, 160)
    arrow([(1480, y - 52), (1480, y - 16)], purple)
    circ(1480, y, "+", purple)
    arrow([(1496, y), (1650, y)], blue)
    txt(1572, y - 12, f"F{idx}*", 27, blue)
box(
    1650,
    145,
    305,
    365,
    ["Existing trained", "SAM2 mask decoder", "", "Preserved weights", "Frozen by default"],
    "#e5edf6",
    blue,
    29,
)
arrow([(1955, 320), (2014, 320)], blue)
rect(2014, 276, 105, 88, "#111", "#111", 0, 0)
path("M2041,318 C2033,291 2087,290 2090,319 C2098,349 2041,353 2041,318 Z", "white", 0, "white")
txt(2066, 410, "Mask*", 25)
txt(
    1035,
    555,
    "F1/F2: original high-resolution interface; F3: original embedding including no-memory term.",
    22,
    gray,
)
# Original conditioning remains identical. Separate input arrows below the decoder.
box(1640, 635, 240, 92, ["Existing learned", "task tokens"], "#e5edf6", blue, 24)
arrow([(1760, 635), (1760, 510)], blue)
box(1910, 635, 255, 92, ["Original no-mask", "embedding + PE"], "#e5edf6", blue, 24)
arrow([(2037, 635), (2037, 550), (1895, 550), (1895, 510)], blue)
panel(
    30,
    620,
    1575,
    365,
    "Add-on feature branch",
    teal,
    "#f7fbfb",
)
port(65, 741, "F1, F2", teal, 130)
arrow([(195, 757), (250, 757)], teal)
box(250, 707, 270, 100, ["B. Inner–Outer", "Contrast Boundary", "Module"], "#e8f3f3", teal, 25)
port(320, 652, "p0", gold, 120)
arrow([(380, 684), (380, 707)], gold)
arrow([(520, 757), (600, 757)], teal)
txt(560, 739, "B", 25, teal)
box(600, 707, 300, 100, ["C. Boundary-Constrained", "Cross-Scale Fusion"], "#f0e7f8", purple, 24)
port(625, 652, "F1, F2, F3, p0", green, 250)
arrow([(750, 684), (750, 707)], green)
arrow([(900, 757), (950, 757)], purple)
box(950, 707, 260, 100, ["Per-scale projection", "Pi: width → original Ci"], "#f0e7f8", purple, 23)
arrow([(1210, 757), (1265, 757)], purple)
box(1265, 707, 275, 100, ["Per-scale gate αi", "Initialized to ZERO"], "#f0e7f8", purple, 25)
arrow([(1402, 807), (1402, 856)], purple)
port(1225, 856, "ΔF1, ΔF2, ΔF3", purple, 355)
txt(675, 899, "Module B / C internals remain in their separate detailed figures.", 24, gray)
txt(675, 945, "Only feature deltas are added. There is NO baseline-mask-logit addition.", 23, teal)
# Local coarse prior: a learned head, never a final mask or a ground-truth prompt.
port(1805, 781, "F3", green, 100)
arrow([(1855, 813), (1855, 842)], green)
box(1640, 842, 520, 74, ["Learned coarse head → resize → sigmoid → p0"], "#fcf4e6", gold, 24)
txt(1900, 962, "Same-name ports share a tensor; scales are never shorted.", 21, gray)
panel(30, 1020, 2140, 150, "Identity-preserving feature gate", purple, "#faf7fd")
txt(650, 1115, "Fi* = Fi + αi · Pi(Ri)", 34, purple)
txt(1540, 1115, "At αi = 0: Fi* = Fi; initial output = baseline", 28, purple)
txt(
    34,
    1203,
    "*Mask schematic. Original checkpoint, tokens, no-mask embedding and positional encoding are retained.",
    22,
    gray,
    anchor="start",
)
finish("boundary_contrast")
print("Saved baseline-preserving overall and detailed module figures")
