"""Supplementary Fig. 1 - cohort flow chart (round 5 / 2A redraw).

Grid layout: three columns, each = main-flow box (left) + exclusion box (right).
Only vertical downward arrows inside a column; exclusion boxes linked to the main
arrow by a short horizontal dashed line (STROBE style).
A geometry self-check is run after drawing.

Usage: python make_flow_v4.py <out_dir>
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42

W, H = 13.0, 9.5
fig = plt.figure(figsize=(W, H), dpi=300, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# ---------------- grid ----------------
MARG = 0.30
COLW = (W - 2 * MARG) / 3            # 4.133 in
MAIN_DX, MAIN_W = 0.10, 2.45         # main box inside column
EXC_DX, EXC_W = 2.78, 1.27           # exclusion box inside column (right edge = col + 4.05)
BOX_H = 1.00
ROW_TOP = {1: 8.25, 2: 5.85, 3: 3.45}   # top y of rows 1-3 (identical across columns)
EXC_H = 0.80
TITLE_Y = 8.85
BAND = (0.70, 1.40)                     # bottom full-width band y-range

BLUE_F, BLUE_E = "#eef4fb", "#1f4e79"
RED = "#b2182b"
GREY_F = "#e9e9e9"

def colx(c):
    return MARG + c * COLW

boxes, texts, arrows, dashes = [], [], [], []   # for self-check

def box(c, row, label, kind="main"):
    x = colx(c) + MAIN_DX; y = ROW_TOP[row] - BOX_H
    fc, ec, lw, fw = "white", "black", 1.0, "normal"
    if kind == "final":
        fc, ec, lw, fw = BLUE_F, BLUE_E, 1.6, "bold"
    elif kind == "grey":
        fc = GREY_F
    p = FancyBboxPatch((x, y), MAIN_W, BOX_H, boxstyle="square,pad=0", fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    t = ax.text(x + MAIN_W / 2, y + BOX_H / 2, label, ha="center", va="center", fontsize=10,
                fontweight=fw, color=BLUE_E if kind == "final" else "black", zorder=3, linespacing=1.3)
    boxes.append(dict(id=f"c{c}r{row}", x0=x, x1=x + MAIN_W, y0=y, y1=y + BOX_H, col=c))
    texts.append(dict(t=t, box=f"c{c}r{row}"))

def arrow(c, row_from, row_to):
    xc = colx(c) + MAIN_DX + MAIN_W / 2
    y0 = ROW_TOP[row_from] - BOX_H; y1 = ROW_TOP[row_to]
    a = FancyArrowPatch((xc, y0), (xc, y1), arrowstyle="-|>", mutation_scale=14, lw=1.2,
                        color="black", shrinkA=0, shrinkB=0, zorder=1)
    ax.add_patch(a)
    arrows.append(dict(x=xc, y0=y1, y1=y0, col=c))

def exclusion(c, row_above, label):
    """Exclusion box centred in the gap below row_above, right of the main flow."""
    gap_top = ROW_TOP[row_above] - BOX_H; gap_bot = ROW_TOP[row_above + 1]
    ym = (gap_top + gap_bot) / 2
    x = colx(c) + EXC_DX; y = ym - EXC_H / 2
    ax.add_patch(FancyBboxPatch((x, y), EXC_W, EXC_H, boxstyle="square,pad=0", fc="white",
                                ec=RED, lw=1.0, zorder=2))
    t = ax.text(x + EXC_W / 2, ym, label, ha="center", va="center", fontsize=8.5, color=RED,
                zorder=3, linespacing=1.25)
    xa = colx(c) + MAIN_DX + MAIN_W / 2
    ax.plot([xa, x], [ym, ym], ls=(0, (3, 2)), lw=1.0, color=RED, zorder=1)
    bid = f"c{c}x{row_above}"
    boxes.append(dict(id=bid, x0=x, x1=x + EXC_W, y0=y, y1=y + EXC_H, col=c))
    texts.append(dict(t=t, box=bid))
    dashes.append(dict(x0=xa, x1=x, y=ym, col=c))

titles = ["MIMIC-IV (2008\u20132019)", "MIMIC-III (2001\u20132008)", "eICU-CRD (2014\u20132015)"]
for c, s in enumerate(titles):
    t = ax.text(colx(c) + MAIN_DX + MAIN_W / 2, TITLE_Y, s, ha="center", va="center",
                fontsize=11, fontweight="bold")
    texts.append(dict(t=t, box=None, col=c))

ABI = "Acute brain injury\nextubation attempts\n"
# MIMIC-IV
box(0, 1, ABI + "3,970"); arrow(0, 1, 2)
exclusion(0, 1, "\u2212764\npalliative\nextubation")
box(0, 2, "3,206"); arrow(0, 2, 3)
exclusion(0, 2, "\u221256\n<3 days post-\nextubation SOFA")
box(0, 3, "3,150 analysed", "final")
# MIMIC-III CareVue (row 2 intentionally empty: single exclusion step)
box(1, 1, ABI + "822"); arrow(1, 1, 3)
exclusion(1, 1, "\u221212\nno post-\nextubation SOFA")
box(1, 3, "810 analysed", "final")
note = ax.text(colx(1) + MAIN_DX, ROW_TOP[3] - BOX_H - 0.14,
               "170 comfort-care deaths: retained in class estimation,\nexcluded from outcome comparisons",
               ha="left", va="top", fontsize=8.5, style="italic", color="#555555", linespacing=1.25)
texts.append(dict(t=note, box=None, col=1))
# eICU-CRD
box(2, 1, ABI + "3,756", "grey"); arrow(2, 1, 2)
exclusion(2, 1, "\u22123,102\nextubation time\nnot anchored")
box(2, 2, "654 with extubation\ntime anchored (ventilation\nstate algorithm)"); arrow(2, 2, 3)
exclusion(2, 2, "\u221219\npalliative\nextubation")
box(2, 3, "635 analysed", "final")

# bottom full-width band (no connectors)
bx0, bx1 = MARG + MAIN_DX, W - MARG - 0.05
ax.add_patch(FancyBboxPatch((bx0, BAND[0]), bx1 - bx0, BAND[1] - BAND[0], boxstyle="square,pad=0",
                            fc=BLUE_F, ec=BLUE_E, lw=1.6, zorder=2))
bt = ax.text((bx0 + bx1) / 2, sum(BAND) / 2, "Total analysed: 4,595 patients (3,150 + 810 + 635)",
             ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE_E, zorder=3)
boxes.append(dict(id="band", x0=bx0, x1=bx1, y0=BAND[0], y1=BAND[1], col=None))
texts.append(dict(t=bt, box="band"))

# ---------------- self-check ----------------
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax.transData.inverted()
def bb(t):
    e = t.get_window_extent(r); (x0, y0), (x1, y1) = inv.transform([[e.x0, e.y0], [e.x1, e.y1]])
    return x0, x1, y0, y1
B = {b["id"]: b for b in boxes}
chk = {}
PAD = 0.05
bad = []
for d in texts:
    x0, x1, y0, y1 = bb(d["t"])
    if d["box"]:
        b = B[d["box"]]
        if not (x0 >= b["x0"] + PAD and x1 <= b["x1"] - PAD and y0 >= b["y0"] + PAD and y1 <= b["y1"] - PAD):
            bad.append((d["t"].get_text()[:30], round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2), b["id"]))
    else:
        c = d["col"]
        if not (x0 >= colx(c) and x1 <= colx(c) + COLW and 0 <= y0 and y1 <= H):
            bad.append((d["t"].get_text()[:30], "outside column"))
chk["text_inside_box"] = (not bad, bad)
ov = []
L = list(B.values())
for i in range(len(L)):
    for j in range(i + 1, len(L)):
        a, b = L[i], L[j]
        if a["x0"] < b["x1"] and b["x0"] < a["x1"] and a["y0"] < b["y1"] and b["y0"] < a["y1"]:
            ov.append((a["id"], b["id"]))
chk["no_box_overlap"] = (not ov, ov)
cross = []
for a in arrows:
    for b in L:
        if b["x0"] < a["x"] < b["x1"] and a["y0"] + 1e-6 < b["y1"] - 1e-6 and b["y0"] + 1e-6 < a["y1"] - 1e-6:
            cross.append((a["col"], b["id"]))
for d in dashes:
    for b in L:
        if b["y0"] < d["y"] < b["y1"] and d["x0"] < b["x1"] and b["x0"] < d["x1"] - 1e-6:
            cross.append(("dash", d["col"], b["id"]))
chk["connectors_do_not_cross_boxes"] = (not cross, cross)
al = []
for row in (1, 2, 3):
    rowb = [B[k] for k in B if k.endswith(f"r{row}")]
    if len({(round(b["y0"], 6), round(b["y1"], 6), round(b["x1"] - b["x0"], 6)) for b in rowb}) != 1:
        al.append(row)
exb = [b for k, b in B.items() if "x" in k[2:]]
if len({round(b["x1"] - b["x0"], 6) for b in exb}) != 1 or len({round(b["y1"] - b["y0"], 6) for b in exb}) != 1:
    al.append("exclusion size")
chk["grid_aligned"] = (not al, al)
inv_col = [b["id"] for b in exb if b["x1"] > colx(b["col"]) + COLW or b["x0"] < colx(b["col"])]
chk["exclusion_within_column"] = (not inv_col, inv_col)
chk["within_canvas"] = (all(0 <= b["x0"] and b["x1"] <= W and 0 <= b["y0"] and b["y1"] <= H for b in L), [])

fig.savefig(f"{OUT}/SuppFig1_cohort_flow.png", dpi=300, facecolor="white")
fig.savefig(f"{OUT}/SuppFig1_cohort_flow.pdf", facecolor="white")
res = {k: {"pass": v[0], "detail": v[1]} for k, v in chk.items()}
json.dump(res, open(f"{OUT}/SuppFig1_selfcheck.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
