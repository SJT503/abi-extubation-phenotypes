# -*- coding: utf-8 -*-
# 86s — SuppFig1 队列流程图 v2（修正版）
#   修正: ①三栏严格分隔 ②M3 805→−51→754 补标注 ③eICU 口径与 v3 Methods 一致
#   数字全部来自本地核实: M4 3970/-764/3206/-56/3150; M3 805/-51/754(+104 comfort 注记);
#   eICU 3756/654 锚定/-19/635; 合计 4539
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(14, 10), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")

C_MAIN = "#FFFFFF"; C_EDGE = "#1a1a2e"; C_EXCL = "#fdf0ef"; C_EXCL_E = "#c0392b"
C_GREY = "#f2f2f2"; C_GREY_E = "#8a8a8a"; C_FINAL = "#eef4fb"; C_FINAL_E = "#1f4e79"

def box(x, y, w, h, text, fc=C_MAIN, ec=C_EDGE, fs=9.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.3, linestyle=ls, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.2), zorder=1)

# ===== 三栏标题 =====
COLS = [(0.4, 4.1, "MIMIC-IV\n(2008–2019, n=3,150)"),
        (5.0, 4.1, "MIMIC-III\n(2001–2012, n=754)"),
        (9.6, 4.1, "eICU-CRD\n(2014–2015, n=635)")]
for x, w, t in COLS:
    ax.text(x + w/2, 9.55, t, ha="center", va="center", fontsize=11, fontweight="bold")
# 栏分隔虚线
for lx in (4.75, 9.35):
    ax.plot([lx, lx], [0.55, 9.25], ls=":", color="#bbbbbb", lw=1, zorder=0)

# ===== MIMIC-IV 栏 =====
x, w = 0.4, 4.1
box(x, 8.35, w, 0.85, "Acute brain injury extubation attempts\n3,970", fs=10)
arrow(x+w/2, 8.35, x+w/2, 7.75)
box(x+w+0.15, 7.85, 1.9, 0.6, "– 764\npalliative extubation", fc=C_EXCL, ec=C_EXCL_E, fs=8.5)
ax.plot([x+w, x+w+0.15], [8.75, 8.15], color=C_EXCL_E, lw=1.1)
box(x, 6.95, w, 0.75, "3,206", fs=10)
arrow(x+w/2, 6.95, x+w/2, 6.35)
box(x+w+0.15, 6.45, 1.9, 0.6, "– 56\n< 3 days post-extubation\nSOFA data", fc=C_EXCL, ec=C_EXCL_E, fs=8)
ax.plot([x+w, x+w+0.15], [7.35, 6.75], color=C_EXCL_E, lw=1.1)
box(x, 5.55, w, 0.75, "3,150 analysed", fc=C_FINAL, ec=C_FINAL_E, fs=10.5, bold=True)

# ===== MIMIC-III 栏 =====
x, w = 5.0, 4.1
box(x, 8.35, w, 0.85, "Acute brain injury extubation attempts\n805", fs=10)
arrow(x+w/2, 8.35, x+w/2, 7.75)
box(x+w+0.15, 7.85, 1.9, 0.6, "– 51\nno post-extubation\nSOFA measurement", fc=C_EXCL, ec=C_EXCL_E, fs=8)
ax.plot([x+w, x+w+0.15], [8.75, 8.15], color=C_EXCL_E, lw=1.1)
box(x, 6.95, w, 0.75, "754 analysed", fc=C_FINAL, ec=C_FINAL_E, fs=10.5, bold=True)
box(x, 5.35, w, 1.15, "104 comfort-care deaths:\nretained in class estimation,\nexcluded from outcome comparisons",
    fc=C_GREY, ec=C_GREY_E, fs=8.5, ls="--")
arrow(x+w/2, 6.95, x+w/2, 6.55)

# ===== eICU-CRD 栏 =====
x, w = 9.6, 4.1
box(x, 8.35, w, 0.85, "Acute brain injury extubation attempts\n3,756", fc=C_GREY, ec=C_GREY_E, fs=10)
arrow(x+w/2, 8.35, x+w/2, 7.75)
ax.text(x+w/2+0.12, 8.07, "extubation time anchored\n(ventilation state algorithm)", fontsize=7.5,
        ha="left", va="center", color="#555555", style="italic")
box(x, 6.95, w, 0.75, "654 with anchored extubation time", fs=9.5)
arrow(x+w/2, 6.95, x+w/2, 6.35)
box(x+w+0.15, 6.45, 1.9, 0.6, "– 19\npalliative extubation", fc=C_EXCL, ec=C_EXCL_E, fs=8.5)
ax.plot([x+w, x+w+0.15], [7.35, 6.75], color=C_EXCL_E, lw=1.1)
box(x, 5.55, w, 0.75, "635 analysed", fc=C_FINAL, ec=C_FINAL_E, fs=10.5, bold=True)

# ===== 底部合计 =====
box(3.4, 3.9, 7.2, 0.85, "Total analysed: 4,539 patients\n(3,150 + 754 + 635)",
    fc=C_FINAL, ec=C_FINAL_E, fs=11, bold=True)
arrow(2.45, 5.55, 4.6, 4.8); arrow(7.05, 5.35, 7.0, 4.8); arrow(11.65, 5.55, 9.4, 4.8)

# 脚注
ax.text(0.4, 3.3, "ABI, acute brain injury (traumatic brain injury, intracerebral haemorrhage, subarachnoid\n"
        "haemorrhage or ischaemic stroke); SOFA, Sequential Organ Failure Assessment. Palliative extubation:\n"
        "documented withdrawal of life-sustaining therapy or comfort measures, or death within 5 days without\n"
        "escalation of organ support. First ICU stay and first extubation per patient in each database.",
        fontsize=8, va="top", color="#333333", linespacing=1.5)

out = "E:/TBI subtype/08_extubation_prediction/biomni_output_v3/ICM_manuscript/v3/figures"
fig.savefig(f"{out}/SuppFig1_cohort_flow.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{out}/SuppFig1_cohort_flow.pdf", bbox_inches="tight", facecolor="white")
print("saved SuppFig1_cohort_flow.png/pdf (v2 修正版)")
