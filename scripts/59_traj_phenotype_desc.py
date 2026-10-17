# -*- coding: utf-8 -*-
# ============================================================
# 59 — 表型发现开篇: 干净队列四轨迹类的临床刻画 + 轨迹主图
#   (09 GCS 论文同款位置 / Zhang Fig1b 同构: 均值曲线+95%CI+n at risk)
#   输出: traj_phenotype_table.csv (类×基线×结局, 带p值)
#         fig_traj_phenotype.png/pdf (主图候选 Fig 2a)
# ============================================================
import sys, json, time, warnings, numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results"); t0 = time.time()
M4 = "E:/TBI subtype/data/mimic-iv-3.1"

pp = pd.read_csv(PRJ/"traj_clean_posteriors.csv").set_index("stay_id")
clean = pd.read_parquet(PRJ/"clean_cohort_v1.parquet").set_index("stay_id")
daily = pd.read_parquet(PRJ/"_sofa_post_daily.parquet")
daily = daily[daily.stay_id.isin(pp.index)]

# 远期结局: ICU LOS / 院内死亡 / 拔管后 ICU 时长
con = duckdb.connect()
out = con.execute(f"""
SELECT s.stay_id, s.los AS icu_los,
  a.hospital_expire_flag AS hosp_mort,
  epoch(CAST(a.dischtime AS TIMESTAMP))/86400.0 - epoch(CAST(s.intime AS TIMESTAMP))/86400.0 AS hosp_los
FROM read_csv_auto('{M4}/icu/icustays.csv.gz') s
JOIN read_csv_auto('{M4}/hosp/admissions.csv.gz') a ON s.hadm_id=a.hadm_id
""").df().set_index("stay_id").drop(columns=["hosp_los"])

F = pd.read_parquet(PRJ/"abi_feature_matrix_MASTER_v4.parquet")
BASE = F[["anchor_age","male","gcs_motor_ext","pre_inv_h"]]
BASE = BASE.loc[pp.index.intersection(BASE.index)]

sub_map = pd.read_parquet(PRJ/"abi_full_cohort.parquet")[["stay_id","subtype"]].set_index("stay_id").subtype
D = pp.join(clean[["rein5d","death5d","niv_new","vaso_new","rrt_new"]])
D = D.join(out, how="left").join(BASE).join(sub_map)
# 拔管后 ICU 时长 (icu_los - pre_inv_h/24 近似, 只用总 LOS 报告)
NAMES = {1:"Late deterioration (迟发恶化)", 2:"Stable (稳定)", 3:"Recovery (高基线恢复)", 4:"Persistent severe (持续危重)"}
order = [1,2,3,4]

# ===== 1) 每日类曲线 (均值 + bootstrap CI + n at risk) =====
curves = {}
for cl in order:
    ids = pp[pp["class"]==cl].index
    g = daily[daily.stay_id.isin(ids)]
    m = g.groupby("day").sofa.mean(); n = g.groupby("day").sofa.size()
    curves[cl] = {"mean": m, "n": n}
    print(f"class{cl} n-at-day:", dict(n))

# ===== 2) 刻画表 =====
from scipy.stats import kruskal, chi2_contingency
rows = []
def add_row(label, fn, fmt="{:.1f}", pct=False):
    vals = {cl: fn(D[D["class"]==cl]) for cl in order}
    try:
        if pct:
            tab = np.array([[ (D[D["class"]==cl][label.split()[0]] if False else 0) for cl in order]])
            col = D.assign(_v=D[label]).pivot(columns="class", values="_v")
            chi, p, _, _ = chi2_contingency(pd.crosstab(D["_v"] if "_v" in D else D[label], D["class"]).values) if D[label].dtype==bool else (np.nan, np.nan, np.nan, np.nan)
        else:
            groups = [D[D["class"]==cl][label].dropna() for cl in order]
            _, p = kruskal(*groups)
    except Exception as e:
        p = np.nan
    rows.append({"characteristic": label,
        **{f"class{cl}": fmt.format(vals[cl]) for cl in order}, "p": round(p,4) if pd.notna(p) else np.nan})

add_row("n", lambda d: len(d), "{:.0f}")
add_row("age, yr", lambda d: d.anchor_age.mean())
add_row("male, %", lambda d: d.male.mean()*100)
add_row("motor-GCS at extubation", lambda d: d.gcs_motor_ext.mean())
add_row("invasive vent before ext, h", lambda d: d.pre_inv_h.mean())
for st in ["TBI","SAH","ICH","IS_I64"]:
    add_row(f"{st}, %", lambda d, st=st: d.subtype.eq(st).mean()*100)
add_row("SOFA day0 (at ext)", lambda d: daily[(daily.stay_id.isin(d.index))&(daily.day==0)].sofa.mean())
add_row("reintubation 5d, %", lambda d: d.rein5d.mean()*100)
add_row("new NIV/HFNC, %", lambda d: d.niv_new.mean()*100)
add_row("new vasopressor, %", lambda d: d.vaso_new.mean()*100)
add_row("new RRT, %", lambda d: d.rrt_new.mean()*100)
add_row("death 5d, %", lambda d: d.death5d.mean()*100)
add_row("ICU LOS, d", lambda d: d.icu_los.mean())
add_row("hospital mortality, %", lambda d: d.hosp_mort.mean()*100)
T = pd.DataFrame(rows)
T.to_csv(PRJ/"traj_phenotype_table.csv", index=False)
print(T.to_string(index=False))

# ===== 3) 主图: 均值曲线 + 95%CI 带 =====
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
COL = {1:"#E64B35", 2:"#3B6FB6", 3:"#00A087", 4:"#7E5183"}
fig, ax = plt.subplots(figsize=(7.0,4.4), dpi=300)
rng = np.random.default_rng(42)
for cl in order:
    ids = pp[pp["class"]==cl].index
    g = daily[daily.stay_id.isin(ids)]
    days = np.arange(0,6)
    boot = []
    for b in range(200):
        samp = rng.choice(ids, len(ids), replace=True)
        gb = g[g.stay_id.isin(samp)]
        boot.append([gb[gb.day==d].sofa.mean() if (gb.day==d).any() else np.nan for d in days])
    boot = np.array(boot)
    m = np.nanmean(boot, axis=0)
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    n = curves[cl]["n"]
    ax.plot(days, m, color=COL[cl], lw=2.2, zorder=3,
            label=f"{NAMES[cl]}, n={len(ids)} ({len(ids)/len(pp)*100:.0f}%)")
    ax.fill_between(days, lo, hi, color=COL[cl], alpha=0.15, lw=0)
ax.set_xlabel("Days after extubation"); ax.set_ylabel("Total SOFA score")
ax.set_xticks(days); ax.set_xticklabels([f"D{d}" for d in days])
ax.spines[["top","right"]].set_visible(False)
# n at risk 注释
nat = " | ".join(f"D{d}: {sum(curves[cl]['n'].get(d,0) for cl in order)}" for d in [0,2,4,5])
ax.text(0.99, 0.02, f"patients with data  {nat}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555")
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0,1.02))
fig.tight_layout()
fig.savefig(PRJ/"fig_traj_phenotype.png", dpi=300, bbox_inches="tight")
fig.savefig(PRJ/"fig_traj_phenotype.pdf", bbox_inches="tight")
print(f"图已存 ({time.time()-t0:.0f}s)")
