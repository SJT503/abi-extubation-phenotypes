# -*- coding: utf-8 -*-
# 59c — k=5 (D0-D4, 无伪影) 表型刻画 + 轨迹主图 + 个体斜率验证
import sys, json, time, warnings, numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results"); t0 = time.time()
M4 = "E:/TBI subtype/data/mimic-iv-3.1"

pp = pd.read_csv(PRJ/"traj_d04_k6_posteriors.csv").set_index("stay_id")
clean = pd.read_parquet(PRJ/"clean_cohort_v1.parquet").set_index("stay_id")
daily = pd.read_parquet(PRJ/"_sofa_post_daily.parquet")
daily = daily[daily.stay_id.isin(pp.index)]
con = duckdb.connect()
out = con.execute(f"""
SELECT s.stay_id, s.los AS icu_los, a.hospital_expire_flag AS hosp_mort
FROM read_csv_auto('{M4}/icu/icustays.csv.gz') s
JOIN read_csv_auto('{M4}/hosp/admissions.csv.gz') a ON s.hadm_id=a.hadm_id
""").df().set_index("stay_id")
F = pd.read_parquet(PRJ/"abi_feature_matrix_MASTER_v4.parquet")[["anchor_age","male","gcs_motor_ext","pre_inv_h"]]
F = F.loc[F.index.isin(pp.index)]
sub_map = pd.read_parquet(PRJ/"abi_full_cohort.parquet")[["stay_id","subtype"]].set_index("stay_id").subtype

D = pp.join(clean[["rein5d","death5d","niv_new","vaso_new","rrt_new"]]).join(out, how="left").join(F).join(sub_map)
# 个体斜率
sl = {}
for sid, g in daily[daily.day<=4].groupby("stay_id"):
    if len(g)>=3: sl[sid] = np.polyfit(g.day.values, g.sofa.values, 1)[0]
D = D.join(pd.Series(sl, name="ind_slope"))

# 类均值 + 斜率验证
print("=== k=5 类轨迹与特征 ===")
info = {}
for cl in sorted(D["class"].unique()):
    d = D[D["class"]==cl]
    tr = daily[daily.stay_id.isin(d.index)].groupby("day").sofa.mean()
    info[int(cl)] = {"n": len(d), "traj": [round(float(tr.get(i,np.nan)),1) for i in range(5)],
        "ind_slope_mean": round(float(d.ind_slope.mean()),3), "hosp_mort": round(float(d.hosp_mort.mean())*100,1),
        "rein": round(float(d.rein5d.mean())*100,1)}
    print(f"class{cl}: n={len(d)} ({len(d)/len(D)*100:.0f}%) SOFA {'->'.join(f'{tr.get(i,np.nan):.1f}' for i in range(5))} | 个体斜率 {d.ind_slope.mean():+.2f}/d | 重插 {d.rein5d.mean()*100:.0f}% 院死 {d.hosp_mort.mean()*100:.0f}%")
json.dump(info, open(PRJ/"traj_k6_summary.json","w"), ensure_ascii=False, indent=1)

# 主图 (D0-D4, 均值+95%CI bootstrap)
NAMES = {}
for cl in sorted(D["class"].unique()):
    d = D[D["class"]==cl]
    NAMES[cl] = f"Class {cl} (n={len(d)}, {len(d)/len(D)*100:.0f}%)"
COL = {1:"#3B6FB6", 2:"#91D1C2", 3:"#00A087", 4:"#8B6F47", 5:"#E64B35", 6:"#7E5183"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
fig, ax = plt.subplots(figsize=(8.2,4.2), dpi=300)
rng = np.random.default_rng(42); days = np.arange(0,5)
for cl in sorted(D["class"].unique()):
    ids = D[D["class"]==cl].index
    g = daily[daily.stay_id.isin(ids)]
    boot = []
    for b in range(300):
        samp = rng.choice(ids, len(ids), replace=True)
        gb = g[g.stay_id.isin(samp)]
        boot.append([gb[gb.day==d0].sofa.mean() for d0 in days])
    boot = np.array(boot); m = boot.mean(0)
    lo, hi = np.percentile(boot,2.5,axis=0), np.percentile(boot,97.5,axis=0)
    ax.plot(days, m, color=COL[cl], lw=2.2, label=NAMES[cl], zorder=3)
    ax.fill_between(days, lo, hi, color=COL[cl], alpha=0.15, lw=0)
ax.set_xlabel("Days after extubation"); ax.set_ylabel("Total SOFA score")
ax.set_xticks(days); ax.set_xticklabels([f"Day {d}" for d in days])
ax.set_ylim(0, None); ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
fig.savefig(PRJ/"fig_traj_k6.png", dpi=300, bbox_inches="tight")
fig.savefig(PRJ/"fig_traj_k6.pdf", bbox_inches="tight")
print(f"图已存 ({time.time()-t0:.0f}s)")
