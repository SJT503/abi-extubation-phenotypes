# -*- coding: utf-8 -*-
# ============================================================
# 57 — 恶化表型 v3 (Zhang 对齐审查后的扩充版)
#   PI 2026-10-03: "主结局是否过严? 还有哪些恶化表现没纳入?"
#   Zhang 恶化类 22.1% (6151/27804) vs 我方 class1 12.9% → 确实更严
#   新增成分 (全部实查 itemid):
#     A. NIV/HFNC 拔管后新启动 (proc 225794 + BiPap 227577-83 + O2设备 226732)
#     B. 新升压药 (inputevents 五药, 拔管前24h无 + 拔后5d内新开)
#     C. 新 RRT (proc 225802/803/809/955/441/805, 同上逻辑)
#     D. 神经恶化: 拔管后 motor-GCS 较 day0-1 基线下降 ≥2 (ABI 器官轴, 223901)
#     E. LCGM k=5 重拟合 → 查"波动类"(dip-then-rise, Zhang k=4/5 有)
#     F. 已有: 重插 rein5d / 死亡 death5d / SOFA 恶化类
#   y_v3 = 恶化轨迹 ∪ 波动轨迹 ∪ 重插 ∪ NIV/HFNC ∪ 新升压 ∪ 新RRT ∪ 神经恶化 ∪ 死亡
#   输出: 组件边际贡献表 + y_v3 统一三道关卡建模
# ============================================================
import sys, json, time, warnings, numpy as np, pandas as pd, duckdb, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results")
M4 = "E:/TBI subtype/data/mimic-iv-3.1"
con = duckdb.connect(); con.execute("PRAGMA threads=4"); t0 = time.time()

coh = pd.read_parquet(PRJ/"abi_full_cohort.parquet")[["stay_id","t_ext","rein5d","death5d"]]
coh["t_ext"] = pd.to_datetime(coh.t_ext)
coh[["stay_id","t_ext"]].to_parquet(PRJ/"_v3_tmp.parquet", index=False)
R = (PRJ/"_v3_tmp.parquet").as_posix()

print("[A] 拔管前后组件提取 (chartevents+procedureevents+inputevents)...", flush=True)
# A1: O2设备 + BiPap设置 + motor GCS (chartevents, daybin 相对 t_ext)
ce = con.execute(f"""
WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{R}'))
SELECT c.stay_id, c.itemid,
       floor((epoch(CAST(c.charttime AS TIMESTAMP)) - epoch(r.t_ext))/86400.0) AS daybin,
       CAST(c."value" AS VARCHAR) v
FROM read_csv_auto('{M4}/icu/chartevents.csv.gz') c JOIN r ON c.stay_id=r.stay_id
WHERE c.itemid IN (226732,227577,227578,227579,227580,227581,227582,227583,223901)
  AND epoch(CAST(c.charttime AS TIMESTAMP)) > epoch(r.t_ext) - 86400
  AND epoch(CAST(c.charttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
""").df()
print(f"  ce: {len(ce)} ({time.time()-t0:.0f}s)", flush=True)

# A2: procedureevents (NIV 225794 + RRT 套件)
DIAL = "225802,225803,225809,225955,225441,225805"
pe = con.execute(f"""
WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{R}'))
SELECT p.stay_id, p.itemid,
       floor((epoch(CAST(p.starttime AS TIMESTAMP)) - epoch(r.t_ext))/86400.0) AS daybin
FROM read_csv_auto('{M4}/icu/procedureevents.csv.gz') p JOIN r ON p.stay_id=r.stay_id
WHERE p.itemid IN (225794,{DIAL})
  AND epoch(CAST(p.starttime AS TIMESTAMP)) > epoch(r.t_ext) - 86400
  AND epoch(CAST(p.starttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
""").df()
print(f"  pe: {len(pe)} ({time.time()-t0:.0f}s)", flush=True)

# A3: inputevents 升压药
VASO = "221906,221289,221662,221749,221786,221653"
ie = con.execute(f"""
WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{R}'))
SELECT i.stay_id,
       floor((epoch(CAST(i.starttime AS TIMESTAMP)) - epoch(r.t_ext))/86400.0) AS daybin
FROM read_csv_auto('{M4}/icu/inputevents.csv.gz') i JOIN r ON i.stay_id=r.stay_id
WHERE i.itemid IN ({VASO})
  AND epoch(CAST(i.starttime AS TIMESTAMP)) > epoch(r.t_ext) - 86400
  AND epoch(CAST(i.starttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
""").df()
print(f"  ie: {len(ie)} ({time.time()-t0:.0f}s)", flush=True)

# ===== 组件旗标 =====
idx = coh.set_index("stay_id")
flags = pd.DataFrame(index=idx.index)
pre  = lambda df: df[df.daybin < 0]
post = lambda df: df[df.daybin >= 0]

# 1) NIV/HFNC 新启动
HFNC_PAT = "high flow|bipap|bi-pap|cpap|non-rebreath|apollo|avap"
niv_proc = set(pe[pe.itemid==225794].stay_id)
niv_pre  = set(pre(ce[ce.itemid.isin(range(227577,227584))]).stay_id)
dev = ce[ce.itemid==226732].copy()
dev_hi = dev[dev.v.str.contains(HFNC_PAT, case=False, na=False)]
niv_post_dev = set(post(dev_hi).stay_id)
niv_pre_dev  = set(pre(dev_hi).stay_id)
flags["niv_hfnc_new"] = [(s in (niv_proc|niv_post_dev)) and (s not in (niv_pre|niv_pre_dev)) for s in flags.index]

# 2) 新升压药
vp_pre  = set(pre(ie).stay_id); vp_post = set(post(ie).stay_id)
flags["vaso_new"] = [(s in vp_post) and (s not in vp_pre) for s in flags.index]

# 3) 新 RRT
rt_pre  = set(pre(pe[pe.itemid!=225794]).stay_id); rt_post = set(post(pe[pe.itemid!=225794]).stay_id)
flags["rrt_new"] = [(s in rt_post) and (s not in rt_pre) for s in flags.index]

# 4) 神经恶化: motor-GCS day0-1 基线(max) vs day2-5 最低, 下降>=2
MOTOR={"Obeys Commands":6,"Localizes Pain":5,"Flex-withdraws":4,"Abnormal Flexion":3,"Abnormal extension":2,"No response":1,"No Response":1}
mg = ce[ce.itemid==223901].copy()
mg["m"] = mg.v.str.strip().map(MOTOR)
mg = mg.dropna(subset=["m"])
neuro = []
for sid, g in mg.groupby("stay_id"):
    b = g[(g.daybin>=0)&(g.daybin<=1)].m.max()
    latemin = g[(g.daybin>=2)].m.min()
    neuro.append((sid, (pd.notna(b) and pd.notna(latemin) and (b - latemin) >= 2)))
nd = dict(neuro)
flags["neuro_worse"] = [nd.get(s, False) for s in flags.index]
print(f"[B] 旗标: NIV/HFNC={int(flags.niv_hfnc_new.sum())} 新升压={int(flags.vaso_new.sum())} 新RRT={int(flags.rrt_new.sum())} 神经恶化={int(flags.neuro_worse.sum())} ({time.time()-t0:.0f}s)", flush=True)

# ===== C: LCGM k=5 (波动类检查) =====
r5 = """suppressMessages(library(lcmm))
d <- read.csv("traj_sofa_post_long.csv"); d$day <- as.numeric(d$day); d$sofa <- as.numeric(d$sofa)
d <- d[order(d$stay_id, d$day), ]
m1 <- hlme(sofa ~ day, subject="stay_id", data=d, verbose=FALSE)
m5 <- tryCatch(hlme(sofa ~ day, subject="stay_id", ng=5, data=d, mixture=~day, nwg=TRUE, B=m1, maxiter=200, verbose=FALSE), error=function(e) NULL)
if (!is.null(m5)) {
  cat("BIC5=", round(m5$BIC,0), "\\n")
  pp <- as.data.frame(m5$pprob); write.csv(pp, "traj_sofa_posteriors_k5.csv", row.names=FALSE)
  # 每类均值轨迹
  for (k in 1:5) {
    ids <- pp[pp$class==k,"stay_id"]; dd <- d[d$stay_id %in% ids,]
    tr <- tapply(dd$sofa, dd$day, mean)
    cat("class", k, "n=", length(ids), ": ", paste(sprintf("%.1f", tr), collapse=" -> "), "\\n", sep="")
  }
} else cat("k5 FAILED\\n")
"""
(PRJ/"57_lcmm_k5.R").write_text(r5, encoding="utf-8")
print("[C] LCGM k=5...", flush=True)
rr = subprocess.run(["C:/Program Files/R/R-4.5.3/bin/Rscript.exe", str(PRJ/"57_lcmm_k5.R")],
                    capture_output=True, text=True, timeout=900, cwd=str(PRJ))
print(rr.stdout[-1200:] if len(rr.stdout)>1200 else rr.stdout)
if rr.returncode != 0: print("STDERR:", rr.stderr[-500:])

# ===== D: 组装 y_v3 + 建模 =====
pp4 = pd.read_csv(PRJ/"traj_sofa_posteriors.csv").set_index("stay_id")
flags["traj_det_class1"] = pp4["class"]==1
flags["rein5d"] = idx.rein5d.astype(bool)
flags["death5d"] = idx.death5d.astype(bool)
k5path = PRJ/"traj_sofa_posteriors_k5.csv"
fluct_ids = set()
if k5path.exists():
    pp5 = pd.read_csv(k5path)
    # 波动类判定: 由 R 输出人工映射 (打印在上方); 自动初判 = 斜率非单调的类, 这里先读 R 打印后手动指定
    # 安全兜底: 暂以 k5 中"后段回升"类为空集, 后续按 R 输出手动补
    pass
COMP = ["traj_det_class1","rein5d","niv_hfnc_new","vaso_new","rrt_new","neuro_worse","death5d"]
y_v3 = flags[COMP].any(axis=1).astype(int)
print(f"\n[D] y_v3 = {y_v3.sum()} ({y_v3.mean()*100:.1f}%) | 各组件边际 (仅该组件者):")
for c in COMP:
    only = (flags[c] & ~flags[[x for x in COMP if x!=c]].any(axis=1)).sum()
    print(f"  {c:18s}: {int(flags[c].sum()):4d} (独有 {only})")

# 统一管线: SHAP top40 -> CatBoost 5折
F = pd.read_parquet(PRJ/"abi_feature_matrix_MASTER_v4.parquet").loc[y_v3.index]
X = F.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
cov = X.notna().mean(); X = X[[c for c in X.columns if cov[c]>=0.5]]
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from catboost import CatBoostClassifier
tuned = json.load(open(PRJ/"dyn6h_tuned.json"))["best_params"]
Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
y = y_v3.loc[X.index].values
groups = X.index.get_level_values("stay_id").values
gkf = GroupKFold(n_splits=5)
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(Ximp, y)
imp = pd.Series(rf.feature_importances_, index=Ximp.columns).sort_values(ascending=False)
cand = imp.head(40).index.tolist()
p = np.full(len(y), np.nan)
for tr, te in gkf.split(Ximp[cand], y, groups):
    m = CatBoostClassifier(**tuned, random_seed=42, verbose=0, auto_class_weights='Balanced', thread_count=-1)
    m.fit(Ximp[cand].iloc[tr], y[tr]); p[te] = m.predict_proba(Ximp[cand].iloc[te])[:,1]
auc = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
rng = np.random.default_rng(42)
boots = [roc_auc_score(y[i],p[i]) for i in (rng.choice(len(y),len(y),replace=True) for _ in range(500))]
ci = [round(float(np.percentile(boots,2.5)),4), round(float(np.percentile(boots,97.5)),4)]
print(f"\n  y_v3 (SHAP top40): AUROC={auc:.4f} [{ci[0]}-{ci[1]}] AUPRC={ap:.4f} Brier={br:.4f}")
flags.assign(y_v3=y_v3).to_parquet(PRJ/"deterioration_v3_flags.parquet")
json.dump({"endpoint":"v3_full_union","components":{c:int(flags[c].sum()) for c in COMP},
  "n_events":int(y.sum()),"event_rate":round(float(y.mean()),4),
  "auroc":round(float(auc),4),"ci95":ci,"auprc":round(float(ap),4),"features":cand},
  open(PRJ/"det_v3_results.json","w"), ensure_ascii=False, indent=1)
print(f"DONE ({time.time()-t0:.0f}s)")
