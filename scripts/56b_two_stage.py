# -*- coding: utf-8 -*-
# 56b — 两阶段架构: 死亡头 + 轨迹头 分开建模再组合 (y_B = class{1,4} ∪ death5d)
#   临床逻辑: "这个病人会不会死"(futility轴) vs "拔了会不会恶化"(生理轴) 是两个机制
#   Head1: death5d 模型 (全队列)
#   Head2: 轨迹模型 (class∈{1,4} 或 class==1, 存活者训练, 全队列应用)
#   组合: rank-max / LR stack → y_B AUROC
import sys, json, time, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results"); t0 = time.time()

F = pd.read_parquet(PRJ/"abi_feature_matrix_MASTER_v4.parquet")
pp = pd.read_csv(PRJ/"traj_sofa_posteriors.csv")[["stay_id","class"]]
coh = pd.read_parquet(PRJ/"abi_full_cohort.parquet")[["stay_id","death5d","rein5d"]]
sub = pp.merge(coh, on="stay_id", how="left").set_index("stay_id")
F = F.loc[F.index.isin(sub.index)]
sub = sub.loc[F.index]
X = F.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
cov = X.notna().mean(); X = X[[c for c in X.columns if cov[c]>=0.5]]
from sklearn.impute import SimpleImputer
Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
tuned = json.load(open(PRJ/"dyn6h_tuned.json"))["best_params"]
groups = X.index.get_level_values("stay_id").values
gkf = GroupKFold(n_splits=5)
rng = np.random.default_rng(42)

def oof(y, Xd):
    p = np.full(len(y), np.nan)
    for tr, te in gkf.split(Xd, y, groups):
        m = CatBoostClassifier(**tuned, random_seed=42, verbose=0, auto_class_weights='Balanced', thread_count=-1)
        m.fit(Xd.iloc[tr], y[tr]); p[te] = m.predict_proba(Xd.iloc[te])[:,1]
    return p

def rank01(v):
    r = pd.Series(v).rank().values
    return (r - r.min())/(r.max()-r.min())

dth = (sub.death5d.values==1).astype(int)
yB = (sub["class"].isin([1,4]).values | (sub.death5d.values==1)).astype(int)
surv = (sub.death5d.values==0)

# 快速特征集: 每头独立 SHAP top 30 + 已选特征并集
from sklearn.ensemble import RandomForestClassifier
def shaptop(y, k=30, rows=None):
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(Ximp.iloc[rows] if rows is not None else Ximp, y)
    return pd.Series(rf.feature_importances_, index=Ximp.columns).sort_values(ascending=False).head(k).index.tolist()

res = {}
# Head1: death
fd = shaptop(dth); pd_ = oof(dth, Ximp[fd])
auc_d = roc_auc_score(dth, pd_)
print(f"Head1 死亡: {len(fd)} feats AUROC={auc_d:.4f}", flush=True)
# Head2a: class1+4 among survivors
y14 = (sub["class"].isin([1,4]).values).astype(int)
surv_idx = np.where(surv)[0]
f14 = shaptop(y14[surv], rows=surv_idx); p14 = oof(y14, Ximp[f14])
auc14 = roc_auc_score(y14[surv], p14[surv])
print(f"Head2a 轨迹1+4(存活者评): {len(f14)} feats AUROC={auc14:.4f}", flush=True)
# Head2b: class1 only
y1 = (sub["class"].values==1).astype(int)
f1 = shaptop(y1); p1 = oof(y1, Ximp[f1])
auc1 = roc_auc_score(y1, p1)
print(f"Head2b 轨迹1: {len(f1)} feats AUROC={auc1:.4f}", flush=True)

# 组合 → y_B
for name, pdt, ptr in [("max_d_c14", pd_, p14), ("max_d_c1", pd_, p1)]:
    comb = np.maximum(rank01(pdt), rank01(ptr))
    auc = roc_auc_score(yB, comb)
    res[name] = round(float(auc),4)
    print(f"组合[{name}] y_B AUROC={auc:.4f}", flush=True)
# LR stack (嵌套: 折内拟合stack防泄漏)
for name, pdt, ptr in [("stack_d_c14", pd_, p14), ("stack_d_c1", pd_, p1)]:
    S = np.column_stack([pdt, ptr])
    p = np.full(len(yB), np.nan)
    for tr, te in gkf.split(S, yB, groups):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(S[tr], yB[tr]); p[te] = lr.predict_proba(S[te])[:,1]
    auc = roc_auc_score(yB, p)
    res[name] = round(float(auc),4)
    print(f"组合[{name}] y_B AUROC={auc:.4f}", flush=True)

json.dump({"head_death_auroc":round(float(auc_d),4),"head_c14_surv_auroc":round(float(auc14),4),
    "head_c1_auroc":round(float(auc1),4),"yB_combos":res,
    "note":"两阶段: 各头独立SHAP top30 CatBoost 5折OOF; yB=class14∪death5d"},
    open(PRJ/"traj_two_stage_results.json","w"), ensure_ascii=False, indent=1)
print(f"DONE ({time.time()-t0:.0f}s)")
