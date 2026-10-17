# -*- coding: utf-8 -*-
# ============================================================
# 56 — 拔管后5天不良轨迹终点（PI 2026-10-03 拍板：含死亡）
#   LCGM k=4: class1=恶化(417) class4=危重/支持依赖(110, 含17死+80重插)
#   death5d=122, 其中105例在class2/3 → 必须显式并入
#   主终点 y_B = class∈{1,4} OR death5d   (n=632, ~19.5%)
#   对照   y_A = class∈{1,4}              (LCGM 原生, n=527)
#         y_C = class==1                  (纯恶化, 已有0.760, 统一管线复跑)
#   管线: 三道关卡(域SHAP→cov50%→LASSO+RF双筛60%→r>0.85) + 调优CatBoost
#   附加: 死亡信号分解 (存活者中AUROC / 死亡单预测)
# 输出: results/traj_adverse_results.json + traj_adverse_features_B.txt
# ============================================================
import sys, json, time, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results"); t0 = time.time()

# ---- 数据 ----
F = pd.read_parquet(PRJ/"abi_feature_matrix_MASTER_v4.parquet")
pp = pd.read_csv(PRJ/"traj_sofa_posteriors.csv")[["stay_id","class"]]
coh = pd.read_parquet(PRJ/"abi_full_cohort.parquet")[["stay_id","death5d","rein5d"]]
sub = pp.merge(coh, on="stay_id", how="left").set_index("stay_id")
F = F.loc[F.index.isin(sub.index)]
sub = sub.loc[F.index]
y_raw = sub.copy()
variants = {
    "B_adverse_with_death": (sub["class"].isin([1,4]) | (sub.death5d==1)).astype(int),
    "A_lcgm_native":        sub["class"].isin([1,4]).astype(int),
    "C_class1_only":        (sub["class"]==1).astype(int),
}
Xall = F.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
cov = Xall.notna().mean()
Xall = Xall[[c for c in Xall.columns if cov[c]>=0.5]]
print(f"[0] 矩阵 {Xall.shape} | 事件: " + " | ".join(f"{k}={v.sum()}({v.mean()*100:.1f}%)" for k,v in variants.items()), flush=True)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from catboost import CatBoostClassifier
tuned = json.load(open(PRJ/"dyn6h_tuned.json"))["best_params"]
Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xall), columns=Xall.columns, index=Xall.index)

def classify(feat):
    fl = feat.lower()
    if any(k in fl for k in ["223901","mt_","motor","gcs_m","traj_prob"]): return "neuro_motor"
    if any(k in fl for k in ["223900","220739","verbal","eye","gcs_t","gcs_total"]): return "neuro_other"
    if any(k in fl for k in ["sed_","prop_","mida_","fent_","dexm_","221906","norepi","epi_","vaso"]): return "sed_vaso"
    if any(k in fl for k in ["vt_","peep","fio2","vent_","220339","223835","rsbi","rr_ext"]): return "vent"
    if any(k in fl for k in ["sofa"]): return "sofa"
    if any(k in fl for k in ["age","male","sub_","anchor"]): return "static"
    if any(k in fl for k in ["hr_","rr_","spo2","map_","220045","220210","220277","temp","trd_","ent_"]): return "vitals_trend"
    if any(k in fl for k in ["508","509","510","512","513","lab_","creat","platelet","bilirubin","wbc","lactate","glucose","bun","rdw","hematocrit","anion"]): return "labs"
    return "other"

OUT = {}
for vname, y in variants.items():
    y = y.values.astype(int)
    gkf = GroupKFold(n_splits=5)
    groups = Xall.index.get_level_values("stay_id").values
    # ---- 关卡1: 域分类 + 每域SHAP top8 ----
    rf_q = RandomForestClassifier(n_estimators=200, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_q.fit(Ximp, y)
    shap_imp = pd.Series(rf_q.feature_importances_, index=Xall.columns)
    DOM = {}
    for c in Xall.columns: DOM.setdefault(classify(c), []).append(c)
    cand = []
    for d, cols in DOM.items():
        cand.extend(shap_imp[cols].sort_values(ascending=False).head(min(8,len(cols))).index.tolist())
    for must in ["anchor_age","male"]:
        if must in Xall.columns and must not in cand: cand.append(must)
    # ---- 关卡3: LASSO+RF 双筛 ----
    sel_l1 = np.zeros(len(cand)); sel_rf = np.zeros(len(cand))
    for tr, te in gkf.split(Ximp[cand], y, groups):
        sc = StandardScaler().fit(Ximp[cand].iloc[tr])
        lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=3000, class_weight='balanced')
        lasso.fit(sc.transform(Ximp[cand].iloc[tr]), y[tr]); sel_l1 += (lasso.coef_[0]!=0)
        rfb = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
        rfb.fit(Ximp[cand].iloc[tr], y[tr])
        imp = pd.Series(rfb.feature_importances_, index=cand); sel_rf += (imp > imp.median()).values
    freq = pd.DataFrame({"f":cand,"l1":sel_l1/5,"rf":sel_rf/5})
    C3 = freq[(freq.l1>=0.6)&(freq.rf>=0.6)].f.tolist()
    if len(C3) < 12: C3 = freq[(freq.l1>=0.4)|(freq.rf>=0.4)].f.tolist()
    # ---- 关卡4: r>0.85 冗余 ----
    corr = Ximp[C3].corr().abs()
    alive = list(shap_imp[C3].sort_values(ascending=False).index)
    for f in list(alive):
        if f not in alive: continue
        for g2 in alive:
            if g2==f: break
            if corr.loc[f,g2]>0.85: alive.remove(g2)
    feats = alive
    # ---- 最终: 调优CatBoost 5折 ----
    Xf = Ximp[feats]
    p = np.full(len(y), np.nan)
    for tr, te in gkf.split(Xf, y, groups):
        m = CatBoostClassifier(**tuned, random_seed=42, verbose=0, auto_class_weights='Balanced', thread_count=-1)
        m.fit(Xf.iloc[tr], y[tr]); p[te] = m.predict_proba(Xf.iloc[te])[:,1]
    auc = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    rng = np.random.default_rng(42)
    boots = [roc_auc_score(y[i],p[i]) for i in (rng.choice(len(y),len(y),replace=True) for _ in range(500))]
    ci = [round(float(np.percentile(boots,2.5)),4), round(float(np.percentile(boots,97.5)),4)]
    OUT[vname] = {"n_events":int(y.sum()), "event_rate":round(float(y.mean()),4),
        "n_features":len(feats), "auroc":round(float(auc),4), "ci95":ci,
        "auprc":round(float(ap),4), "brier":round(float(br),4), "features":feats}
    print(f"[{vname}] ev={y.sum()}({y.mean()*100:.1f}%) feats={len(feats)} AUROC={auc:.4f} [{ci[0]}-{ci[1]}] AUPRC={ap:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---- 死亡信号分解 (主终点 y_B) ----
yB = variants["B_adverse_with_death"].values
surv = (y_raw.death5d.values==0)
aucB = OUT["B_adverse_with_death"]["auroc"]
# 1) 存活者中 y_B 模型的判别力 (阳性=纯轨迹恶化)
pB = None
Xf = Ximp[OUT["B_adverse_with_death"]["features"]]
gkf = GroupKFold(5); groups = Xall.index.get_level_values("stay_id").values
pB = np.full(len(yB), np.nan)
for tr, te in gkf.split(Xf, yB, groups):
    m = CatBoostClassifier(**tuned, random_seed=42, verbose=0, auto_class_weights='Balanced', thread_count=-1)
    m.fit(Xf.iloc[tr], yB[tr]); pB[te] = m.predict_proba(Xf.iloc[te])[:,1]
auc_surv = roc_auc_score(yB[surv], pB[surv])
# 2) 死亡单预测
dth = (y_raw.death5d.values==1).astype(int)
pdth = np.full(len(dth), np.nan)
for tr, te in gkf.split(Xf, dth, groups):
    m = CatBoostClassifier(**tuned, random_seed=42, verbose=0, auto_class_weights='Balanced', thread_count=-1)
    m.fit(Xf.iloc[tr], dth[tr]); pdth[te] = m.predict_proba(Xf.iloc[te])[:,1]
auc_dth = roc_auc_score(dth, pdth)
OUT["decomposition"] = {"auroc_B_full":aucB, "auroc_B_survivors_only":round(float(auc_surv),4),
    "auroc_death_alone_same_features":round(float(auc_dth),4),
    "n_survivors":int(surv.sum()), "n_survivor_events":int(yB[surv].sum())}
print(f"\n[分解] y_B 全体={aucB:.4f} | 存活者中={auc_surv:.4f} (n={surv.sum()}, ev={yB[surv].sum()}) | 死亡单预测={auc_dth:.4f}", flush=True)

OUT["decision"] = "PI 2026-10-03: 主终点=拔管后5天不良轨迹(含死亡) y_B; LCGM k=4 class1+4 ∪ death5d"
OUT["class_composition"] = {"class1_deteriorating":417,"class2_recovering":520,"class3_stable":2185,
    "class4_severe":110,"death5d":122,"death_in_class14":17,"death_in_class23":105}
json.dump(OUT, open(PRJ/"traj_adverse_results.json","w"), ensure_ascii=False, indent=1)
(PRJ/"traj_adverse_features_B.txt").write_text("\n".join(OUT["B_adverse_with_death"]["features"]), encoding="utf-8")
print(f"DONE ({time.time()-t0:.0f}s)")
