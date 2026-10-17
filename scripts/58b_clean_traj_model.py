# -*- coding: utf-8 -*-
# ============================================================
# 58b — 干净队列终局建模: 双终点 + 恶化类表型构成 + Zhang 对照
#   y1 (主) = 恶化轨迹类 (LCGM, 纯生理表型)
#   y2 (敏感性, Zhang 对齐) = 恶化类 ∪ 重插 ∪ 新NIV ∪ 新升压 ∪ 新RRT ∪ 神经恶化 ∪ 残留死亡
#   恶化类自动识别: 斜率最大 且 起始SOFA低于队列中位
#   管线: 三道关卡 + 调优CatBoost 5折 GroupKFold
# 输出: clean_traj_final.json
# ============================================================
import sys, json, time, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results"); t0 = time.time()

clean = pd.read_parquet(PRJ/"clean_cohort_v1.parquet")
pp = pd.read_csv(PRJ/"traj_clean_posteriors.csv")
daily = pd.read_parquet(PRJ/"_sofa_post_daily.parquet")
daily = daily[daily.stay_id.isin(clean.stay_id)]
base_med = daily[daily.day<=0].groupby("stay_id").sofa.mean().median()

info = {}
for cl, g in pp.groupby("class"):
    ids = g.stay_id
    tr = daily[daily.stay_id.isin(ids)].groupby("day").sofa.mean()
    slope = tr.get(max(tr.index), np.nan) - tr.get(0, np.nan)
    info[int(cl)] = {"n": int(len(ids)), "start": round(float(tr.get(0,np.nan)),1),
                     "end": round(float(tr.get(max(tr.index),np.nan)),1), "slope": round(float(slope),2)}
    print(f"  class{cl}: n={len(ids)} {tr.get(0,np.nan):.1f}->{tr.get(max(tr.index),np.nan):.1f} (slope {slope:+.2f})")
cands = [cl for cl, v in info.items() if v["slope"] > 0 and v["start"] < base_med]
if not cands: cands = [cl for cl, v in info.items() if v["slope"] > 0]
DET = max(cands, key=lambda c: info[c]["slope"])
print(f"[0] 恶化类 = class{DET} ({info[DET]}) | 基线中位SOFA={base_med:.1f}", flush=True)

yi = pp.set_index("stay_id")["class"].eq(DET).astype(int)
yi = yi.loc[yi.index.isin(clean.stay_id)]
CL = clean.set_index("stay_id").loc[yi.index]

# 神经恶化 (motor-GCS)
neuro = None
NEURO_PATH = PRJ/"_neuro_daily.parquet"
if NEURO_PATH.exists():
    mg = pd.read_parquet(NEURO_PATH)
    MOTOR={"Obeys Commands":6,"Localizes Pain":5,"Flex-withdraws":4,"Abnormal Flexion":3,"Abnormal extension":2,"No response":1,"No Response":1}
    nd = {}
    for sid, g in mg.groupby("stay_id"):
        b = g[(g.daybin>=0)&(g.daybin<=1)].m.max()
        lo = g[g.daybin>=2].m.min()
        if pd.notna(b) and pd.notna(lo): nd[sid] = (b - lo) >= 2
    neuro = pd.Series(nd)
    print(f"[0b] 神经恶化(motor-GCS降>=2): {int(neuro.sum())}/{len(neuro)}", flush=True)

y2 = (yi.astype(bool) | CL.rein5d.astype(bool) | CL.niv_new.astype(bool)
      | CL.vaso_new.astype(bool) | CL.rrt_new.astype(bool) | CL.death5d.astype(bool)).astype(int)
if neuro is not None:
    neuro_al = neuro.reindex(yi.index).fillna(False).astype(bool)
    y2 = (y2.astype(bool) | neuro_al).astype(int)
print(f"[0c] y1={yi.sum()}({yi.mean()*100:.1f}%) | y2={y2.sum()}({y2.mean()*100:.1f}%)", flush=True)

# ===== 建模管线 =====
F = pd.read_parquet(PRJ/"abi_feature_matrix_MASTER_v4.parquet").loc[yi.index]
X = F.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
cov = X.notna().mean(); X = X[[c for c in X.columns if cov[c]>=0.5]]
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from catboost import CatBoostClassifier
tuned = json.load(open(PRJ/"dyn6h_tuned.json"))["best_params"]
Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
groups = X.index.get_level_values("stay_id").values
gkf = GroupKFold(n_splits=5)

def classify(f):
    fl = f.lower()
    if any(k in fl for k in ["223901","mt_","motor","gcs_m"]): return "neuro_motor"
    if any(k in fl for k in ["223900","220739","verbal","eye","gcs"]): return "neuro_other"
    if any(k in fl for k in ["sed_","prop_","mida_","fent_","dexm_","vaso","norepi"]): return "sed_vaso"
    if any(k in fl for k in ["vt_","peep","fio2","vent_","rsbi","rr_ext"]): return "vent"
    if "sofa" in fl: return "sofa"
    if any(k in fl for k in ["age","male","sub_","anchor"]): return "static"
    if any(k in fl for k in ["hr_","rr_","spo2","map_","temp","trd_","ent_"]): return "vitals_trend"
    if any(k in fl for k in ["508","509","510","512","513","lab_","creat","platelet","bili","wbc","lactate","glu","bun","rdw","hemat","anion"]): return "labs"
    return "other"

def run_ep(yv):
    rf_q = RandomForestClassifier(n_estimators=200, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_q.fit(Ximp, yv)
    shap_imp = pd.Series(rf_q.feature_importances_, index=Ximp.columns)
    DOM = {}
    for c in Ximp.columns: DOM.setdefault(classify(c), []).append(c)
    cand = []
    for d, cols in DOM.items():
        cand.extend(shap_imp[cols].sort_values(ascending=False).head(min(8,len(cols))).index.tolist())
    for must in ["anchor_age","male"]:
        if must in Ximp.columns and must not in cand: cand.append(must)
    sel_l1 = np.zeros(len(cand)); sel_rf = np.zeros(len(cand))
    for tr, te in gkf.split(Ximp[cand], yv, groups):
        sc = StandardScaler().fit(Ximp[cand].iloc[tr])
        la = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=3000, class_weight='balanced')
        la.fit(sc.transform(Ximp[cand].iloc[tr]), yv[tr]); sel_l1 += (la.coef_[0]!=0)
        rfb = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight='balanced', random_state=42, n_jobs=-1)
        rfb.fit(Ximp[cand].iloc[tr], yv[tr])
        imp = pd.Series(rfb.feature_importances_, index=cand); sel_rf += (imp > imp.median()).values
    freq = pd.DataFrame({"f":cand,"l1":sel_l1/5,"rf":sel_rf/5})
    C3 = freq[(freq.l1>=0.6)&(freq.rf>=0.6)].f.tolist()
    if len(C3) < 12: C3 = freq[(freq.l1>=0.4)|(freq.rf>=0.4)].f.tolist()
    corr = Ximp[C3].corr().abs()
    alive = list(shap_imp[C3].sort_values(ascending=False).index)
    for f in list(alive):
        if f not in alive: continue
        for g2 in alive:
            if g2==f: break
            if corr.loc[f,g2]>0.85: alive.remove(g2)
    feats = alive
    Xf = Ximp[feats]
    p = np.full(len(yv), np.nan)
    for tr, te in gkf.split(Xf, yv, groups):
        m = CatBoostClassifier(**tuned, random_seed=42, verbose=0, auto_class_weights='Balanced', thread_count=-1)
        m.fit(Xf.iloc[tr], yv[tr]); p[te] = m.predict_proba(Xf.iloc[te])[:,1]
    auc = roc_auc_score(yv, p); ap = average_precision_score(yv, p); br = brier_score_loss(yv, p)
    rng = np.random.default_rng(42)
    boots = [roc_auc_score(yv[i],p[i]) for i in (rng.choice(len(yv),len(yv),replace=True) for _ in range(500))]
    ci = [round(float(np.percentile(boots,2.5)),4), round(float(np.percentile(boots,97.5)),4)]
    return {"n_features":len(feats),"auroc":round(float(auc),4),"ci95":ci,
            "auprc":round(float(ap),4),"brier":round(float(br),4),"features":feats}, p

OUT = {"endpoint":"clean-cohort deterioration trajectory (palliative extubations excluded)",
  "cohort_n": int(len(yi)), "class_info": info, "deteriorating_class": int(DET)}
for name, yv in [("y1_trajectory", yi.values.astype(int)), ("y2_expanded", y2.values.astype(int))]:
    res, p = run_ep(yv)
    OUT[name] = {"events": int(yv.sum()), "event_rate": round(float(yv.mean()),4), **res}
    print(f"[1:{name}] ev={yv.sum()}({yv.mean()*100:.1f}%) AUROC={res['auroc']} [{res['ci95'][0]}-{res['ci95'][1]}] AUPRC={res['auprc']} feats={res['n_features']}", flush=True)

# 恶化类表型构成 + 分亚型 (y1)
det_ids = yi[yi==1].index
dsub = CL.loc[det_ids]
n = len(dsub)
OUT["phenotype_composition"] = {"n": int(n),
  "reintubation": int(dsub.rein5d.sum()), "new_vaso": int(dsub.vaso_new.sum()),
  "new_rrt": int(dsub.rrt_new.sum()), "new_niv": int(dsub.niv_new.sum()), "death5d": int(dsub.death5d.sum())}
sub_map = pd.read_parquet(PRJ/"abi_full_cohort.parquet").set_index("stay_id").subtype
perl = {}
for st in sub_map.unique():
    m_ = (sub_map.loc[yi.index].values==st)
    if m_.sum()>150: perl[st] = {"n": int(m_.sum()), "ev": int(yi.values[m_].sum())}
OUT["by_subtype"] = perl
OUT["zhang_reference"] = {"deterioration_class_pct_dev": 22.1, "mimic_iii": 23.2,
  "note": "Zhang npj DM 2026 sepsis GBTM k=3; 恶化类含支持升级与死亡; 我方 y1 纯轨迹更严, y2 为 Zhang 对齐版"}
json.dump(OUT, open(PRJ/"clean_traj_final.json","w"), ensure_ascii=False, indent=1)
(PRJ/"clean_traj_features.txt").write_text("\n".join(OUT["y1_trajectory"]["features"]), encoding="utf-8")
print(f"DONE ({time.time()-t0:.0f}s)")
