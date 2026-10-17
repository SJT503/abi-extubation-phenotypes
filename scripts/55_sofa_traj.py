# -*- coding: utf-8 -*-
# 55 — 拔管后 SOFA 轨迹: 提取 + 日分箱 + LCGM
import sys, time, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
import duckdb, subprocess
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results")
M4 = "E:/TBI subtype/data/mimic-iv-3.1"
con = duckdb.connect(); con.execute("PRAGMA threads=4")
t0 = time.time()

risk = pd.read_parquet(PRJ/"abi_deterioration_riskset_v2.parquet")[["stay_id","deterioration"]]
coh = pd.read_parquet(PRJ/"abi_full_cohort.parquet")[["stay_id","t_ext"]]
risk = risk.merge(coh, on="stay_id", how="left")
risk["t_ext"] = pd.to_datetime(risk.t_ext)
risk[["stay_id","t_ext"]].to_parquet(PRJ/"_sofa_post_tmp.parquet", index=False)
RISK = (PRJ/"_sofa_post_tmp.parquet").as_posix()

print("[A] 提取拔管后数据...", flush=True)
ce = con.execute(f"""
WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{RISK}'))
SELECT c.stay_id, c.itemid,
       floor((epoch(CAST(c.charttime AS TIMESTAMP)) - epoch(r.t_ext)) / 86400.0) AS daybin,
       TRY_CAST(c.valuenum AS DOUBLE) vn, CAST(c."value" AS VARCHAR) v
FROM read_csv_auto('{M4}/icu/chartevents.csv.gz', types={{'valuenum':'VARCHAR'}}) c
JOIN r ON c.stay_id = r.stay_id
WHERE c.itemid IN (223901,220739,223900,220052,225312,220181,223835,223831)
  AND epoch(CAST(c.charttime AS TIMESTAMP)) > epoch(r.t_ext) - 86400
  AND epoch(CAST(c.charttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
""").df()
print(f"  ce: {len(ce)} ({time.time()-t0:.0f}s)", flush=True)

le = con.execute(f"""
WITH rr AS (SELECT s.stay_id, s.hadm_id, CAST(r.t_ext AS TIMESTAMP) t_ext FROM read_parquet('{RISK}') r
            JOIN read_csv_auto('{M4}/icu/icustays.csv.gz') s ON r.stay_id=s.stay_id)
SELECT rr.stay_id, l.itemid,
       floor((epoch(CAST(l.charttime AS TIMESTAMP)) - epoch(rr.t_ext)) / 86400.0) AS daybin,
       TRY_CAST(l.valuenum AS DOUBLE) vn
FROM read_csv_auto('{M4}/hosp/labevents.csv.gz', types={{'valuenum':'VARCHAR'}}) l
JOIN rr ON l.hadm_id = rr.hadm_id
WHERE l.itemid IN (51265,50885,50912,50821,50816)
  AND epoch(CAST(l.charttime AS TIMESTAMP)) > epoch(rr.t_ext) - 86400
  AND epoch(CAST(l.charttime AS TIMESTAMP)) <= epoch(rr.t_ext) + 5*86400
""").df()
print(f"  le: {len(le)} ({time.time()-t0:.0f}s)", flush=True)

VASO = (221906,221289,221662,221786,221749)
ie = con.execute(f"""
WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{RISK}'))
SELECT i.stay_id,
       floor((epoch(CAST(i.starttime AS TIMESTAMP)) - epoch(r.t_ext)) / 86400.0) AS daybin
FROM read_csv_auto('{M4}/icu/inputevents.csv.gz') i
JOIN r ON i.stay_id = r.stay_id
WHERE i.itemid IN ({','.join(map(str,VASO))})
  AND epoch(CAST(i.starttime AS TIMESTAMP)) > epoch(r.t_ext)
  AND epoch(CAST(i.starttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
""").df()

uo = con.execute(f"""
WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{RISK}'))
SELECT o.stay_id,
       floor((epoch(CAST(o.charttime AS TIMESTAMP)) - epoch(r.t_ext)) / 86400.0) AS daybin,
       TRY_CAST(o."value" AS DOUBLE) val
FROM read_csv_auto('{M4}/icu/outputevents.csv.gz', types={{'value':'VARCHAR'}}) o
JOIN r ON o.stay_id = r.stay_id
WHERE epoch(CAST(o.charttime AS TIMESTAMP)) > epoch(r.t_ext)
  AND epoch(CAST(o.charttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
""").df()
print(f"  ie={len(ie)} uo={len(uo)} ({time.time()-t0:.0f}s)", flush=True)

# SOFA 计算
MOTOR={"Obeys Commands":6,"Localizes Pain":5,"Flex-withdraws":4,"Abnormal Flexion":3,"Abnormal extension":2,"No response":1,"No Response":1}
EYE={"Spontaneously":4,"To Speech":3,"To Pain":2,"None":1}
VERB={"Oriented":5,"Confused":4,"Inappropriate Words":3,"Incomprehensible sounds":2,"No Response":1,"No response":1}
ce["ord"]=np.nan
for iid,mp in [(223901,MOTOR),(220739,EYE)]:
    m=ce.itemid==iid; ce.loc[m,"ord"]=ce.loc[m,"v"].str.strip().map(mp)
m=ce.itemid==223900
vt=ce.loc[m,"v"].str.strip().map(VERB)
ce.loc[m,"ord"]=ce.loc[m,"vn"].fillna(vt)
m=ce.itemid==223835
ce.loc[m,"vn"]=np.where(ce.loc[m,"vn"]>1.5, ce.loc[m,"vn"]/100, ce.loc[m,"vn"])

def s_resp(p,f,mech=True):
    if pd.isna(p) or pd.isna(f) or f<=0: return 0
    r=p/f
    return 4 if(r<100 and mech) else 3 if(r<200 and mech) else 2 if r<300 else 1 if r<400 else 0
def s_coag(x): return 0 if pd.isna(x) else 4 if x<20 else 3 if x<50 else 2 if x<100 else 1 if x<150 else 0
def s_hep(x): return 0 if pd.isna(x) else 4 if x>=12 else 3 if x>=6 else 2 if x>=2 else 1 if x>=1.2 else 0
def s_cns(g): return 0 if pd.isna(g) else 4 if g<6 else 3 if g<10 else 2 if g<13 else 1 if g<15 else 0
def s_renal(cr,uo_):
    s=0
    if not pd.isna(cr): s=max(s,4 if cr>=5 else 3 if cr>=3.5 else 2 if cr>=2 else 1 if cr>=1.2 else 0)
    if not pd.isna(uo_): s=max(s,4 if uo_<200 else 3 if uo_<500 else 0)
    return s

vaso_days = set(map(tuple, ie[["stay_id","daybin"]].drop_duplicates().values))
uo_g = {k: g for k, g in uo.groupby("stay_id")}
rows=[]
ce_groups = {k: g for k, g in ce.groupby("stay_id")}
le_groups = {k: g for k, g in le.groupby("stay_id")}
for sid in set(ce_groups.keys()) | set(le_groups.keys()):
    g_ce = ce_groups.get(sid, pd.DataFrame())
    g_le = le_groups.get(sid, pd.DataFrame())
    g_uo = uo_g.get(sid)
    for d in range(0,6):
        if g_ce is not None and len(g_ce): c_ = g_ce[g_ce.daybin==d]
        else: c_ = pd.DataFrame({"itemid":[],"vn":[],"ord":[]})
        if g_le is not None and len(g_le): l_ = g_le[g_le.daybin==d]
        else: l_ = pd.DataFrame({"itemid":[],"vn":[]})
        if len(c_)==0 and len(l_)==0: continue
        pao2 = min([x for x in list(c_[c_.itemid==223831].vn)+list(l_[l_.itemid==50821].vn) if not pd.isna(x)] or [np.nan])
        fio2 = max([x for x in list(c_[c_.itemid==223835].vn)+list(l_[l_.itemid==50816].vn) if not pd.isna(x)] or [0.21])
        plt=l_[l_.itemid==51265].vn.min() if len(l_) else np.nan
        bili=l_[l_.itemid==50885].vn.max() if len(l_) else np.nan
        cr=l_[l_.itemid==50912].vn.max() if len(l_) else np.nan
        mv=c_[c_.itemid.isin([220052,225312,220181])].vn.min() if len(c_) else np.nan
        gm=c_[c_.itemid==223901].ord.min() if len(c_) else np.nan
        ge=c_[c_.itemid==220739].ord.min() if len(c_) else np.nan
        gv=c_[c_.itemid==223900].ord.min() if len(c_) else np.nan
        vals=[x for x in [gm,ge,gv] if not pd.isna(x)]
        gcs=sum(vals) if vals else np.nan
        uo_ = g_uo[g_uo.daybin==d].val.sum() if g_uo is not None else np.nan
        if pd.isna(uo_): uo_=np.nan
        cv=3 if (sid,d) in vaso_days else (1 if(not pd.isna(mv) and mv<70) else 0)
        tot=s_resp(pao2,fio2)+s_coag(plt)+s_hep(bili)+cv+s_cns(gcs)+s_renal(cr,uo_)
        rows.append({"stay_id":sid,"day":d,"sofa":tot,"gcs":gcs})

SOFA_D = pd.DataFrame(rows)
SOFA_D["day"]=SOFA_D["day"].astype(int)
SOFA_D.to_parquet(PRJ/"_sofa_post_daily.parquet", index=False)
print(f"[B] 拔管后日SOFA: {len(SOFA_D)} 行 / {SOFA_D.stay_id.nunique()} 患者 ({time.time()-t0:.0f}s)", flush=True)

# 长表 + R lcmm
SOFA_D[["stay_id","day","sofa"]].to_csv(PRJ/"traj_sofa_post_long.csv", index=False)

r_script = """suppressMessages(library(lcmm))
d <- read.csv("traj_sofa_post_long.csv")
d$day <- as.numeric(d$day); d$sofa <- as.numeric(d$sofa)
d <- d[order(d$stay_id, d$day), ]
cat("N:", length(unique(d$stay_id)), "obs:", nrow(d), "\\n")
m1 <- hlme(sofa ~ day, subject="stay_id", data=d, verbose=FALSE)
fits <- list(); bics <- c()
for (k in 2:4) {
  m <- tryCatch(hlme(sofa ~ day, subject="stay_id", ng=k, data=d,
      mixture=~day, nwg=TRUE, B=m1, maxiter=150, verbose=FALSE),
    error=function(e) NULL)
  if (!is.null(m)) {
    fits[[as.character(k)]] <- m; bics <- c(bics, m$BIC)
    pp <- m$pprob[, paste0("prob", 1:k)]
    cat("k=", k, " BIC=", round(m$BIC,0), " APP=", round(mean(apply(pp,1,max)),3), "\\n")
  } else bics <- c(bics, NA)
}
valid <- which(!is.na(bics))
best_k <- valid[which.min(bics[valid])] + 1
cat("Best k =", best_k, "\\n")
m_best <- fits[[as.character(best_k)]]
saveRDS(m_best, "traj_sofa_model.rds")
pp <- as.data.frame(m_best$pprob)
write.csv(pp, "traj_sofa_posteriors.csv", row.names=FALSE)
cat("Done\\n")
"""
(PRJ/"55_lcmm_sofa.R").write_text(r_script, encoding="utf-8")
print("[C] R lcmm...", flush=True)
r = subprocess.run(["C:/Program Files/R/R-4.5.3/bin/Rscript.exe", str(PRJ/"55_lcmm_sofa.R")],
                   capture_output=True, text=True, timeout=600, cwd=str(PRJ))
print(r.stdout[-800:] if len(r.stdout)>800 else r.stdout)
print(f"DONE ({time.time()-t0:.0f}s)")
