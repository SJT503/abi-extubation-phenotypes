# -*- coding: utf-8 -*-
# ============================================================
# 58 — 终局终点: 干净队列的拔管后恶化轨迹表型预测
#   PI 2026-10-04 拍板: 主终点=拔管后轨迹表型; 剔除临终关怀/comfort 拔管
#   排除集 = ① WLST 裁定 EXCLUDE (79例, 03c号裁定含DCD器官捐献)
#          ∪ ② 无支持死亡代理: death5d 且 拔后无重插/新升压/新RRT/新NIV
#             (= 死亡时无任何器官支持升级 → 临终轨迹; ENIO "end-of-life
#                extubation" 入组排除的回顾性对应物)
#   UNDET(11) 保留 (无证据不排除); RETAIN(3) 保留
#   然后在干净队列上重拟合 LCGM (k=2/3/4, Zhang k=3 参照) → 恶化类 → 建模
# 输出: clean_cohort_trajectory.json / traj_clean_posteriors.csv / daily 长表缓存
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
CACHE = PRJ/"_v3_support_flags.parquet"

# ---- [A] 拔后支持升级旗标 (缓存) ----
if CACHE.exists():
    sf = pd.read_parquet(CACHE)
    print(f"[A] 支持旗标缓存命中 ({len(sf)})", flush=True)
else:
    coh[["stay_id","t_ext"]].to_parquet(PRJ/"_v3_tmp.parquet", index=False)
    R = (PRJ/"_v3_tmp.parquet").as_posix()
    print("[A] 提取拔管前后 NIV/升压/RRT...", flush=True)
    HFNC = "high flow|bipap|bi-pap|cpap|non-rebreath"
    ce = con.execute(f"""
    WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{R}'))
    SELECT c.stay_id, c.itemid,
           floor((epoch(CAST(c.charttime AS TIMESTAMP)) - epoch(r.t_ext))/86400.0) AS daybin,
           CAST(c."value" AS VARCHAR) v
    FROM read_csv_auto('{M4}/icu/chartevents.csv.gz') c JOIN r ON c.stay_id=r.stay_id
    WHERE c.itemid IN (226732,227577,227578,227579,227580,227581,227582,227583)
      AND epoch(CAST(c.charttime AS TIMESTAMP)) > epoch(r.t_ext) - 86400
      AND epoch(CAST(c.charttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
    """).df()
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
    # 修正: pre 窗只认 高强度氧疗(HFNC/NRB) 或 BiPap设置 或 NIV procedure —— 任意 O2 设备行不算
    dev = ce[ce.itemid==226732]
    dev_hi = dev[dev.v.str.contains(HFNC, case=False, na=False)]
    bip = ce[ce.itemid.isin(range(227577,227584))]
    niv_pre  = set(bip[bip.daybin<0].stay_id) | set(dev_hi[dev_hi.daybin<0].stay_id) | set(pe[(pe.itemid==225794)&(pe.daybin<0)].stay_id)
    niv_post = set(bip[bip.daybin>=0].stay_id) | set(dev_hi[dev_hi.daybin>=0].stay_id) | set(pe[(pe.itemid==225794)&(pe.daybin>=0)].stay_id)
    # 原始行缓存 (后续微调免重扫)
    ce.to_parquet(PRJ/"_v3_ce_raw.parquet", index=False)
    pe.to_parquet(PRJ/"_v3_pe_raw.parquet", index=False)
    ie.to_parquet(PRJ/"_v3_ie_raw.parquet", index=False)
    vp_pre = set(ie[ie.daybin<0].stay_id); vp_post = set(ie[ie.daybin>=0].stay_id)
    rt_pre = set(pe[(pe.itemid!=225794)&(pe.daybin<0)].stay_id); rt_post = set(pe[(pe.itemid!=225794)&(pe.daybin>=0)].stay_id)
    sf = pd.DataFrame({"stay_id": coh.stay_id})
    sf["niv_new"]  = [(s in niv_post) and (s not in niv_pre) for s in sf.stay_id]
    sf["vaso_new"] = [(s in vp_post) and (s not in vp_pre) for s in sf.stay_id]
    sf["rrt_new"]  = [(s in rt_post) and (s not in rt_pre) for s in sf.stay_id]
    sf.to_parquet(CACHE, index=False)
    print(f"[A] 提取完成 niv={int(sf.niv_new.sum())} vaso={int(sf.vaso_new.sum())} rrt={int(sf.rrt_new.sum())} ({time.time()-t0:.0f}s)", flush=True)

coh = coh.merge(sf, on="stay_id", how="left")

# ---- [B] comfort 排除集 ----
adj = pd.read_csv(PRJ/"wlst_adjudication_AI_verdicts.csv")
excl_adj = set(adj[adj.verdict=="EXCLUDE"].stay_id) & set(coh.stay_id)
unsupported_death = (coh.death5d==1) & ~(coh.rein5d.astype(bool)) & ~coh.niv_new & ~coh.vaso_new & ~coh.rrt_new
excl_proxy = set(coh[unsupported_death].stay_id)
excl_all = excl_adj | excl_proxy
print(f"[B] 排除集: 裁定EXCLUDE∩队列={len(excl_adj)} | 无支持死亡={len(excl_proxy)} | 并集={len(excl_all)}", flush=True)
clean = coh[~coh.stay_id.isin(excl_all)].copy()
print(f"    干净队列: {len(clean)} (原 {len(coh)}, 剔 {len(excl_all)}) | 死亡残留={int(clean.death5d.sum())}", flush=True)

# ---- [C] 干净队列 LCGM 重拟合 ----
daily = pd.read_parquet(PRJ/"_sofa_post_daily.parquet")
daily = daily[daily.stay_id.isin(clean.stay_id)]
long_csv = daily[["stay_id","day","sofa"]].rename(columns={"day":"day"})
long_csv.to_csv(PRJ/"traj_clean_long.csv", index=False)
r_code = """suppressMessages(library(lcmm))
d <- read.csv("traj_clean_long.csv"); d$day <- as.numeric(d$day); d$sofa <- as.numeric(d$sofa)
d <- d[order(d$stay_id, d$day), ]
cat("N:", length(unique(d$stay_id)), "obs:", nrow(d), "\\n")
m1 <- hlme(sofa ~ day, subject="stay_id", data=d, verbose=FALSE)
best <- NULL; best_bic <- Inf; best_k <- NA
for (k in 2:4) {
  m <- tryCatch(hlme(sofa ~ day, subject="stay_id", ng=k, data=d, mixture=~day, nwg=TRUE, B=m1, maxiter=200, verbose=FALSE), error=function(e) NULL)
  if (is.null(m)) next
  cat("k=", k, " BIC=", round(m$BIC,0), "\\n", sep="")
  pp <- m$pprob
  for (cl in 1:k) {
    ids <- pp[pp$class==cl,"stay_id"]; dd <- d[d$stay_id %in% ids,]
    tr <- tapply(dd$sofa, dd$day, mean)
    cat("  class", cl, " n=", length(ids), ": ", paste(sprintf("%.1f", tr), collapse="->"), "\\n", sep="")
  }
  if (m$BIC < best_bic) { best_bic <- m$BIC; best_k <- k; best <- m }
}
cat("BEST_K=", best_k, "\\n", sep="")
pp <- as.data.frame(best$pprob)
write.csv(pp, "traj_clean_posteriors.csv", row.names=FALSE)
cat("Done\\n")
"""
(PRJ/"58_lcmm_clean.R").write_text(r_code, encoding="utf-8")
print("[C] LCGM 重拟合 (k=2-4)...", flush=True)
rr = subprocess.run(["C:/Program Files/R/R-4.5.3/bin/Rscript.exe", str(PRJ/"58_lcmm_clean.R")],
                    capture_output=True, text=True, timeout=1200, cwd=str(PRJ))
print(rr.stdout[-2000:] if len(rr.stdout)>2000 else rr.stdout)
if rr.returncode != 0: print("STDERR:", rr.stderr[-800:]); sys.exit(1)

clean.to_parquet(PRJ/"clean_cohort_v1.parquet", index=False)
print(f"DONE-Stage1 ({time.time()-t0:.0f}s) — 请核对恶化类编号后跑 58b")
