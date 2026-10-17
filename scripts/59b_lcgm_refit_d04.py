# -*- coding: utf-8 -*-
# 59b — 修正版 LCGM: D0-D4 (剔除边界伪影), 二次混合项, k=2-6
#    目的: 检验真实"爬升类"能否被潜类别模型分离
import sys, subprocess, time, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results")
pp = pd.read_csv(PRJ/"traj_clean_posteriors.csv")
daily = pd.read_parquet(PRJ/"_sofa_post_daily.parquet")
daily = daily[daily.stay_id.isin(pp.stay_id) & (daily.day<=4)]
daily[["stay_id","day","sofa"]].to_csv(PRJ/"traj_d04_long.csv", index=False)
r = """suppressMessages(library(lcmm))
d <- read.csv("traj_d04_long.csv"); d$day <- as.numeric(d$day); d$sofa <- as.numeric(d$sofa)
d <- d[order(d$stay_id, d$day), ]
cat("N:", length(unique(d$stay_id)), "\\n")
m1 <- hlme(sofa ~ day, subject="stay_id", data=d, verbose=FALSE)
for (k in 2:6) {
  m <- tryCatch(hlme(sofa ~ day, subject="stay_id", ng=k, data=d,
      mixture=~day+I(day^2), nwg=TRUE, B=m1, maxiter=250, verbose=FALSE), error=function(e) NULL)
  if (is.null(m)) { cat("k=",k," FAILED\\n",sep=""); next }
  pp <- m$pprob
  cat("k=", k, " BIC=", round(m$BIC,0), "\\n", sep="")
  for (cl in 1:k) {
    ids <- pp[pp$class==cl,"stay_id"]; dd <- d[d$stay_id %in% ids,]
    tr <- tapply(dd$sofa, dd$day, mean)
    # 模型估计的类轨迹也输出
    cat("  class", cl, " n=", length(ids), ": ", paste(sprintf("%.1f", tr), collapse="->"), "\\n", sep="")
  }
  if (k==4 || k==5 || k==6) saveRDS(m, paste0("traj_d04_k", k, ".rds"))
}
cat("Done\\n")
"""
(PRJ/"59b_lcgm.R").write_text(r, encoding="utf-8")
print("R 启动...", flush=True)
rr = subprocess.run(["C:/Program Files/R/R-4.5.3/bin/Rscript.exe", str(PRJ/"59b_lcgm.R")],
                    capture_output=True, text=True, timeout=1500, cwd=str(PRJ))
print(rr.stdout[-3000:] if len(rr.stdout)>3000 else rr.stdout)
if rr.returncode != 0: print("STDERR:", rr.stderr[-600:])
