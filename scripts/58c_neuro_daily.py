# -*- coding: utf-8 -*-
# 58c — 拔管后每日 motor-GCS 提取 (神经恶化轴: day0-1 基线 vs day2-5 最低, 降>=2)
import sys, time, warnings, numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings('ignore')
from pathlib import Path
PRJ = Path(r"E:/TBI subtype/08_extubation_prediction/results")
M4 = "E:/TBI subtype/data/mimic-iv-3.1"
con = duckdb.connect(); con.execute("PRAGMA threads=4"); t0 = time.time()
OUT = PRJ/"_neuro_daily.parquet"
if OUT.exists():
    print("缓存命中"); sys.exit(0)
coh = pd.read_parquet(PRJ/"abi_full_cohort.parquet")[["stay_id","t_ext"]]
coh["t_ext"] = pd.to_datetime(coh.t_ext)
coh[["stay_id","t_ext"]].to_parquet(PRJ/"_neuro_tmp.parquet", index=False)
R = (PRJ/"_neuro_tmp.parquet").as_posix()
ce = con.execute(f"""
WITH r AS (SELECT stay_id, CAST(t_ext AS TIMESTAMP) t_ext FROM read_parquet('{R}'))
SELECT c.stay_id,
       floor((epoch(CAST(c.charttime AS TIMESTAMP)) - epoch(r.t_ext))/86400.0) AS daybin,
       CAST(c."value" AS VARCHAR) v
FROM read_csv_auto('{M4}/icu/chartevents.csv.gz') c JOIN r ON c.stay_id=r.stay_id
WHERE c.itemid = 223901
  AND epoch(CAST(c.charttime AS TIMESTAMP)) > epoch(r.t_ext)
  AND epoch(CAST(c.charttime AS TIMESTAMP)) <= epoch(r.t_ext) + 5*86400
""").df()
MOTOR={"Obeys Commands":6,"Localizes Pain":5,"Flex-withdraws":4,"Abnormal Flexion":3,"Abnormal extension":2,"No response":1,"No Response":1}
ce["m"] = ce.v.str.strip().map(MOTOR)
ce = ce.dropna(subset=["m"])
ce.to_parquet(OUT, index=False)
print(f"motor-GCS 行: {len(ce)} / {ce.stay_id.nunique()} 患者 ({time.time()-t0:.0f}s)")
