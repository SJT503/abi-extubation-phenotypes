# abi-extubation-phenotypes

Analysis code for:

> **Post-extubation organ failure trajectory phenotypes in acute brain injury: a multi-database cohort study**
> Jiangtao Sheng, Xueer Liu, Runmin Lin, Qiaochu Sun, Xiaoxuan Chen, Kangsheng Li, Weiqiang Chen

## What this code does

Latent class growth modelling (LCGM) of daily SOFA scores from day 0 to day 4 after extubation, in adults with acute brain injury (traumatic brain injury, intracerebral haemorrhage, subarachnoid haemorrhage, ischaemic stroke), in three databases:

| Database | Role | Analysed |
|---|---|---|
| MIMIC-IV v3.1 | discovery | 3,150 patients |
| MIMIC-III v1.4 | temporal replication (same institution) | 754 patients |
| eICU-CRD v2.0 | external replication (multi-centre) | 635 patients |

Models are fitted independently in each database. Outcomes are never used to define classes. Six latent classes map to four phenotypes: Stable, Recovery, Rising deterioration, Persistent severe.

## Repository layout

```
scripts/        Python: cohort assembly, SOFA extraction, comfort-care exclusion, endpoint construction, figure generation
r_scripts/      R: LCGM fitting (lcmm) and bootstrap stability
```

### Pipeline order

1. `scripts/55_sofa_traj.py` — extract post-extubation daily SOFA (day 0–4), bin by day
2. `scripts/58_clean_traj_final.py` — exclude palliative/comfort-directed extubation (WLST adjudication ∪ no-support death proxy)
3. `scripts/58b_clean_traj_model.py`, `58c_neuro_daily.py` — clean-cohort modelling inputs, neurological daily series
4. `scripts/59b_lcgm_refit_d04.py` — LCGM refit on day 0–4 (removes boundary artefact), k = 2–6
5. `r_scripts/03b_lcgm_gbtm_fit.R` — LCGM enumeration (`hlme`, quadratic time, nwg = TRUE), model selection by BIC / entropy / average posterior probability / minimum class size
6. `r_scripts/03d_lcgm_bootstrap_ari.R` — 80% resampling stability (adjusted Rand index)
7. `scripts/59c_k5_phenotype.py`, `scripts/59d_k6_phenotype.py` — phenotype characterisation, trajectory figures, individual slope validation
8. `scripts/59_traj_phenotype_desc.py` — descriptive tables
9. `scripts/86_suppfig1_flow_v2.py` — cohort flow figure
10. `scripts/56_traj_adverse_endpoint.py`, `56b_two_stage.py`, `57_deterioration_v3_full.py` — trajectory-based adverse endpoint (sensitivity analyses)

## Data

**No patient-level data are included in this repository.** All three databases are available through PhysioNet to credentialed users who complete the required human-subjects research training and sign a data use agreement:

- MIMIC-IV v3.1 — https://physionet.org/content/mimiciv/3.1/
- MIMIC-III v1.4 — https://physionet.org/content/mimiciii/1.4/
- eICU-CRD v2.0 — https://physionet.org/content/eicu-crd/2.0/

Under these agreements the authors are not permitted to redistribute patient-level data. Scripts read from a local database root that each user must configure; the paths appear as constants at the top of each script.

## Requirements

- Python 3.11+ (pandas, numpy, scipy, duckdb, matplotlib)
- R 4.5+ with `lcmm`

## Reporting

The study is reported according to STROBE. Trajectory model selection follows the recommendations summarised by van de Schoot (2015) and Nagin & Odgers (2010); LCGM implementation uses `lcmm` (Proust-Lima et al., J Stat Softw 2017;78(2)).

## Licence

Code is released under the MIT Licence (see `LICENSE`). This licence does not extend to any data obtained from PhysioNet, which remain governed by their respective data use agreements.

## Contact

Jiangtao Sheng — jtsheng@stu.edu.cn
Department of Microbiology and Immunology, Shantou University Medical College, Shantou, Guangdong, China
