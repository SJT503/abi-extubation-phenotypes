# ============================================================
# 03b — 08 拔管队列 LCGM/GBTM 拟合 + 冻结选类 (适配自 07 项目 08a2 模板)
# 依据: SAP v1.6-LOCKED 锁4 轨迹层: M4 拔管队列重拟, 通气小时轴 6h bin,
#       网格上限 240h, 序列截断于 t_ext (防泄漏, 03a 已切)
# 地雷图 (继承 08a2 实录, 违者复踩):
#   1) 禁止直跳 ng=5 — 07 实测收敛空类解 (class5=0); 必须 ng=1 起顺序拟合消耗 RNG
#   2) 种子纪律: 设 seed 后立即 m1, 再顺序 ng=2..5, 不得插入其他随机操作
#   3) 选类规则在拟合前冻结 (本脚本头即冻结): ng∈{2..5} 通过可行性门禁
#      (各类占比>5% 且 各类APP>0.7 无NaN) 者取 BIC 最低; 全不可行 → stop() 报 PI
#   4) 退化解/替换产物必须留审计痕, 不得静默删
# 输入: results/lcgm_motor_binned6h_trunc.parquet (03a 产物, 截断于 t_ext)
# 输出: results/lcgm/ 08_lcgm_fits.rds / 08_lcgm_classes.csv /
#       08_lcgm_metrics.json / 08_lcgm_coefs.json (供 03c 后验引擎)
# ============================================================
suppressWarnings(suppressMessages({ library(arrow); library(lcmm); library(jsonlite) }))
ROOT <- "E:/TBI subtype"
PRJD <- file.path(ROOT, "08_extubation_prediction")
OUTD <- file.path(PRJD, "results/lcgm")
dir.create(OUTD, showWarnings = FALSE, recursive = TRUE)
lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n")
PROG <- file.path(OUTD, "03b_progress.txt")
pf <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n", file = PROG, append = TRUE)

set.seed(20260917)   # 08 项目独立种子 (纪律: seed 后立即 m1)

b <- as.data.frame(read_parquet(file.path(PRJD, "results/lcgm_motor_binned6h_trunc.parquet")))
stopifnot(all(c("stay_id","t","motor") %in% names(b)))   # 03a 实际列: stay_id, bin, t(中心), motor, n_obs
ids <- data.frame(stay_id = unique(b$stay_id)); ids$sid <- seq_len(nrow(ids))
b <- merge(b, ids, by = "stay_id"); b$t10 <- b$t / 10
riskn <- nrow(as.data.frame(arrow::read_parquet(file.path(PRJD, "results/hardgate1_final_riskset.parquet"))))
stopifnot(nrow(ids) == riskn, max(b$t) <= 240, min(b$motor) >= 1, max(b$motor) <= 6)  # 2026-09-17: 413魔法数改动态对齐(C45纠正教训)
lg("08 LCGM 输入:", nrow(b), "行 /", nrow(ids), "患者; t 范围 0-", max(b$t), "h")
pf("03B sequential LCGM n=", nrow(ids), " rows=", nrow(b))

# --- ng=1 (gridsearch minit; 种子后第一动作) ---
m1 <- hlme(motor ~ t10 + I(t10^2), random = ~ -1, subject = "sid", ng = 1, data = b)
pf("ng=1 BIC=", round(m1$BIC, 1)); lg("ng=1 BIC=", round(m1$BIC, 1))

# --- 顺序 ng=2..5 (地雷#1: 禁直跳) ---
fits <- list("1" = m1); curve <- list()
for (k in 2:5) {
  pf("fitting ng=", k); t0 <- Sys.time()
  mk <- gridsearch(rep = 20, maxiter = 20, minit = m1,
          hlme(motor ~ t10 + I(t10^2), mixture = ~ t10 + I(t10^2),
               random = ~ -1, subject = "sid", ng = k, data = b))
  fits[[as.character(k)]] <- mk
  pp <- postprob(mk)
  share <- as.numeric(pp[[1]][1, ]); app <- as.numeric(diag(as.matrix(pp[[2]])))
  viable <- all(share / sum(share) > 0.05) && all(!is.na(app) & app > 0.70)
  curve[[as.character(k)]] <- list(BIC = round(mk$BIC, 1), share = share,
                                   APP = round(app, 3), viable = viable)
  pf(sprintf("ng=%d BIC=%.1f share=%s APP=%s viable=%s 用时=%.0fmin", k, mk$BIC,
             paste(round(share), collapse = "/"), paste(round(app, 3), collapse = "/"),
             viable, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  lg(sprintf("ng=%d BIC=%.1f | share=%s | APP=%s | viable=%s", k, mk$BIC,
             paste(round(share), collapse = "/"), paste(round(app, 3), collapse = "/"), viable))
  if (k == 2) { lg("=== ng=2 完整 summary (参数化/命名实证, 供 03c 解析校准) ===")
                print(summary(mk)) }
}

# --- 冻结选类规则执行 (规则见脚本头, 拟合前已冻结) ---
vn <- names(curve)[sapply(curve, function(x) isTRUE(x$viable))]
if (length(vn) == 0) { pf("03B_NO_VIABLE"); stop("03b: 无可行 ng — 停止报 PI") }
sel <- as.integer(vn[which.min(sapply(curve[vn], function(x) x$BIC))])
bestfit <- fits[[as.character(sel)]]
lg(sprintf("选类: 可行 ng={%s} → 选中 ng=%d (BIC=%.1f)",
           paste(vn, collapse = ","), sel, bestfit$BIC))
pf(sprintf("SELECTED ng=%d BIC=%.1f", sel, bestfit$BIC))
lg(sprintf("=== selected ng=%d summary ===", sel)); print(summary(bestfit))

# --- 表型画像 (对准拔管终点, 非死亡率) ---
cls <- bestfit$pprob[, c("sid", "class")]; cls <- merge(cls, ids, by = "sid")
risk <- as.data.frame(read_parquet(file.path(PRJD, "results/hardgate1_final_riskset.parquet")))[, c("stay_id","reint48")]
feat <- as.data.frame(read_parquet(file.path(PRJD, "results/features_m4.parquet")))[, c("stay_id","gcs_motor_ext","pre_inv_h")]
o <- Reduce(function(a, bb) merge(a, bb, by = "stay_id", all.x = TRUE), list(cls, risk, feat))
prof <- aggregate(o[, c("reint48","gcs_motor_ext","pre_inv_h")],
                  list(phenotype = o$class), function(x) round(mean(x, na.rm = TRUE), 3))
prof$n <- as.integer(table(o$class))
print(prof, row.names = FALSE)

# --- 落盘 (fits RDS 供 03d bootstrap ARI 复用) ---
saveRDS(fits, file.path(OUTD, "08_lcgm_fits.rds"))
write.csv(o[, c("stay_id","class")], file.path(OUTD, "08_lcgm_classes.csv"), row.names = FALSE)
write_json(list(
  n = nrow(ids), n_rows = nrow(b),
  BIC_by_ng = c("1" = round(m1$BIC, 1), sapply(curve, function(x) x$BIC)),
  selected_ng = sel,
  selection_rule = "viable = 各类占比>5% 且 各类APP>0.7; 可行者取 BIC 最低 (规则冻结于拟合前; 继承 08a2 同族条款: ng 敏感性 + 正面披露, 不得静默调类)",
  curve = curve,
  phenotype_profile = lapply(seq_len(nrow(prof)), function(i) as.list(prof[i, ])),
  spec = "y=motor(1-6) ~ t10 + I(t10^2), mixture 同, random intercept, t10=t/10, 6h bin 截断于 t_ext, 上限 240h, seed 20260917, 413 人拔管风险集"),
  file.path(OUTD, "08_lcgm_metrics.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")

# --- 系数冻结 (数组式, 保重复名; 供 03c Python 截断后验引擎) ---
dump_fit2 <- function(fit) {
  nm <- names(fit$best); vv <- as.numeric(fit$best)
  list(BIC = round(fit$BIC, 1),
       best_vec = lapply(seq_along(nm), function(i) list(name = nm[i], value = vv[i])))
}
coefs <- list(
  generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  spec = list(formula = "motor ~ t10 + I(t10^2), mixture = ~ t10 + I(t10^2), random = ~ -1 (random intercept), t10 = t/10, binned 6h grid 截断于 t_ext, seed 20260917",
              var_time = "t10", resp_scale = "motor 1-6",
              selection = "ng 由 03b 冻结规则选出 (见 08_lcgm_metrics.json)"),
  motor = list(n = nrow(ids), ng = sel,
               class_share = curve[[as.character(sel)]]$share,
               APP = round(curve[[as.character(sel)]]$APP, 3),
               fit = dump_fit2(bestfit)))
write_json(coefs, file.path(OUTD, "08_lcgm_coefs.json"),
           pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "null")

pf("03B_DONE"); lg("03B_LCGM_GBTM_FIT_DONE")
