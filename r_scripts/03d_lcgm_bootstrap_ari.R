# ============================================================
# 03d — SAP 锁4 GRoLTS 收口: bootstrap ARI 稳定性 (x50) + 邻类灵敏度
# 门禁: 中位 ARI >= 0.75 为 PASS (同时给出 ARI>=0.75 的样本比例)
# 输入 (只读):
#   results/lcgm_motor_binned6h_trunc.parquet  (7170 行 / 413 人)
#   results/lcgm/08_lcgm_fits.rds              (list("1".."5"), ng=3 选中)
#   results/lcgm/08_lcgm_classes.csv           (原类标签)
# 输出 (只新建, 不改 03b 产物):
#   results/lcgm/03d_bootstrap_ari.json / 03d_progress.txt /
#   results/lcgm/03d_run.log / 03d_ari_partial.csv (续传中间件)
# 断言: 可续传 (partial csv 已有 rep 跳过); 收敛失败记 failure 不中断;
#       bootstrap 内重复患者各赋独立 sid (random intercept 要求不同 subject)
# ============================================================
suppressWarnings(suppressMessages({ library(arrow); library(lcmm); library(jsonlite) }))
ROOT <- "E:/TBI subtype"
PRJD <- file.path(ROOT, "08_extubation_prediction")
OUTD <- file.path(PRJD, "results/lcgm")
lg  <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n")
pf  <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n",
                         file = file.path(OUTD, "03d_progress.txt"), append = TRUE)
PARTIAL <- file.path(OUTD, "03d_ari_partial.csv")
NREP <- 50; SEED0 <- 20260918

b <- as.data.frame(read_parquet(file.path(PRJD, "results/lcgm_motor_binned6h_trunc.parquet")))
NPAT <- length(unique(b$stay_id))   # 2026-09-17 16:00 终版: 412 人 / 7141 行 (动态, 不写死)
stopifnot(nrow(b) == 7141, NPAT == 412)
ids <- data.frame(stay_id = unique(b$stay_id)); ids$sid <- seq_len(nrow(ids))
b <- merge(b, ids, by = "stay_id"); b$t10 <- b$t / 10
stopifnot(max(b$t) <= 240, min(b$motor) >= 1, max(b$motor) <= 6)

orig_cls <- read.csv(file.path(OUTD, "08_lcgm_classes.csv"))
stopifnot(nrow(orig_cls) == NPAT, setequal(orig_cls$stay_id, ids$stay_id))
orig_map <- setNames(orig_cls$class, orig_cls$stay_id)

# --- ARI 手写实现 (Hubert & Arabie 1985; 与 mclust::adjustedRandIndex 一致,
#     期望机会校正版; 本机未装 mclust 故手写, 公式短可审计) ---
adjusted_rand_index <- function(x, y) {
  tab <- table(x, y)
  choose2 <- function(n) n * (n - 1) / 2
  sij <- sum(choose2(tab)); si <- sum(choose2(rowSums(tab))); sj <- sum(choose2(colSums(tab)))
  n <- sum(tab); tot <- choose2(n)
  exp_idx <- si * sj / tot; max_idx <- (si + sj) / 2
  if (max_idx == exp_idx) return(if (si == 0) 1 else 0)
  (sij - exp_idx) / (max_idx - exp_idx)
}

# ---------- Part 1: 邻类灵敏度 (用已存 fits, 无新拟合) ----------
fits <- readRDS(file.path(OUTD, "08_lcgm_fits.rds"))
m3 <- fits[["3"]]

# 类排序锚: hlme best 参数命名在 mixture 模型下不稳定, 故统一用观测锚 —
# 每类患者基线(首 bin)观测 motor 均值排序(低->高), ng=2/3/4 同锚对齐
class_order_obs <- function(fit) {
  pp <- fit$pprob  # sid, class
  mm <- merge(pp, b, by = "sid")
  ft <- tapply(mm$t, mm$sid, min); mm$ft <- ft[as.character(mm$sid)]
  bm <- aggregate(motor ~ sid, data = mm[mm$t == mm$ft, ], FUN = mean)
  pp2 <- merge(pp[, c("sid","class")], bm, by = "sid")
  om <- aggregate(motor ~ class, pp2, mean, na.rm = TRUE)
  as.integer(om$class[order(om$motor)])
}
ord3 <- class_order_obs(m3)
lab3 <- setNames(seq_along(ord3), ord3)  # 原类 -> 秩 (按基线 motor 低->高)
# 用 data.frame 合并键对齐 (命名向量子集化会丢 stay_id 名, 2026-09-17 实录教训)
d3 <- data.frame(stay_id = ids$stay_id,
                 r3 = as.integer(lab3[as.character(orig_map[as.character(ids$stay_id)])]))

mk_xtab <- function(fit) {
  pp <- merge(fit$pprob[, c("sid","class")], ids, by = "sid")
  ordk <- class_order_obs(fit)
  labk <- setNames(seq_along(ordk), ordk)
  dk <- data.frame(stay_id = pp$stay_id,
                   rk = as.integer(labk[as.character(pp$class)]))
  m <- merge(d3, dk, by = "stay_id")
  tb <- table(ng3_rank = m$r3, ngk_rank = m$rk)
  dimnames(tb)[[2]] <- paste0("k", dimnames(tb)[[2]])
  tb
}
xt2 <- mk_xtab(fits[["2"]]); xt4 <- mk_xtab(fits[["4"]])
lg("邻类交叉表 (类秩按基线 motor 低->高):"); print(xt2); print(xt4)
pf("neighbor xtabs built: ng2xng3, ng4xng3")

# ---------- Part 2: Bootstrap ARI x50 ----------
done <- data.frame()
if (file.exists(PARTIAL)) {
  done <- read.csv(PARTIAL); lg("续传: 已完成", nrow(done), "rep")
}
for (i in 1:NREP) {
  if (i %in% done$rep) next
  t0 <- Sys.time()
  res_i <- tryCatch({
    set.seed(SEED0 + i)
    draws <- sample(ids$stay_id, NPAT, replace = TRUE)  # 患者级有放回
    # 每个抽样单位独立 sid (重复患者 = 独立 subject)
    rows <- do.call(rbind, lapply(seq_along(draws), function(j)
      data.frame(draw = j, stay_id = draws[j], b[b$stay_id == draws[j], c("t10","motor")])))
    rows$sid <- rows$draw
    m1b <- hlme(motor ~ t10 + I(t10^2), random = ~ -1, subject = "sid", ng = 1, data = rows)
    m3b <- gridsearch(rep = 10, maxiter = 20, minit = m1b,
            hlme(motor ~ t10 + I(t10^2), mixture = ~ t10 + I(t10^2),
                 random = ~ -1, subject = "sid", ng = 3, data = rows))
    bl <- m3b$pprob$class; names(bl) <- draws  # 每抽样单位一行
    ari <- adjusted_rand_index(orig_map[as.character(draws)], bl)
    # 空类检查
    sh <- table(bl); empty <- any(sh == 0)
    list(rep = i, ari = ari, status = if (empty) "empty_class" else "ok")
  }, error = function(e) list(rep = i, ari = NA, status = paste0("failure: ", conditionMessage(e))))
  write.table(data.frame(rep = res_i$rep, ari = res_i$ari, status = res_i$status),
              PARTIAL, append = file.exists(PARTIAL), sep = ",",
              col.names = !file.exists(PARTIAL), row.names = FALSE)
  pf(sprintf("rep=%02d ari=%s status=%s 用时=%.1fmin", res_i$rep,
             ifelse(is.na(res_i$ari), "NA", sprintf("%.3f", res_i$ari)), res_i$status,
             as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}

# ---------- Part 3: 汇总 + JSON ----------
d <- read.csv(PARTIAL)
aris <- d$ari[!is.na(d$ari) & d$status == "ok"]
n_fail <- sum(grepl("^failure", d$status)); n_empty <- sum(d$status == "empty_class")
qs <- quantile(aris, c(0.25, 0.5, 0.75), names = FALSE)
gate_pass <- (qs[2] >= 0.75)
pct_ge <- mean(aris >= 0.75) * 100
xt2l <- lapply(rownames(xt2), function(r) setNames(as.integer(xt2[r, ]), colnames(xt2)))
xt4l <- lapply(rownames(xt4), function(r) setNames(as.integer(xt4[r, ]), colnames(xt4)))
names(xt2l) <- paste0("ng3_rank", rownames(xt2)); names(xt4l) <- paste0("ng3_rank", rownames(xt4))
out <- list(
  task = "03d GRoLTS 收口: bootstrap ARI x50 + 邻类灵敏度",
  generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  bootstrap = list(
    n_rep = NREP, seed = paste0(SEED0, " + rep_i (rep_i=1..50)"),
    n_ok = length(aris), n_failure = n_fail, n_empty_class = n_empty,
    ari_all = round(aris, 4),
    ari_median = round(qs[2], 4), ari_q1 = round(qs[1], 4), ari_q3 = round(qs[3], 4),
    ari_min = round(min(aris), 4), ari_max = round(max(aris), 4),
    pct_ge_0.75 = round(pct_ge, 1),
    gate = "中位 ARI >= 0.75", PASS = gate_pass,
    method = "患者级有放回重抽样(412人); 每 rep: set.seed(20260918+i), ng=1 起, ng=3 gridsearch(rep=10, maxiter=20, minit=m1_boot); ARI=手写 Hubert-Arabie 期望机会校正版(对标签置换不变), 比较原类标签 vs bootstrap 类标签 (412 抽样单位, 重复患者各为独立 subject)"),
  neighbor_sensitivity = list(
    class_rank_note = "类秩按各患者基线(首 bin)观测 motor 均值排序(低->高), ng=2/3/4 同锚对齐",
    xtab_ng2_vs_ng3 = xt2l,
    xtab_ng4_vs_ng3 = xt4l,
    ng4_rejected_because = "最小类占比 < 5% 可行性门禁 (03b 冻结规则)"),
  input_check = list(n_rows = nrow(b), n_patients = NPAT,
                     n_classes_csv = nrow(orig_cls)))
write_json(out, file.path(OUTD, "03d_bootstrap_ari.json"),
           pretty = TRUE, auto_unbox = TRUE, na = "null")
pf(sprintf("03D_DONE median_ARI=%.3f pct>=0.75=%.1f%% PASS=%s", qs[2], pct_ge, gate_pass))
cat("DONE\n"); cat(toJSON(out, pretty = TRUE, auto_unbox = TRUE, na = "null"), "\n")
