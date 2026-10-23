# ==============================================================================
# 04_m3_carevue_lcgm_k2_8.R
# Latent class growth modelling (k = 2..8) for MIMIC-III CareVue (2001-2008, n = 810)
# Class-specific quadratic time trajectories of daily post-extubation SOFA (day 0-4)
# ==============================================================================
suppressMessages(library(lcmm))

calc_entropy <- function(probs, k) {
  n <- nrow(probs)
  p <- pmax(probs, 1e-12)
  1 - (-sum(p * log(p))) / (n * log(k))
}

run_lcgm_k2_8 <- function(csv_file = "m3_cv_a_long.csv", prefix = "m3_cv_a") {
  d <- read.csv(csv_file)
  d$day <- as.numeric(d$day)
  d$sofa <- as.numeric(d$sofa)
  d <- d[order(d$stay_id, d$day), ]
  cat("Cohort:", prefix, "| N stays:", length(unique(d$stay_id)), "| Obs:", nrow(d), "\n")

  m1 <- hlme(sofa ~ day + I(day^2), subject = "stay_id", data = d, verbose = FALSE)
  stats_rows <- list()

  for (k in 2:8) {
    m <- hlme(
      fixed = sofa ~ day + I(day^2),
      mixture = ~ day + I(day^2),
      subject = "stay_id",
      ng = k,
      nwg = FALSE,
      B = m1,
      maxiter = 200,
      data = d,
      verbose = FALSE
    )
    saveRDS(m, paste0(prefix, "_k", k, ".rds"))
    pp <- m$pprob
    write.csv(pp, paste0(prefix, "_k", k, "_pprob.csv"), row.names = FALSE)

    probs <- as.matrix(pp[, paste0("prob", 1:k)])
    ent <- calc_entropy(probs, k)
    app <- mean(apply(probs, 1, max))
    tab <- table(factor(pp$class, levels = 1:k))
    sizes_str <- paste(as.integer(tab), collapse = "; ")
    min_nonempty <- min(as.integer(tab[tab > 0]))

    cat(sprintf(
      "k=%d | conv=%d | BIC=%.1f | entropy=%.3f | APP=%.3f | min_n=%d | sizes=%s\n",
      k, m$conv, m$BIC, ent, app, min_nonempty, sizes_str
    ))
    stats_rows[[length(stats_rows) + 1]] <- data.frame(
      k = k, conv = m$conv, loglik = m$loglik, BIC = m$BIC,
      entropy = ent, mean_app = app, min_n = min_nonempty,
      sizes = sizes_str, stringsAsFactors = FALSE
    )
  }

  out_df <- do.call(rbind, stats_rows)
  write.csv(out_df, paste0(prefix, "_k28_stats.csv"), row.names = FALSE)
}

if (file.exists("m3_cv_a_long.csv")) {
  run_lcgm_k2_8("m3_cv_a_long.csv", "m3_cv_a")
}
