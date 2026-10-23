suppressMessages(library(lcmm))
run_cohort <- function(csv_file, prefix) {
  cat("\n======================================================\n")
  cat("COHORT:", prefix, "\n")
  d <- read.csv(csv_file)
  d$day <- as.numeric(d$day); d$sofa <- as.numeric(d$sofa)
  d <- d[order(d$stay_id, d$day), ]
  cat("N stays:", length(unique(d$stay_id)), " | Total obs:", nrow(d), " | Mean SOFA:", round(mean(d$sofa), 2), "\n")
  m1 <- hlme(sofa ~ day + I(day^2), subject="stay_id", data=d, verbose=FALSE)
  for (k in 4:6) {
    m <- tryCatch(hlme(sofa ~ day + I(day^2), subject="stay_id", ng=k, data=d,
        mixture=~day+I(day^2), nwg=TRUE, B=m1, maxiter=150, verbose=FALSE), error=function(e) NULL)
    if (is.null(m)) { cat("k=", k, " FAILED\n"); next }
    pp <- m$pprob
    probs <- as.matrix(pp[, paste0("prob", 1:k)])
    app <- mean(apply(probs, 1, max))
    cat("k=", k, " conv=", m$conv, " BIC=", round(m$BIC, 1), " APP=", round(app, 3), "\n", sep="")
    for (cl in 1:k) {
      ids <- pp[pp$class==cl, "stay_id"]
      dd <- d[d$stay_id %in% ids, ]
      tr <- tapply(dd$sofa, dd$day, mean)
      cat("  class ", cl, " (n=", length(ids), ", ", round(100*length(ids)/nrow(pp),1), "%): ",
          paste(sprintf("%.2f", tr), collapse=" -> "), "\n", sep="")
    }
    write.csv(pp, paste0(prefix, "_k", k, "_pprob.csv"), row.names=FALSE)
  }
}
run_cohort("m3_cv_a_long.csv", "m3_cv_a")
run_cohort("m3_unified_a_long.csv", "m3_unified_a")
run_cohort("eicu_1b_ge3_long.csv", "eicu_1b_ge3_nwg")
