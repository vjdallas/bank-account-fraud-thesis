# random forest with inverse frequency class weights.
# grid: num.trees {500, 1000, 2000} x min.node.size {40, 55, 60, 65}.
# num.trees is swept in an outer loop because caret's "ranger" method only
# tunes mtry, splitrule and min.node.size on its own.
#
# once weights are on, the predicted probabilities are no longer calibrated to
# the 1.1% base rate, so accuracy and precision read at 0.011 look extreme.

library(caret)
library(ranger)
library(doParallel)

FRAUD_FIRST=FALSE
source("R/00_data.R")

out_dir="outputs/tuning"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

n_cores=4
cl=makePSOCKcluster(n_cores)
registerDoParallel(cl)

fit_ctrl=trainControl(method="repeatedcv",
                      number=5,
                      repeats=5,
                      classProbs=TRUE,
                      summaryFunction=twoClassSummary,
                      allowParallel=TRUE)

grid_rf=expand.grid(mtry=c(1,2,3), splitrule="extratrees", min.node.size=c(40, 55, 60, 65))
num_trees_values=c(500, 1000, 2000)

rf_bal_results=list()

for (ntrees in num_trees_values) {
  cat(format(Sys.time(), "[%H:%M:%S]"), "num.trees =", ntrees, "\n")

  set.seed(123)
  cv_rf_bal=train(x=x_cv,
                  y=y_cv,
                  method="ranger",
                  trControl=fit_ctrl,
                  metric="ROC",
                  tuneGrid=grid_rf,
                  num.trees=ntrees,
                  num.threads=1,
                  class.weights=rf_class_weights,
                  verbose=FALSE)

  rf_bal_results[[as.character(ntrees)]]=data.frame(Trees=ntrees, cv_rf_bal$results)
}

stopCluster(cl)
registerDoSEQ()

rf_bal_results=do.call(rbind, rf_bal_results)
write.csv(rf_bal_results, file.path(out_dir, "rf_balanced_cv_results.csv"),
          row.names=FALSE)

best=rf_bal_results[which.max(rf_bal_results$ROC), ]
cat("\nbest configuration:\n")
print(best[, c("Trees", "mtry", "splitrule", "min.node.size", "ROC")],
      row.names=FALSE)
