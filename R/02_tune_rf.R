# random forest, default weights. how many trees?
# each count gets a cv roc on the 100k subsample and a refit on the full
# training set scored on the test set. that is what picks 1000

library(caret)
library(ranger)
library(pROC)

FRAUD_FIRST=FALSE
source("R/00_data.R")

out_dir="outputs/tuning"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

fit_ctrl=trainControl(method="repeatedcv",
                      number=5,
                      repeats=5,
                      classProbs=TRUE,
                      summaryFunction=twoClassSummary)

grid_rf=expand.grid(mtry=3, splitrule="extratrees", min.node.size=300)
num_trees_values=c(100, 1000, 2000, 3000, 4000, 5000)   # 1000 is the one used in 04_final_models.R
n_cores=4

rf_results=list()

for (ntrees in num_trees_values) {
  cat(format(Sys.time(), "[%H:%M:%S]"), "num.trees =", ntrees, "\n")

  set.seed(123)
  cv_rf=train(x=x_cv,
              y=y_cv,
              method="ranger",
              trControl=fit_ctrl,
              metric="ROC",
              tuneGrid=grid_rf,
              num.trees=ntrees,
              num.threads=1,
              verbose=FALSE)

  final_rf=ranger(x=x_train,
                  y=y_train,
                  num.trees=ntrees,
                  mtry=3,
                  splitrule="extratrees",
                  min.node.size=300,
                  probability=TRUE,
                  num.threads=n_cores,
                  seed=123)

  test_probs=predict(final_rf, data=x_test)$predictions[, "Fraud"]
  test_auc=as.numeric(auc(roc(y_test, test_probs, levels=c("Legit", "Fraud"),
                              direction="<", quiet=TRUE)))

  rf_results[[as.character(ntrees)]]=data.frame(Trees=ntrees,
                                                CV_ROC=max(cv_rf$results$ROC),
                                                Test_AUC=test_auc)
  cat("  cv roc =", round(max(cv_rf$results$ROC), 4),
      "| test auc =", round(test_auc, 4), "\n")
}

rf_results=do.call(rbind, rf_results)
write.csv(rf_results, file.path(out_dir, "rf_tree_sweep.csv"), row.names=FALSE)
print(rf_results, row.names=FALSE)

plot(rf_results$Trees, rf_results$Test_AUC, type="b", col="firebrick", lwd=2,
     xlab="Number of trees", ylab="AUC", main="AUC vs number of trees")
lines(rf_results$Trees, rf_results$CV_ROC, type="b", col="steelblue", lwd=2)
legend("bottomright", legend=c("Test AUC", "CV ROC"),
       col=c("firebrick", "steelblue"), lwd=2, bty="n")
