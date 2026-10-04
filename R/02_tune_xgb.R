# xgboost, default (unweighted) variant.

library(caret)
library(xgboost)

FRAUD_FIRST=FALSE
source("R/00_data.R")

out_dir="outputs/tuning"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

n_cores=4

# one hot encoding, full rank not enforced
enc=dummyVars(~ ., data=x_train)
x_cv_enc=predict(enc, newdata=x_cv)
d_cv=xgb.DMatrix(data=x_cv_enc, label=as.numeric(y_cv == "Fraud"))

grid_xgb=expand.grid(eta=c(0.02, 0.05, 0.1, 0.4),
                     subsample=c(0.60, 0.70, 0.80),
                     max_depth=c(1,2,3),
                     gamma=c(0,1,2),
                     colsample_bytree=c(0.6,0.7,0.8),
                     min_child_weight=c(0,1,2))

max_nrounds=700
early_stop=20
cat("grid size:", nrow(grid_xgb), "combinations\n")

grid_results=list()

for (j in 1:nrow(grid_xgb)) {
  p=grid_xgb[j, ]
  set.seed(123 + j)

  cv_res=xgb.cv(params=list(booster="gbtree",
                            objective="binary:logistic",
                            eval_metric="auc",
                            eta=p$eta,
                            max_depth=p$max_depth,
                            gamma=p$gamma,
                            colsample_bytree=p$colsample_bytree,
                            min_child_weight=p$min_child_weight,
                            subsample=p$subsample,
                            nthread=n_cores),
                data=d_cv,
                nrounds=max_nrounds,
                nfold=5,
                stratified=TRUE,
                early_stopping_rounds=early_stop,
                verbose=0)

  # xgboost 2.0 names the column test-auc-mean, older versions used underscores
  log=cv_res$evaluation_log
  auc_col=grep("test.*auc.*mean", names(log), value=TRUE, ignore.case=TRUE)[1]

  grid_results[[j]]=data.frame(p,
                               Best_Nrounds=which.max(log[[auc_col]]),
                               Mean_AUC=max(log[[auc_col]], na.rm=TRUE))

  cat(format(Sys.time(), "[%H:%M:%S]"), j, "/", nrow(grid_xgb),
      "| auc =", round(grid_results[[j]]$Mean_AUC, 4), "\n")
}

grid_results=do.call(rbind, grid_results)

grid_results$Interior=(grid_results$eta > min(grid_results$eta) &
                       grid_results$eta < max(grid_results$eta) &
                       grid_results$subsample > min(grid_results$subsample) &
                       grid_results$subsample < max(grid_results$subsample))

write.csv(grid_results, file.path(out_dir, "xgb_default_grid.csv"), row.names=FALSE)

best=grid_results[which.max(grid_results$Mean_AUC), ]
cat("\nbest configuration, interior =", best$Interior, "\n")
print(best, row.names=FALSE)
