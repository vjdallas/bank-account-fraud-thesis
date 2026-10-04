# xgboost with scale_pos_weight = n_legit / n_fraud.
library(caret)
library(xgboost)

FRAUD_FIRST=FALSE
source("R/00_data.R")

out_dir="outputs/tuning"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

n_cores=4

enc=dummyVars(~ ., data=x_train)
x_cv_enc=predict(enc, newdata=x_cv)
d_cv=xgb.DMatrix(data=x_cv_enc, label=as.numeric(y_cv == "Fraud"))

grid_xgb=expand.grid(max_depth=c(1, 2, 3),
                     eta=c(0.01, 0.03, 0.04),
                     gamma=c(0, 2, 5),
                     colsample_bytree=c(0.01, 0.05, 0.08),
                     min_child_weight=c(210, 230, 250, 300),
                     subsample=c(0.8, 0.9, 1.0))

max_nrounds=2000
early_stop=75
cat("grid size:", nrow(grid_xgb), "combinations |",
    "scale_pos_weight =", round(xgb_scale_pos_weight, 4), "\n")

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
                            scale_pos_weight=xgb_scale_pos_weight,
                            nthread=n_cores),
                data=d_cv,
                nrounds=max_nrounds,
                nfold=5,
                stratified=TRUE,
                early_stopping_rounds=early_stop,
                verbose=0)

  log=cv_res$evaluation_log
  auc_col=grep("test.*auc.*mean", names(log), value=TRUE, ignore.case=TRUE)[1]

  grid_results[[j]]=data.frame(p,
                               Best_Nrounds=which.max(log[[auc_col]]),
                               Mean_AUC=max(log[[auc_col]], na.rm=TRUE))

  cat(format(Sys.time(), "[%H:%M:%S]"), j, "/", nrow(grid_xgb),
      "| auc =", round(grid_results[[j]]$Mean_AUC, 4), "\n")
}

grid_results=do.call(rbind, grid_results)

# interiority: every swept axis strictly inside its own range
swept=c("eta", "max_depth", "gamma", "colsample_bytree", "min_child_weight",
        "subsample")
grid_results$Interior=TRUE
for (a in swept) {
  v=grid_results[[a]]
  grid_results$Interior=grid_results$Interior & v > min(v) & v < max(v)
}

write.csv(grid_results, file.path(out_dir, "xgb_balanced_grid.csv"), row.names=FALSE)

best=grid_results[which.max(grid_results$Mean_AUC), ]
cat("\nbest configuration overall, interior =", best$Interior, "\n")
print(best, row.names=FALSE)

interior=grid_results[grid_results$Interior, ]
cat("\nbest interior configuration:\n")
print(interior[which.max(interior$Mean_AUC), ], row.names=FALSE)
