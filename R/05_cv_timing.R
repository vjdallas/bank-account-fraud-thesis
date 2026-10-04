# wall clock cost of the hyperparameter searches.
# 04_final_models.R times the final refits

library(caret)
library(rpart)
library(ranger)
library(xgboost)
library(Matrix)

FRAUD_FIRST=FALSE
source("R/00_data.R")

out_csv="outputs/cv_timings.csv"
dir.create("outputs", showWarnings=FALSE)

n_threads=4
repeats=5
models=c("LR", "DT_Default", "DT_Balanced", "RF_Default", "RF_Balanced",
         "XGB_Default", "XGB_Balanced")

# the tuning subsample for the forests and xgboost
cv_data=train_data[sample_idx, ]

# the logistic regression was tuned on the reduced design matrix
train_lr=train_data[sample_idx, ]

fit_ctrl=trainControl(method="repeatedcv",
                      number=5,
                      repeats=repeats,
                      classProbs=TRUE,
                      summaryFunction=twoClassSummary)

# caret reports total time and final refit time, the difference is the search
cv_seconds=function(fit) {
  unname(fit$times$everything["elapsed"] - fit$times$final["elapsed"])
}

record=function(model, seconds, fits) {
  row=data.frame(Model=model, Threads=n_threads, Seconds=round(seconds, 1),
                 Hours=round(seconds / 3600, 2), Fits=fits,
                 Finished=format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  # one row per model: a re-run replaces that model's row instead of adding a duplicate
  if (file.exists(out_csv)) {
    old=read.csv(out_csv, stringsAsFactors=FALSE)
    row=rbind(old[old$Model != model, ], row)
  }
  write.csv(row, out_csv, row.names=FALSE)
  print(row[row$Model == model, ])
}

# ------------------------------------------------------------------------------
# logistic regression. no tuning, the 25 fits only check stability
if ("LR" %in% models) {
  set.seed(123)
  fit=train(fraud_bool ~ ., data=train_lr, method="glm", family=binomial,
            trControl=fit_ctrl, metric="ROC")
  record("LR", cv_seconds(fit), 5 * repeats)
  rm(fit); gc()
}

# ------------------------------------------------------------------------------
# decision trees. caret grows one tree per resample at the smallest cp and
# prunes it for the other 19 values, so the search costs 25 fits whatever the
# grid size
grid_tree=expand.grid(cp=c(0.0, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03,
                           0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3,
                           0.4, 0.5, 1))

if ("DT_Default" %in% models) {
  set.seed(123)
  fit=train(fraud_bool ~ ., data=train_data, method="rpart",
            trControl=fit_ctrl, metric="ROC", tuneGrid=grid_tree)
  record("DT_Default", cv_seconds(fit), 5 * repeats)
  rm(fit); gc()
}

if ("DT_Balanced" %in% models) {
  set.seed(123)
  fit=train(fraud_bool ~ ., data=train_data, method="rpart",
            trControl=fit_ctrl, metric="ROC", tuneGrid=grid_tree,
            parms=list(prior=c(0.5, 0.5)))
  record("DT_Balanced", cv_seconds(fit), 5 * repeats)
  rm(fit); gc()
}

# ------------------------------------------------------------------------------
# random forests. num.trees is not a caret tuning parameter for ranger, so it
# is looped over
if ("RF_Default" %in% models) {
  grid_rf=expand.grid(mtry=1:5, splitrule="extratrees",
                      min.node.size=c(250, 300, 350, 400, 450))
  seconds=0
  for (nt in c(250, 500, 750, 1000, 2000)) {
    set.seed(123)
    fit=train(fraud_bool ~ ., data=cv_data, method="ranger",
              trControl=fit_ctrl, metric="ROC", tuneGrid=grid_rf,
              num.trees=nt, num.threads=n_threads, importance="none")
    t=cv_seconds(fit)
    cat("RF_Default | num.trees =", nt, "|", round(t, 1), "s\n")
    seconds=seconds + t
    rm(fit); gc()
  }
  record("RF_Default", seconds, 5 * nrow(grid_rf) * 5 * repeats)
}

if ("RF_Balanced" %in% models) {
  grid_rf=expand.grid(mtry=1:3, splitrule="extratrees",
                      min.node.size=c(40, 55, 60, 65))
  seconds=0
  for (nt in c(500, 1000, 2000)) {
    set.seed(123)
    fit=train(fraud_bool ~ ., data=cv_data, method="ranger",
              trControl=fit_ctrl, metric="ROC", tuneGrid=grid_rf,
              num.trees=nt, num.threads=n_threads, importance="none",
              class.weights=rf_class_weights)
    t=cv_seconds(fit)
    cat("RF_Balanced | num.trees =", nt, "|", round(t, 1), "s\n")
    seconds=seconds + t
    rm(fit); gc()
  }
  record("RF_Balanced", seconds, 3 * nrow(grid_rf) * 5 * repeats)
}

# ------------------------------------------------------------------------------
# xgboost
x_cv_enc=sparse.model.matrix(fraud_bool ~ . - 1, data=cv_data)
d_cv=xgb.DMatrix(data=x_cv_enc, label=as.integer(cv_data$fraud_bool == "Fraud"))

if ("XGB_Default" %in% models) {
  grid_xgb=expand.grid(max_depth=c(1, 2, 3),
                       eta=c(0.02, 0.05, 0.1, 0.4),
                       gamma=c(0, 1, 2),
                       colsample_bytree=c(0.6, 0.7, 0.8),
                       min_child_weight=c(0, 1, 2),
                       subsample=c(0.6, 0.7, 0.8))
  seconds=0
  for (i in 1:nrow(grid_xgb)) {
    params=c(list(objective="binary:logistic", eval_metric="auc",
                  nthread=n_threads), as.list(grid_xgb[i, ]))
    set.seed(123)
    t=system.time(xgb.cv(params=params, data=d_cv, nrounds=700, nfold=5,
                         stratified=TRUE, early_stopping_rounds=20,
                         verbose=0))["elapsed"]
    seconds=seconds + t
  }
  record("XGB_Default", seconds, nrow(grid_xgb) * 5)
}

if ("XGB_Balanced" %in% models) {
  grid_xgb=expand.grid(max_depth=c(1, 2, 3),
                       eta=c(0.01, 0.03, 0.04),
                       gamma=c(0, 2, 5),
                       colsample_bytree=c(0.01, 0.05, 0.08),
                       min_child_weight=c(210, 230, 250, 300),
                       subsample=c(0.8, 0.9, 1.0))
  seconds=0
  for (i in 1:nrow(grid_xgb)) {
    params=c(list(objective="binary:logistic", eval_metric="auc",
                  nthread=n_threads, scale_pos_weight=xgb_scale_pos_weight),
             as.list(grid_xgb[i, ]))
    set.seed(123)
    t=system.time(xgb.cv(params=params, data=d_cv, nrounds=3000, nfold=5,
                         stratified=TRUE, early_stopping_rounds=100,
                         verbose=0))["elapsed"]
    seconds=seconds + t
  }
  record("XGB_Balanced", seconds, nrow(grid_xgb) * 5)
}

print(read.csv(out_csv))
