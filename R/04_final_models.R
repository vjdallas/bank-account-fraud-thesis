# final models for chapter 4. one cleaned sample, one split, eight models
#
#   logistic regression                            threshold 0.011
#   decision tree 3A   rpart cp=0                  threshold 0.5
#   decision tree 2A   same tree, read lower       threshold 0.011
#   decision tree 3B   rpart balanced priors       threshold 0.5
#   random forest default    ranger 1000 trees     threshold 0.011
#   random forest balanced   ranger 2000 trees     threshold 0.011
#   xgboost default    442 rounds                  threshold 0.011
#   xgboost balanced   1413 rounds                 threshold 0.5
#

library(ranger)
library(xgboost)
library(rpart)
library(pROC)
library(PRROC)
library(Matrix)

FRAUD_FIRST=TRUE
source("R/00_data.R")

out_dir="outputs/ch4"
fig_dir=file.path(out_dir, "figures")
dir.create(fig_dir, recursive=TRUE, showWarnings=FALSE)

n_threads=4
thr_prev=0.011
thr_half=0.5

# xgboost design matrix, drop-first dummies, 50 columns
x_all=model.matrix(~ ., data=df[, predictors])[, -1]
x_train_enc=x_all[train_index, ]
x_test_enc=x_all[-train_index, ]
y_train_num=as.numeric(y_train == "Fraud")
y_test_num=as.numeric(y_test == "Fraud")
cat("design matrix columns:", ncol(x_all), "\n")
rm(x_all); gc()

# ------------------------------------------------------------------------------
# metrics. one block, called once per model and per set, so the eight numbers
# are computed the same way everywhere
get_metrics=function(y, probs, thr) {
  pred=ifelse(probs >= thr, "Fraud", "Legit")
  
  TP=sum(pred == "Fraud" & y == "Fraud")
  FP=sum(pred == "Fraud" & y == "Legit")
  FN=sum(pred == "Legit" & y == "Fraud")
  TN=sum(pred == "Legit" & y == "Legit")
  
  accuracy=(TP + TN) / length(y)
  sensitivity=TP / (TP + FN)
  specificity=TN / (TN + FP)
  precision=TP / (TP + FP)
  f1=2 * (precision * sensitivity) / (precision + sensitivity)
  
  roc_auc=as.numeric(auc(roc(y, probs, levels=c("Legit", "Fraud"),
                             direction="<", quiet=TRUE)))
  pr_auc=pr.curve(scores.class0=probs[y == "Fraud"],
                  scores.class1=probs[y == "Legit"])$auc.integral
  
  # tpr at the 5% false positive budget: sort by score, walk down the list
  ord=order(probs, decreasing=TRUE)
  y_ord=y[ord]
  tpr=cumsum(y_ord == "Fraud") / sum(y == "Fraud")
  fpr=cumsum(y_ord == "Legit") / sum(y == "Legit")
  tpr_at_5fpr=max(tpr[fpr <= 0.05])
  
  data.frame(Threshold=thr, TP=TP, FP=FP, FN=FN, TN=TN,
             Accuracy=accuracy, Sensitivity=sensitivity, Specificity=specificity,
             Precision=precision, F1=f1, ROC_AUC=roc_auc, PR_AUC=pr_auc,
             TPR_at_5FPR=tpr_at_5fpr)
}

# three panel figure: confusion matrix, roc, precision-recall
base_rate=mean(y_test == "Fraud")

plot_model=function(y, probs, thr, file) {
  png(file, width=1650, height=520, res=150)
  par(mfrow=c(1, 3), mar=c(4.4, 4.4, 1, 1), cex=0.9)
  
  pred=ifelse(probs >= thr, "Fraud", "Legit")
  cnt=matrix(c(sum(pred == "Fraud" & y == "Fraud"), sum(pred == "Fraud" & y == "Legit"),
               sum(pred == "Legit" & y == "Fraud"), sum(pred == "Legit" & y == "Legit")),
             nrow=2, byrow=TRUE)
  plot.new(); plot.window(xlim=c(0, 2), ylim=c(0, 2))
  for (r in 1:2) {
    for (k in 1:2) {
      fill=if (r == k) "#cfe3f5" else "#f3d6d1"
      rect(k - 1, 2 - r, k, 3 - r, col=fill, border="white", lwd=2)
      text(k - 0.5, 2.5 - r, formatC(cnt[r, k], format="d", big.mark=","), font=2)
    }
  }
  axis(1, at=c(0.5, 1.5), labels=c("Fraud", "Legit"), tick=FALSE, line=-0.5)
  axis(2, at=c(1.5, 0.5), labels=c("Fraud", "Legit"), tick=FALSE, line=-0.5, las=1)
  mtext("Actual", side=1, line=2.2); mtext("Predicted", side=2, line=2.6)
  
  ord=order(probs, decreasing=TRUE)
  y_ord=y[ord]
  tpr=cumsum(y_ord == "Fraud") / sum(y == "Fraud")
  fpr=cumsum(y_ord == "Legit") / sum(y == "Legit")
  plot(fpr, tpr, type="l", col="red", lwd=2, xlim=c(0, 1), ylim=c(0, 1),
       xlab="False positive rate", ylab="True positive rate")
  abline(0, 1, col="grey70")
  abline(v=0.05, col="grey50", lty=3)
  points(sum(probs >= thr & y == "Legit") / sum(y == "Legit"),
         sum(probs >= thr & y == "Fraud") / sum(y == "Fraud"),
         pch=1, cex=2.4, lwd=2, col="blue")
  legend("bottomright", bty="n", legend=c(paste("operating point", thr), "5% FPR"),
         col=c("blue", "grey50"), lty=c(NA, 3), pch=c(1, NA))
  
  pr=pr.curve(scores.class0=probs[y == "Fraud"], scores.class1=probs[y == "Legit"],
              curve=TRUE)
  plot(pr$curve[, 1], pr$curve[, 2], type="l", col="red", lwd=2,
       xlim=c(0, 1), ylim=c(0, 1), xlab="Recall", ylab="Precision")
  abline(h=base_rate, col="grey60", lty=3)
  dev.off()
}

results=list()
timings=list()

# ------------------------------------------------------------------------------
# logistic regression
cat("\nlogistic regression\n")
lr_data=train_data[, predictors]
lr_data$fraud=y_train_num

t=system.time(logistic_model <- glm(fraud ~ ., data=lr_data, family=binomial))
timings$lr_fit=as.numeric(t["elapsed"])

lr_probs_train=predict(logistic_model, newdata=train_data[, predictors], type="response")
t=system.time(lr_probs_test <- predict(logistic_model, newdata=test_data[, predictors],
                                       type="response"))
timings$lr_predict=as.numeric(t["elapsed"])

results$lr_train=get_metrics(y_train, lr_probs_train, thr_prev)
results$lr_test=get_metrics(y_test, lr_probs_test, thr_prev)

# threshold sweep, table 4.2
thresholds=c(0.001, 0.005, 0.01, 0.011, 0.05, 0.10, 0.50)
sweep=list()
for (th in thresholds) sweep[[as.character(th)]]=get_metrics(y_test, lr_probs_test, th)
sweep=do.call(rbind, sweep)
write.csv(sweep, file.path(out_dir, "lr_threshold_sweep.csv"), row.names=FALSE)

# coefficients and odds ratios
co=coef(logistic_model)
co=co[names(co) != "(Intercept)"]
lr_coefs=data.frame(Feature=names(co), Coefficient=as.numeric(co),
                    Odds_Ratio=exp(as.numeric(co)))
lr_coefs=lr_coefs[order(-abs(lr_coefs$Coefficient)), ]
write.csv(lr_coefs, file.path(out_dir, "lr_coefficients.csv"), row.names=FALSE)

# ------------------------------------------------------------------------------
# decision tree 3A, default priors, cp=0
cat("\ndecision tree 3A\n")
set.seed(123)
t=system.time(tree_3a <- rpart(fraud_bool ~ ., data=train_data, method="class", cp=0))
timings$dt3a_fit=as.numeric(t["elapsed"])

dt3a_probs_train=predict(tree_3a, train_data, type="prob")[, "Fraud"]
t=system.time(dt3a_probs_test <- predict(tree_3a, test_data, type="prob")[, "Fraud"])
timings$dt3a_predict=as.numeric(t["elapsed"])

results$dt3a_train=get_metrics(y_train, dt3a_probs_train, thr_half)
results$dt3a_test=get_metrics(y_test, dt3a_probs_test, thr_half)

# decision tree 2A is the same tree and the same probabilities, read at 0.011
results$dt2a_train=get_metrics(y_train, dt3a_probs_train, thr_prev)
results$dt2a_test=get_metrics(y_test, dt3a_probs_test, thr_prev)

# decision tree 3B, balanced priors, cp=0.001
cat("\ndecision tree 3B\n")
set.seed(123)
t=system.time(tree_3b <- rpart(fraud_bool ~ ., data=train_data, method="class",
                               parms=list(prior=c(0.5, 0.5)), cp=0.001))
timings$dt3b_fit=as.numeric(t["elapsed"])

dt3b_probs_train=predict(tree_3b, train_data, type="prob")[, "Fraud"]
t=system.time(dt3b_probs_test <- predict(tree_3b, test_data, type="prob")[, "Fraud"])
timings$dt3b_predict=as.numeric(t["elapsed"])

results$dt3b_train=get_metrics(y_train, dt3b_probs_train, thr_half)
results$dt3b_test=get_metrics(y_test, dt3b_probs_test, thr_half)

# ------------------------------------------------------------------------------
# random forest. predicting 695k rows at once runs out of memory, so in chunks
rf_predict_chunked=function(model, newdata) {
  out=numeric(nrow(newdata))
  for (s in seq(1, nrow(newdata), by=100000)) {
    e=min(s + 99999, nrow(newdata))
    out[s:e]=predict(model, data=newdata[s:e, predictors],
                     num.threads=n_threads)$predictions[, "Fraud"]
  }
  out
}

# table 4.12: 1000 trees, mtry 3, min.node.size 300, extratrees
cat("\nrandom forest default, 1000 trees\n")
set.seed(123)
t=system.time(rf_default <- ranger(x=x_train, y=y_train,
                                   num.trees=1000, mtry=3, min.node.size=300,
                                   splitrule="extratrees", probability=TRUE,
                                   importance="impurity",
                                   respect.unordered.factors="ignore",
                                   num.threads=n_threads))
timings$rf_default_fit=as.numeric(t["elapsed"])

rfd_probs_train=rf_predict_chunked(rf_default, train_data)
t=system.time(rfd_probs_test <- rf_predict_chunked(rf_default, test_data))
timings$rf_default_predict=as.numeric(t["elapsed"])

results$rf_default_train=get_metrics(y_train, rfd_probs_train, thr_prev)
results$rf_default_test=get_metrics(y_test, rfd_probs_test, thr_prev)

imp=ranger::importance(rf_default)
rf_imp=data.frame(Feature=names(imp), Mean_Decrease_Gini=as.numeric(imp))
rf_imp=rf_imp[order(-rf_imp$Mean_Decrease_Gini), ]
write.csv(rf_imp, file.path(out_dir, "rf_importance.csv"), row.names=FALSE)

# table 4.12: 2000 trees, mtry 2, min.node.size 55, extratrees, class weights
cat("\nrandom forest balanced, 2000 trees\n")
set.seed(123)
t=system.time(rf_balanced <- ranger(x=x_train, y=y_train,
                                    num.trees=2000, mtry=2, min.node.size=55,
                                    splitrule="extratrees", probability=TRUE,
                                    class.weights=rf_class_weights,
                                    respect.unordered.factors="ignore",
                                    num.threads=n_threads))
timings$rf_balanced_fit=as.numeric(t["elapsed"])

rfb_probs_train=rf_predict_chunked(rf_balanced, train_data)
t=system.time(rfb_probs_test <- rf_predict_chunked(rf_balanced, test_data))
timings$rf_balanced_predict=as.numeric(t["elapsed"])

results$rf_balanced_train=get_metrics(y_train, rfb_probs_train, thr_prev)
results$rf_balanced_test=get_metrics(y_test, rfb_probs_test, thr_prev)

# ------------------------------------------------------------------------------
# xgboost
t=system.time(d_train <- xgb.DMatrix(data=x_train_enc, label=y_train_num))
timings$xgb_prep=as.numeric(t["elapsed"])
t=system.time(d_test <- xgb.DMatrix(data=x_test_enc, label=y_test_num))
timings$xgb_prep=timings$xgb_prep + as.numeric(t["elapsed"])

# table 4.19, default column
cat("\nxgboost default, 442 rounds\n")
params_default=list(objective="binary:logistic", eval_metric="auc",
                    max_depth=2, eta=0.05, gamma=1, colsample_bytree=0.7,
                    min_child_weight=1, subsample=0.6,
                    tree_method="hist", nthread=n_threads)
set.seed(123)
t=system.time(xgb_default <- xgb.train(params=params_default, data=d_train,
                                       nrounds=442, verbose=0))
timings$xgb_default_fit=as.numeric(t["elapsed"])

xgd_probs_train=predict(xgb_default, d_train)
t=system.time(xgd_probs_test <- predict(xgb_default, d_test))
timings$xgb_default_predict=as.numeric(t["elapsed"])

results$xgb_default_train=get_metrics(y_train, xgd_probs_train, thr_prev)
results$xgb_default_test=get_metrics(y_test, xgd_probs_test, thr_prev)

# table 4.19, balanced column. nrounds 1413 is the value printed there.
cat("\nxgboost balanced, 1413 rounds\n")
params_balanced=list(objective="binary:logistic", eval_metric="auc",
                     max_depth=2, eta=0.03, gamma=2, colsample_bytree=0.05,
                     min_child_weight=230, subsample=0.9,
                     scale_pos_weight=xgb_scale_pos_weight,
                     tree_method="hist", nthread=n_threads)
set.seed(123)
t=system.time(xgb_balanced <- xgb.train(params=params_balanced, data=d_train,
                                        nrounds=1413, verbose=0))
timings$xgb_balanced_fit=as.numeric(t["elapsed"])

xgb_probs_train=predict(xgb_balanced, d_train)
t=system.time(xgb_probs_test <- predict(xgb_balanced, d_test))
timings$xgb_balanced_predict=as.numeric(t["elapsed"])

results$xgb_balanced_train=get_metrics(y_train, xgb_probs_train, thr_half)
results$xgb_balanced_test=get_metrics(y_test, xgb_probs_test, thr_half)

# ------------------------------------------------------------------------------
# figures, test set
plot_model(y_test, lr_probs_test,   thr_prev, file.path(fig_dir, "lr.png"))
plot_model(y_test, dt3a_probs_test, thr_half, file.path(fig_dir, "dt3a.png"))
plot_model(y_test, dt3a_probs_test, thr_prev, file.path(fig_dir, "dt2a.png"))
plot_model(y_test, dt3b_probs_test, thr_half, file.path(fig_dir, "dt3b.png"))
plot_model(y_test, rfd_probs_test,  thr_prev, file.path(fig_dir, "rf_default.png"))
plot_model(y_test, rfb_probs_test,  thr_prev, file.path(fig_dir, "rf_balanced.png"))
plot_model(y_test, xgd_probs_test,  thr_prev, file.path(fig_dir, "xgb_default.png"))
plot_model(y_test, xgb_probs_test,  thr_half, file.path(fig_dir, "xgb_balanced.png"))

# master table
metrics_all=do.call(rbind, results)
metrics_all=cbind(Model=rownames(metrics_all), metrics_all)
write.csv(metrics_all, file.path(out_dir, "metrics.csv"), row.names=FALSE)

timings_all=data.frame(Stage=names(timings), Seconds=round(unlist(timings), 2))
write.csv(timings_all, file.path(out_dir, "timings.csv"), row.names=FALSE)

capture.output(sessionInfo(), file=file.path(out_dir, "sessionInfo.txt"))

print(metrics_all[grep("_test$", metrics_all$Model), ], row.names=FALSE)
print(timings_all, row.names=FALSE)