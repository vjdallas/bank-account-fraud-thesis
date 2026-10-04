# cross validation for the decision trees. one cp grid, two variants:
# default priors and balanced priors. selection metric is cv roc-auc.
# the winners (cp=0 and cp=0.001) are what 04_final_models.R refits

library(caret)
library(rpart)

FRAUD_FIRST=FALSE
source("R/00_data.R")

out_dir="outputs/tuning"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

fit_ctrl=trainControl(method="repeatedcv",
                      number=5,
                      repeats=5,
                      classProbs=TRUE,
                      summaryFunction=twoClassSummary)

grid_tree=expand.grid(cp=c(0.0, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03,
                           0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3,
                           0.4, 0.5, 1))

set.seed(123)
cv_tree_default=train(fraud_bool ~ .,
                      data=train_data,
                      method="rpart",
                      trControl=fit_ctrl,
                      metric="ROC",
                      tuneGrid=grid_tree)

set.seed(123)
cv_tree_balanced=train(fraud_bool ~ .,
                       data=train_data,
                       method="rpart",
                       trControl=fit_ctrl,
                       metric="ROC",
                       tuneGrid=grid_tree,
                       parms=list(prior=c(0.5, 0.5)))

results=rbind(data.frame(Variant="default", cv_tree_default$results),
              data.frame(Variant="balanced", cv_tree_balanced$results))
write.csv(results, file.path(out_dir, "dt_cv_results.csv"), row.names=FALSE)

cat("best cp, default priors :", cv_tree_default$bestTune$cp, "\n")
cat("best cp, balanced priors:", cv_tree_balanced$bestTune$cp, "\n")

plot(cv_tree_default, main="Decision tree, default priors")
plot(cv_tree_balanced, main="Decision tree, balanced priors")
