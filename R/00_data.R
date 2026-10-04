library(data.table)
library(caret)

  df=read.csv("data/Base.csv", stringsAsFactors=FALSE)


cols_to_factor=c("email_is_free", "phone_home_valid", "phone_mobile_valid",
                 "has_other_cards", "foreign_request", "keep_alive_session",
                 "month", "payment_type", "employment_status",
                 "housing_status", "source", "device_os",
                 "device_distinct_emails_8w")


  # recode, drop, factor, delete
  df$prev_address_months_count[df$prev_address_months_count == -1]=NA
  df$current_address_months_count[df$current_address_months_count == -1]=NA
  df$bank_months_count[df$bank_months_count == -1]=NA
  df$session_length_in_minutes[df$session_length_in_minutes == -1]=NA
  df$device_distinct_emails_8w[df$device_distinct_emails_8w == -1]=NA
  df$intended_balcon_amount[df$intended_balcon_amount < 0]=NA

  df$device_fraud_count=NULL
  df$prev_address_months_count=NULL
  df$intended_balcon_amount=NULL
  df$bank_months_count=NULL

  df[cols_to_factor]=lapply(df[cols_to_factor], as.factor)
  df=na.omit(df)
  # na.omit empties the -1 level but leaves it there, and model.matrix would
  # then write a 51st all-zero column
  df$device_distinct_emails_8w=droplevels(df$device_distinct_emails_8w)
  if (!exists("FRAUD_FIRST")) FRAUD_FIRST=TRUE
  lev=if (FRAUD_FIRST) c("Fraud", "Legit") else c("Legit", "Fraud")
  df$fraud_bool=factor(ifelse(df$fraud_bool == 1, "Fraud", "Legit"),
                       levels=lev)

cat("rows after listwise deletion:", nrow(df),
    "| fraud:", sum(df$fraud_bool == "Fraud"), "\n")

set.seed(123)
train_index=createDataPartition(df$fraud_bool, p=0.7, list=FALSE)
train_data=as.data.frame(df[train_index, ])
test_data=as.data.frame(df[-train_index, ])

predictors=setdiff(names(df), "fraud_bool")
x_train=train_data[, predictors]
y_train=train_data$fraud_bool
x_test=test_data[, predictors]
y_test=test_data$fraud_bool

cat("train:", nrow(train_data), "| test:", nrow(test_data),
    "| test fraud:", sum(y_test == "Fraud"), "\n")

# 100,000 row subsample. every cross-validated search runs on this
set.seed(123)
sample_idx=sample(nrow(x_train), 100000)
x_cv=x_train[sample_idx, ]
y_cv=y_train[sample_idx]

# class weights, w_c = n / (2 * n_c).
w_legit=nrow(train_data) / (2 * sum(y_train == "Legit"))
w_fraud=nrow(train_data) / (2 * sum(y_train == "Fraud"))
rf_class_weights=c(Legit=w_legit, Fraud=w_fraud)
xgb_scale_pos_weight=sum(y_train == "Legit") / sum(y_train == "Fraud")

cat("class weights: Legit", round(w_legit, 4), "| Fraud", round(w_fraud, 4),
    "| scale_pos_weight", round(xgb_scale_pos_weight, 4), "\n")
