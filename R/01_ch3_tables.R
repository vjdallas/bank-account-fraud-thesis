# chapter 3 tables and figures, all on one sample with N declared
#
#   raw    1,000,000 rows, sentinel values recoded to NA
#
#   clean  raw minus the rows deleted listwise on the three low missingness
#          variables -> 993,607 rows
#
#   model  clean--the four dropped columns -> 14 continuous + 13 categorical

library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

set.seed(123)

out_dir="outputs/ch3"
fig_dir=file.path(out_dir, "figures")
dir.create(fig_dir, recursive=TRUE, showWarnings=FALSE)

fig_width=6.69    # inches = 17 cm = \linewidth. it doesn't scale in latex please do not change if you work in latex
fig_dpi=300
base_size=12

# 1. load and preprocess
raw=as.data.frame(fread("data/Base.csv", stringsAsFactors=FALSE))
cat("raw rows =", nrow(raw), "| raw fraud =", sum(raw$fraud_bool == 1), "\n")

raw$fraud_bool=factor(raw$fraud_bool, levels=c(0, 1), labels=c("Legit", "Fraud"))

cols_to_factor=c("email_is_free", "phone_home_valid", "phone_mobile_valid",
                 "has_other_cards", "foreign_request", "keep_alive_session",
                 "month", "payment_type", "employment_status",
                 "housing_status", "source", "device_os",
                 "device_distinct_emails_8w")

# sentinel recoding, all six variables, before any factor conversion
raw$prev_address_months_count[raw$prev_address_months_count == -1]=NA
raw$current_address_months_count[raw$current_address_months_count == -1]=NA
raw$bank_months_count[raw$bank_months_count == -1]=NA
raw$session_length_in_minutes[raw$session_length_in_minutes == -1]=NA
raw$device_distinct_emails_8w[raw$device_distinct_emails_8w == -1]=NA
raw$intended_balcon_amount[raw$intended_balcon_amount < 0]=NA

raw[cols_to_factor]=lapply(raw[cols_to_factor], as.factor)

# listwise deletion on the three low missingness variables only
small3=c("current_address_months_count", "device_distinct_emails_8w",
         "session_length_in_minutes")
missing_table=is.na(raw[, small3])
n_miss_per_row=rowSums(missing_table)

cat("rows with exactly 1 / 2 / 3 of the small three missing:",
    sum(n_miss_per_row == 1), "/", sum(n_miss_per_row == 2), "/",
    sum(n_miss_per_row == 3), "\n")

# pairwise co-occurrence, diagonal is the total NA per variable
cat("pairwise co-occurrence of the small three:\n")
print(crossprod(missing_table))

deleted=n_miss_per_row > 0
cat("records deleted =", sum(deleted),
    sprintf("(%.2f%% of the sample)", 100 * sum(deleted) / nrow(raw)), "\n")
cat("of which fraud  =", sum(deleted & raw$fraud_bool == "Fraud"),
    sprintf("(%.2f%% of the minority class)",
            100 * sum(deleted & raw$fraud_bool == "Fraud") /
              sum(raw$fraud_bool == "Fraud")), "\n")

clean=raw[!deleted, ]
clean[cols_to_factor]=lapply(clean[cols_to_factor], droplevels)

n_clean=nrow(clean)
n_fraud=sum(clean$fraud_bool == "Fraud")
base_rate=n_fraud / n_clean

cat("cleaned sample: N =", n_clean, "| fraud =", n_fraud,
    "| legit =", n_clean - n_fraud,
    sprintf("| base rate = %.4f%%", 100 * base_rate), "\n")

# column drops
dropped_cols=c("device_fraud_count", "prev_address_months_count",
               "intended_balcon_amount", "bank_months_count")
model=clean[, !(names(clean) %in% dropped_cols)]

cont_all=names(clean)[sapply(clean, is.numeric)]        # 18
cont_retained=names(model)[sapply(model, is.numeric)]   # 14
cat_vars=cols_to_factor

cat("continuous before drops =", length(cont_all),
    "| after drops =", length(cont_retained),
    "| categorical =", length(cat_vars), "\n")

# ------------------------------------------------------------------------------
# 2. tables

# table 3.4, missing values by fraud class. describes the RAW file
miss_vars=c("prev_address_months_count", "current_address_months_count",
            "intended_balcon_amount", "bank_months_count",
            "session_length_in_minutes", "device_distinct_emails_8w")

t34=list()
for (v in miss_vars) {
  na_flag=is.na(raw[[v]])
  nf=sum(na_flag & raw$fraud_bool == "Legit")
  fr=sum(na_flag & raw$fraud_bool == "Fraud")
  t34[[v]]=data.frame(
    Variable=v,
    NonFraud_N_Missing=nf,
    NonFraud_Pct_of_Tot=round(100 * nf / (nf + fr), 2),
    NonFraud_Rate=round(100 * nf / sum(raw$fraud_bool == "Legit"), 2),
    Fraud_N_Missing=fr,
    Fraud_Pct_of_Tot=round(100 * fr / (nf + fr), 2),
    Fraud_Rate=round(100 * fr / sum(raw$fraud_bool == "Fraud"), 2),
    Total_Missing=nf + fr)
}
t34=do.call(rbind, t34)
write.csv(t34, file.path(out_dir, "T3_4_missing_by_class.csv"), row.names=FALSE)

# table 3.5, outlier analysis on the 14 retained, IQR rule
t35=list()
for (v in cont_retained) {
  x=model[[v]]
  q1=quantile(x, 0.25, na.rm=TRUE)
  q3=quantile(x, 0.75, na.rm=TRUE)
  lower_bound=q1 - 1.5 * (q3 - q1)
  upper_bound=q3 + 1.5 * (q3 - q1)

  flag=!is.na(x) & (x < lower_bound | x > upper_bound)
  n_out=sum(flag)

  t35[[v]]=data.frame(
    Variable=v,
    Lower_Bound=round(as.numeric(lower_bound), 2),
    Upper_Bound=round(as.numeric(upper_bound), 2),
    Outliers_n=n_out,
    Outliers_pct=round(100 * n_out / nrow(model), 2),
    # NA rather than 0 when there are none, so the tex prints "--"
    Fraud_Rate_Among_Outliers=if (n_out == 0) NA else
      round(100 * mean(model$fraud_bool[flag] == "Fraud"), 2))
}
t35=do.call(rbind, t35)
write.csv(t35, file.path(out_dir, "T3_5_outliers.csv"), row.names=FALSE)

# table 3.6, descriptive statistics for the continuous variables.
# reported on all 18 columns of the cleaned sample, so the table that justifies
# dropping device_fraud_count
t36=list()
for (v in cont_all) {
  x=clean[[v]]
  t36[[v]]=data.frame(
    Variable=v,
    Mean=round(mean(x, na.rm=TRUE), 2),
    SD=round(sd(x, na.rm=TRUE), 2),
    Min=round(min(x, na.rm=TRUE), 2),
    Q1=round(as.numeric(quantile(x, 0.25, na.rm=TRUE)), 2),
    Median=round(median(x, na.rm=TRUE), 2),
    Q3=round(as.numeric(quantile(x, 0.75, na.rm=TRUE)), 2),
    Max=round(max(x, na.rm=TRUE), 2),
    Missing_Values=sum(is.na(x)),
    Zeros=sum(x == 0, na.rm=TRUE),
    Positive_Values=sum(x > 0, na.rm=TRUE),
    Negative_Values=sum(x < 0, na.rm=TRUE))
}
t36=do.call(rbind, t36)
t36$Row_Total=t36$Missing_Values + t36$Zeros + t36$Positive_Values + t36$Negative_Values
write.csv(t36, file.path(out_dir, "T3_6_descriptive_continuous.csv"), row.names=FALSE)

# every row must account for all N observations
cat("table 3.6 rows that do not sum to N:",
    paste(t36$Variable[t36$Row_Total != n_clean], collapse=", "), "\n")

# table 3.7, descriptive statistics for the categorical variables
t37=list()
for (v in cat_vars) {
  x=clean[[v]]
  tb=table(x)
  t37[[v]]=data.frame(
    Variable=v,
    Distinct_Levels=length(levels(droplevels(x))),
    Missing_Values=sum(is.na(x)),
    Most_Frequent_Level=names(which.max(tb)),
    Count_Most_Freq=as.integer(max(tb)),
    Percent_Most_Freq=round(100 * max(tb) / sum(!is.na(x)), 2))
}
t37=do.call(rbind, t37)
write.csv(t37, file.path(out_dir, "T3_7_descriptive_categorical.csv"), row.names=FALSE)

# table 3.8, continuous variables split by fraud class
t38=list()
for (v in cont_retained) {
  xl=model[[v]][model$fraud_bool == "Legit"]
  xf=model[[v]][model$fraud_bool == "Fraud"]
  t38[[v]]=data.frame(
    Variable=v,
    Median_Legit=round(median(xl, na.rm=TRUE), 3),
    Median_Fraud=round(median(xf, na.rm=TRUE), 3),
    Delta_Median=round(median(xf, na.rm=TRUE) - median(xl, na.rm=TRUE), 3),
    Mean_Legit=round(mean(xl, na.rm=TRUE), 3),
    Mean_Fraud=round(mean(xf, na.rm=TRUE), 3),
    Delta_Mean=round(mean(xf, na.rm=TRUE) - mean(xl, na.rm=TRUE), 3))
}
t38=do.call(rbind, t38)
t38=t38[order(-abs(t38$Delta_Median)), ]
write.csv(t38, file.path(out_dir, "T3_8_bivariate_continuous.csv"), row.names=FALSE)

# table 3.9, fraud ratio per level of the multi level categoricals
multilevel=c("payment_type", "employment_status", "housing_status", "device_os")
t39=list()
for (v in multilevel) {
  levels_v=levels(clean[[v]])
  d=data.frame(Variable=v, Level=levels_v, Fraud_Count=NA, Total_Count=NA)
  for (i in seq_along(levels_v)) {
    rows=which(clean[[v]] == levels_v[i])
    d$Fraud_Count[i]=sum(clean$fraud_bool[rows] == "Fraud")
    d$Total_Count[i]=length(rows)
  }
  d$Fraud_Ratio=round(d$Fraud_Count / d$Total_Count, 4)
  t39[[v]]=d[order(-d$Fraud_Ratio), ]
}
t39=do.call(rbind, t39)
write.csv(t39, file.path(out_dir, "T3_9_fraud_ratio.csv"), row.names=FALSE)

# table 3.10, binary predictors.
binary_vars=c("foreign_request", "keep_alive_session", "source",
              "phone_mobile_valid", "phone_home_valid", "email_is_free",
              "has_other_cards")
t310=list()
for (v in binary_vars) {
  levels_v=levels(clean[[v]])
  d=data.frame(Variable=v, Level=levels_v, Fraud_Count=NA, Total_Count=NA)
  for (i in seq_along(levels_v)) {
    rows=which(clean[[v]] == levels_v[i])
    d$Fraud_Count[i]=sum(clean$fraud_bool[rows] == "Fraud")
    d$Total_Count[i]=length(rows)
  }
  d$Fraud_Ratio=round(d$Fraud_Count / d$Total_Count, 4)
  d$Lift=round(d$Fraud_Ratio / base_rate, 2)
  t310[[v]]=d[order(-d$Fraud_Ratio), ]
}
t310=do.call(rbind, t310)
write.csv(t310, file.path(out_dir, "T3_10_binary_indicators.csv"), row.names=FALSE)

cat("tables written to", out_dir, "\n")


# 3. figures

thesis_theme=theme_minimal(base_size=base_size) +
  theme(legend.position="top",
        plot.title=element_blank(),
        strip.text=element_text(face="bold", size=base_size),
        panel.border=element_rect(colour="grey80", fill=NA),
        axis.text=element_text(size=base_size - 2))

col_fraud=c("Legit"="skyblue3", "Fraud"="firebrick2")

long_cont=model %>%
  select(all_of(cont_retained), fraud_bool) %>%
  pivot_longer(cols=all_of(cont_retained), names_to="Variable", values_to="Value")

# 14 continuous variables split over two figures
group_a=cont_retained[1:7]
group_b=cont_retained[8:14]

for (g in list(list(vars=group_a, tag="a"), list(vars=group_b, tag="b"))) {
  p_density=ggplot(filter(long_cont, Variable %in% g$vars),
                   aes(x=Value, fill=fraud_bool)) +
    geom_density(alpha=0.5, colour=NA) +
    facet_wrap(~ Variable, scales="free", ncol=2) +
    scale_fill_manual(values=col_fraud, name=NULL) +
    labs(x=NULL, y="Density") + thesis_theme

  p_box=ggplot(filter(long_cont, Variable %in% g$vars),
               aes(x=fraud_bool, y=Value, fill=fraud_bool)) +
    geom_boxplot(alpha=0.7, outlier.size=0.3) +
    facet_wrap(~ Variable, scales="free_y", ncol=2) +
    scale_fill_manual(values=col_fraud, name=NULL) +
    labs(x=NULL, y=NULL) + thesis_theme

  ggsave(file.path(fig_dir, paste0("All_Density_Grid_", g$tag, ".png")),
         p_density, width=fig_width, height=8.2, units="in", dpi=fig_dpi)
  ggsave(file.path(fig_dir, paste0("All_Boxplot_Grid_", g$tag, ".png")),
         p_box, width=fig_width, height=8.2, units="in", dpi=fig_dpi)
}

#top three stongest predictors
for (v in c("credit_risk_score", "name_email_similarity", "income")) {
  d=data.frame(x=model[[v]], fraud_bool=model$fraud_bool)

  p1=ggplot(d, aes(x=x, fill=fraud_bool)) +
    geom_histogram(aes(y=after_stat(density)), position="identity",
                   alpha=0.5, bins=60, colour="white", linewidth=0.1) +
    scale_fill_manual(values=col_fraud, name=NULL) +
    labs(x=v, y="Density") + thesis_theme

  p2=ggplot(d, aes(x=x, fill=fraud_bool)) +
    geom_density(alpha=0.5) +
    scale_fill_manual(values=col_fraud) +
    labs(x=v, y="Density") + thesis_theme + theme(legend.position="none")

  p3=ggplot(d, aes(x=fraud_bool, y=x, fill=fraud_bool)) +
    geom_boxplot(alpha=0.7, outlier.size=0.3) +
    scale_fill_manual(values=col_fraud) +
    labs(x=NULL, y=v) + thesis_theme + theme(legend.position="none")

  ggsave(file.path(fig_dir, paste0("Multi_Num_", v, ".png")),
         (p1 | p2) / p3, width=fig_width, height=6.5, units="in", dpi=fig_dpi)
}

# categorical grid,% of level is fraud
long_cat=clean %>%
  select(all_of(cat_vars), fraud_bool) %>%
  pivot_longer(cols=all_of(cat_vars), names_to="Variable", values_to="Level") %>%
  filter(!is.na(Level)) %>%
  count(Variable, Level, fraud_bool) %>%
  group_by(Variable, Level) %>%
  mutate(pct=100 * n / sum(n)) %>%
  ungroup()

p_cat=ggplot(long_cat, aes(x=Level, y=pct, fill=fraud_bool)) +
  geom_col(position="stack", width=0.7) +
  facet_wrap(~ Variable, scales="free_x", ncol=3) +
  scale_fill_manual(values=col_fraud, name=NULL) +
  labs(x=NULL, y="Percentage within level (%)") +
  thesis_theme + theme(axis.text.x=element_text(angle=45, hjust=1))

ggsave(file.path(fig_dir, "All_Cat_Relative_Grid.png"), p_cat,
       width=fig_width, height=9.0, units="in", dpi=fig_dpi)

# pearson correlation
cor_matrix=cor(model[, cont_retained], use="complete.obs", method="pearson")
write.csv(round(cor_matrix, 4), file.path(out_dir, "pearson_matrix.csv"))

cm_lower=cor_matrix
cm_lower[upper.tri(cm_lower, diag=TRUE)]=NA
cm_long=as.data.frame(as.table(cm_lower))
names(cm_long)=c("V1", "V2", "r")
cm_long=cm_long[!is.na(cm_long$r), ]

p_cor=ggplot(cm_long, aes(x=V2, y=V1, fill=r)) +
  geom_tile(colour="white") +
  geom_text(aes(label=sprintf("%.2f", r)), size=2.4) +
  scale_fill_gradient2(low="firebrick3", mid="white", high="steelblue4",
                       midpoint=0, limits=c(-1, 1), name="r") +
  labs(x=NULL, y=NULL) + thesis_theme +
  theme(axis.text.x=element_text(angle=45, hjust=1, size=7),
        axis.text.y=element_text(size=7),
        panel.grid=element_blank(), panel.border=element_blank())

ggsave(file.path(fig_dir, "Pearson_Correlation.png"), p_cor,
       width=fig_width, height=6.3, units="in", dpi=fig_dpi)

top_pairs=cm_long[order(-abs(cm_long$r)), ][1:10, ]
cat("\nten strongest pairwise correlations:\n")
print(top_pairs, row.names=FALSE)

# vif on the same 14 variables
x_vif=as.matrix(model[, cont_retained])
x_vif=x_vif[complete.cases(x_vif), ]
r_matrix=cor(x_vif)
vif_vals=diag(solve(r_matrix))

vif_table=data.frame(Feature=names(vif_vals),
                     VIF=round(as.numeric(vif_vals), 3),
                     R2=round(1 - 1 / as.numeric(vif_vals), 4))
vif_table=vif_table[order(-vif_table$VIF), ]
write.csv(vif_table, file.path(out_dir, "vif.csv"), row.names=FALSE)

p_vif=ggplot(vif_table, aes(x=reorder(Feature, VIF), y=VIF, fill=VIF)) +
  geom_col(width=0.7) + coord_flip() +
  geom_hline(yintercept=5, linetype="dashed", colour="red") +
  geom_hline(yintercept=10, linetype="solid", colour="darkred") +
  scale_fill_viridis_c(guide="none") +
  labs(x=NULL, y="VIF") + expand_limits(y=10.5) + thesis_theme

ggsave(file.path(fig_dir, "vif.png"), p_vif,
       width=fig_width, height=5.2, units="in", dpi=fig_dpi)

cat("\nvif, descending:\n")
print(vif_table, row.names=FALSE)


cat("\n================ numbers for chapter 3 ================\n")
cat("N raw                :", nrow(raw), "\n")
cat("N cleaned            :", n_clean, "\n")
cat("fraud (cleaned)      :", n_fraud, "\n")
cat("legit (cleaned)      :", n_clean - n_fraud, "\n")
cat("base rate (cleaned)  :", sprintf("%.4f%%", 100 * base_rate), "\n")
cat("records deleted      :", sum(deleted), "\n")
cat("fraud deleted        :", sum(deleted & raw$fraud_bool == "Fraud"), "\n")
cat("continuous retained  :", length(cont_retained), "\n")
cat("categorical          :", length(cat_vars), "\n")
cat("highest vif          :", vif_table$Feature[1], "=", vif_table$VIF[1], "\n")
cat("strongest |r| pair   :", as.character(top_pairs$V1[1]), "~",
    as.character(top_pairs$V2[1]), "=", round(top_pairs$r[1], 3), "\n")
