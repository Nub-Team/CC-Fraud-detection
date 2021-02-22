library(tidyverse) 
library(reshape2) 
install.packages('caret')
install.packages('DMwR')
library(caret) 
library(xgboost) 
library(Matrix) 
library(tictoc) 
library(PRROC) 
library(DMwR) 
library(gridExtra) 

theme_set(theme_light())

data <- read.csv("creditcard.csv")

data$Class <- factor(ifelse(data$Class == 0, "zero", "one"))

sapply(data, function(x) sum(is.na(x)))
       
data$time_of_day <- data$Time %%(24*3600)

summary(data$time_of_day)
data$Time<-NULL
ggplot(data, aes(x = time_of_day, fill = Class)) +
  geom_density(alpha = 0.5) + 
  labs(title = "Time of Day", 
       x = "Time of Day", 
       y = "Density", 
       col = "Class") +
  scale_fill_discrete(labels = c("Fraud", "Not Fraud"))
       
summary(data$Amount)
ggplot(data, aes(x = Amount, fill = Class)) +
  geom_density(alpha = 0.3) + 
 scale_x_continuous(limits = c(0, 2000), breaks = seq(0, 1000, 200)) + 
  labs(title = "Amount", 
       x = "Amount", 
       y = "Density", 
       col = "Class") + 
  scale_fill_discrete(labels = c("Fraud", "Not Fraud"))

data$logamount<-log(data$Amount+1)
ggplot(data, aes(x = logamount, fill = Class)) +
  geom_density(alpha = 0.3) + 
 scale_x_continuous(limits = c(0, 10), breaks = seq(0, 10, 1)) + 
  labs(title = "Log Amount", 
       x = "Log Amount", 
       y = "Density", 
       col = "Class") + 
  scale_fill_discrete(labels = c("Fraud", "Not Fraud"))
       
data$Amount<-NULL
       
names(data)

par(mfrow=c(1,1))
for (i in 1:28){
print(ggplot(data, aes_string(x = paste0('V',i), fill = 'Class')) +
  geom_density(alpha = 0.3) + 
  labs(title = paste('PCA feature: V',i), 
       x = paste0('V',i), 
       y = "Density", 
       col = "Class") + 
  scale_fill_discrete(labels = c("Fraud", "Not Fraud")))}
       
corr<-round(cor(data[,!(names(data) %in% c('Class'))]),2)
max(abs(corr)[upper.tri(corr)])
corr[29:30,]
       
features <- data[,!(names(data) %in% c('Class'))]

cbind(melt(apply(features, 2, min), value.name = "min"), 
      melt(apply(features, 2, max), value.name = "max"))
       
rescaler <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

features_rescaled <- as.data.frame(apply(features, 2, rescaler))

cbind(melt(apply(features_rescaled, 2, min), value.name = "min_after_rescaling"), 
      melt(apply(features_rescaled, 2, max), value.name = "max_after_rescaling"))

rescaleddata <- data.frame(cbind(features_rescaled,Class=data$Class))
str(rescaleddata)
       
set.seed(900)
library(caTools)
split <- sample.split(rescaleddata$Class, SplitRatio = 0.7)
train <-  subset(rescaleddata, split == TRUE)
test <- subset(rescaleddata, split == FALSE)
       
train_smote <- SMOTE(Class ~ ., data  = train,
                      perc.over = 500, k = 5, perc.under = 400)

table(train_smote$Class)
table(train$Class)

p1 <- ggplot(train, aes(x = logamount, y = time_of_day, col = Class)) + 
  geom_point(alpha = 0.3) + 
  facet_wrap(~ Class, labeller = labeller(Class = c('zero' = "Not Fraud", 'one' = "Fraud"))) + 
  labs(title = "Before SMOTE", 
       subtitle = "Training Data", 
       col = "Class") + 
  scale_x_continuous(limits = c(0, 1)) + 
  scale_y_continuous(limits = c(0, 1)) + 
  theme(legend.position = "none")

p2 <- ggplot(train_smote, aes(x = logamount, y = time_of_day, col = Class)) + 
  geom_point(alpha = 0.3) + 
  facet_wrap(~ Class, labeller = labeller(Class = c('zero' = "Not Fraud", 'one' = "Fraud"))) + 
  labs(title = "After SMOTE", 
       subtitle = "SMOTE sampling", 
       col = "Class") + 
  scale_x_continuous(limits = c(0, 1)) + 
  scale_y_continuous(limits = c(0, 1)) + 
  theme(legend.position = "none")
grid.arrange(p1, p2, nrow = 2)
       
xgb.data.train <- xgb.DMatrix(as.matrix(train[, colnames(train) != "Class"]), 
                              label = ifelse(train$Class == 'zero', 0, 1))

xgb.data.train.smote <- xgb.DMatrix(as.matrix(train_smote[, colnames(train_smote) != "Class"]), 
                                    label = ifelse(train_smote$Class == 'zero', 0, 1))


xgb.data.test <- xgb.DMatrix(as.matrix(test[, colnames(test) != "Class"]),
                             label = ifelse(test$Class == 'zero', 0, 1))
       
library(microbenchmark)
library(pROC)

xgb.bench.speed = microbenchmark(
    xgb.model.speed <- xgb.train(data = xgb.data.train, 
                                 params = list(objective = "binary:logistic", 
                                               eta = 0.1,
                                               max.depth = 3,
                                               min_child_weight = 100,
                                               subsample = 1,
                                               colsample_bytree = 1,
                                               nthread = 3,
                                               eval_metric = "auc"),
                                 watchlist = list(test = xgb.data.test),
                                 nrounds = 500,
                                 early_stopping_rounds = 40,
                                 print_every_n = 20), times = 5L)
print(xgb.bench.speed)
print(xgb.model.speed$bestScore)
       
xgb.bench.smote.speed = microbenchmark(
    xgb.smote.model.speed <- xgb.train(data = xgb.data.train.smote, 
                                 params = list(objective = "binary:logistic", 
                                               eta = 0.1,
                                               max.depth = 3,
                                               min_child_weight = 100,
                                               subsample = 1,
                                               colsample_bytree = 1,
                                               nthread = 3,
                                               eval_metric = "auc"),
                                 watchlist = list(test = xgb.data.test),
                                 nrounds = 500,
                                 early_stopping_rounds = 40,
                                 print_every_n = 20), times = 5L)
print(xgb.bench.smote.speed)
print(xgb.smote.model.speed$bestScore)
       
pred.resp <- factor(ifelse(xgb.test >= 0.5, 'one', 'zero'))
confusionMatrix(pred.resp, test$Class, positive="one")
       
pred.resp.smote <- factor(ifelse(xgb.test.smote >= 0.5, 'one', 'zero'))
confusionMatrix(pred.resp.smote, test$Class, positive="one")
