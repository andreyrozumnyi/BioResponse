rm(list=ls())
library(e1071)
set.seed(123)

# load raw data
train <- read.csv("../../data/train.csv", header = T, stringsAsFactors = F)
test <- read.csv("../../data/test.csv", header = T, stringsAsFactors = F)


# load preprocessed data
# train <- readRDS("../../data/preprocessed/train_preprocessed_full.rds")
# test <- readRDS("../../data/preprocessed/test_preprocessed_full.rds")

# Radial kernel
obj = tune(svm, as.factor(Activity) ~ ., data = train, kernel = "radial", 
          ranges = list(gamma = 10^(seq(-9, -3, 3)), cost = 10^(seq(3, 6, 3))))

print("best parameters are:")
print(obj)
svm.rbf = svm(as.factor(Activity) ~ ., data = train, kernel = "radial", 
              gamma = obj$best.parameters[1], 
              cost = obj$best.parameters[2],
              probability = TRUE)

preds.rbf = predict(svm.rbf, test, probability = TRUE, decision.values = TRUE)
submission.rbf <- data.frame(MoleculeId=seq(1, nrow(test), 1), PredictedProbability=attr(preds.rbf, "probabilities")[, 2])
write.csv(submission.rbf, file=sprintf("../results/svm/submission_rbfsvm.csv"), row.names = F)