rm(list=ls())

library(randomForest)
library(Matrix)
library(foreach)
library(doSNOW)

nodes <- 2 
registerDoSNOW(makeCluster(nodes, type="SOCK"))

set.seed(123)

# load raw data
train <- read.csv("../../data/train.csv", header = T, stringsAsFactors = F)
test <- read.csv("../../data/test.csv", header = T, stringsAsFactors = F)
train.y <- as.factor(train$Activity)

# load preprocessed data
train <- readRDS("../../data/preprocessed/train_preprocessed_full.rds")
test <- readRDS("../../data/preprocessed/test_preprocessed_full.rds")
train.y <- as.factor(train$Activity)


tgt <- 1
print("training started")
rf <- foreach(trees = rep(500/nodes, nodes), .combine = combine, .packages = "randomForest") %dopar%
    randomForest(x=train[,-tgt], y = train.y, ntree = trees, importance = TRUE)

#saveRDS(rf, file=sprintf("Rf_classifier.rds"))

print("testing started")
results <- predict(rf, test, type = "prob")

submission <- data.frame(MoleculeId=seq(1, nrow(test), 1), PredictedProbability=results[,1])

write.csv(submission, file=sprintf("submission_rf.csv"), row.names = F)