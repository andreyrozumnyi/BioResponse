rm(list=ls())
library(xgboost)
library(ggplot2)
library(MASS)
library(ROCR)
library(foreach)
library(doSNOW)
library(Matrix)

# load raw data
train <- read.csv("../../data/train.csv", header = T, stringsAsFactors = F)
test <- read.csv("../../data/test.csv", header = T, stringsAsFactors = F)
train.y <- as.numeric(as.character(train$Activity))

# problem with rocessed data as test containts constants
# load preprocessed data
# train <- readRDS("../../data/preprocessed/train_preprocessed_full.rds")
# test <- readRDS("../../data/preprocessed/test_preprocessed_full.rds")
# train.y <- as.numeric(as.character(train$Activity))

set.seed(123)
nodes = 4

Train <- sparse.model.matrix(Activity ~ ., data = train)
dtrain <- xgb.DMatrix(data=Train, label=train.y)
watchlist <- list(Train=dtrain)

tim = proc.time()
folds = c(8,10)
eta=seq(0.01,0.03,0.002)
max_depth=4:7
colsample_bytree = c(0.85,0.8,1)
subsample = c(0.8,0.95,1)
lambda = c(0,1,3)
# combine all parameters into data frame
N = expand.grid(folds,eta,max_depth,colsample_bytree,subsample,lambda)
results = as.data.frame(N)
names(results)=c("folds","eta","max_depth","colsample_bytree","subsample","lambda")
results$bestLoss = -1
results$bestRound = -1

# main loop for estimating different parameters
for (i in 1:nrow(N)) {
    print(i)
    param <- list(objective  = "binary:logistic",
                  booster  = "gbtree",
                  eval_metric  = "logloss",
                  eta = N[i,2],
                  max_depth  = N[i,3],
                  colsample_bytree  = N[i,4],
                  subsample  = N[i,5],
                  lambda = N[i,6])
    # use early stoping to speed up the process
    CV_current = xgb.cv(params <- param, data <- dtrain, early.stop.round=100, nrounds <- round(10/N[i,2]), 
                        nfold <- N[i,1], nthread = nodes, maximize = FALSE)
    CV_current$Loss = CV_current$test.logloss.mean
    results$bestLoss[i] = min(CV_current$Loss)
    results$bestRound[i] = which.min(CV_current$Loss)
    # may change the name of file for intermediate results if you need
    write.table(results, file="../results/grid-results-xgb.csv", quote = FALSE,
                append = FALSE, col.names=TRUE, row.names=FALSE, sep=",")
}

bestres = which.min(results$bestLoss)
# best params from X-VALIDATION
param <- list(
    objective  = "binary:logistic",
    booster  = "gbtree",
    eval_metric  = "logloss",
    eta = N[bestres,2],
    max_depth  = N[bestres,3],
    colsample_bytree  = N[bestres,4],
    subsample  = N[bestres,5],
    lambda = N[bestres,6],
    nthread=nodes
)
# train with the best parameters from earlier tuning
clf <- xgb.train(
    params= param,
    data  = dtrain,
    nrounds  = round(1*results$bestRound[bestres]),
    verbose  = 1,
    watchlist  = watchlist,
    maximize  = FALSE
)

test$PredictedProbability <- -1
Test <- sparse.model.matrix(PredictedProbability ~ ., data = test)
test$ID <- NULL
# make a prediction for test data
preds <- predict(clf, Test)
# make data.frame with final results
submission <- data.frame(MoleculeId=seq(1, nrow(test), 1), PredictedProbability=preds)
# you may change the name and the path to the file for submission
write.csv(submission, "../results/submission-xgboost_1.csv", row.names = F)