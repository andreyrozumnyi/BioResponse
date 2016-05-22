rm(list=ls())
library(xgboost)
library(ggplot2)
library(MASS)
library(Matrix)

# choose which source of data you need and comment another part

# read raw data
# train <- read.csv("../../data/train.csv", header = T, stringsAsFactors = F)
# test <- read.csv("../../data/test.csv", header = T, stringsAsFactors = F)
# train.y <- as.numeric(as.character(train$Activity))

# load data after dimensionality reduction
train <- readRDS("../../data/preprocessed/train_pca_1646_comp.rds")
test <- readRDS("../../data/preprocessed/test_pca_1646_comp.rds")
train.y <- as.numeric(as.character(train$Activity))

Train <- sparse.model.matrix(Activity ~ ., data = train)
dtrain <- xgb.DMatrix(data=Train, label=train.y)
watchlist <- list(Train=dtrain)

# take top N results from training grid
data <- read.csv("../results/grid-results-xgb_753.csv")
data <- data[data$bestLoss != -1,]
top_N <- 10
run <- data[order(data$bestLoss)[1:top_N], ]

set.seed(123)
nodes = 4
for(p in 1:top_N){
    param <- list(
        objective  = "binary:logistic",
        booster  = "gbtree",
        eval_metric  = "logloss",
        eta = run[p, 2],
        max_depth  = run[p, 3],
        colsample_bytree  = run[p, 4],
        subsample  = run[p, 5],
        lambda = run[p, 6],
        nthread=nodes
    )
    
    rounds = run[p, 8]
    multiplier = c(1.1, 1.2, 1.3, 1.4)
    for(m in multiplier){
        clf <- xgb.train(
            params= param,
            data  = dtrain,
            nrounds  = round(m*rounds),
            verbose  = 1,
            watchlist  = watchlist,
            maximize  = FALSE
        )
        
        
        test <- as.data.frame(test)
        test$PredictedProbability <- -1
        Test <- sparse.model.matrix(PredictedProbability ~ ., data = test)
        test$ID <- NULL
        
        preds <- predict(clf, Test)
        
        submission <- data.frame(MoleculeId=seq(1, nrow(test), 1), PredictedProbability=preds)
        # write results to submission files
        write.csv(submission, sprintf("../results/xgb_results_753/pca_1646/xgb_pca_%s_753_%s.csv", p, m), row.names=F)
    }
}

