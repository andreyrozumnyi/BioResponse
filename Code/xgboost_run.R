rm(list=ls())
library(xgboost)
library(ggplot2)
library(MASS)
library(Matrix)

train <- read.csv("../data/train.csv", header = T, stringsAsFactors = F)
test <- read.csv("../data/test.csv", header = T, stringsAsFactors = F)
train.y <- as.numeric(as.character(train$Activity))

set.seed(123)
nodes = 4

Train <- sparse.model.matrix(Activity ~ ., data = train)
dtrain <- xgb.DMatrix(data=Train, label=train.y)
watchlist <- list(Train=dtrain)


data <- read.csv("grid-results-xgb.csv")
data <- data[data$bestLoss != -1,]
run <- data[order(data$bestLoss)[2:25], ]

rnd.seed = c(1234, 12345, 777, 4321)
for(rnd in rnd.seed){
    set.seed(rnd)
    for(p in c(3, 6, 12)){
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
        multiplier = c(1.2, 1.3, 1.4)
        for(m in multiplier){
            clf <- xgb.train(
                params= param,
                data  = dtrain,
                nrounds  = round(m*rounds),
                verbose  = 1,
                watchlist  = watchlist,
                maximize  = FALSE
            )
            
            
            test$PredictedProbability <- -1
            Test <- sparse.model.matrix(PredictedProbability ~ ., data = test)
            test$ID <- NULL
            
            #preds <- predict(clf, test)
            preds <- predict(clf, Test)
            
            submission <- data.frame(MoleculeId=seq(1, nrow(test), 1), PredictedProbability=preds)
            write.csv(submission, sprintf("xgb_results/xgboost_%s_%s_283_%s.csv", p, rnd, m), row.names = F)
        }
    }
}
