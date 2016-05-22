rm(list=ls())

library(randomForest)
library(Matrix)
library(foreach)
library(doSNOW)

nodes = 2
registerDoSNOW(makeCluster(nodes, type="SOCK"))

# load raw data
train <- read.csv("../../data/train.csv", header = T, stringsAsFactors = F)
test <- read.csv("../../data/test.csv", header = T, stringsAsFactors = F)
train.y <- as.factor(train$Activity)

# load preprocessed data
# train <- readRDS("../../data/preprocessed/train_preprocessed_full.rds")
# test <- readRDS("../../data/preprocessed/test_preprocessed_full.rds")
# train.y <- as.factor(as.character(train$Activity))

tgt <- 1
print("training started")
set.seed(123)
tree_num <- seq(100, 1000, 50)
for(num in tree_num){
    # if you want to run faster
#     rf.model <- foreach(trees = rep(num/nodes, nodes), .combine = combine, .packages = "randomForest") %dopar%
#         randomForest(x=train[,-tgt], y = train.y, ntree = trees, 
#                      importance = TRUE, type="classification", do.trace = TRUE)
    
    # if you want to tune mtry parameter (takes longer time)    
    rf.model = tuneRF(x=train[,-tgt], y = train.y, mtryStart=25, ntreeTry=num, stepFactor=5, 
                improve=0.05, plot=FALSE, doBest=TRUE, importance = TRUE, trace = TRUE, 
                type="classification")
    
    results <- predict(rf.model, test, type = "prob")
    submission <- data.frame(MoleculeId=seq(1, nrow(test), 1), PredictedProbability=results[,1])
    write.csv(submission, file=sprintf("../results/rf/submission_rf_%s.csv", num), row.names = F)
}