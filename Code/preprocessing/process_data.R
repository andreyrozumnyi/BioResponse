rm(list=ls())

# Reading the data
dat_train <- read.csv("../../data/train.csv", header = T, stringsAsFactors = F)
dat_test <- read.csv("../../data/test.csv", header = T, stringsAsFactors = F)

preprocess <- function(trainset = dat_train, testset = dat_test) {
    # convert to factor categorical features
    train_names <- names(dat_train)[2:ncol(dat_train)]
    for (i in train_names){
        u.train <- unique(dat_train[[i]])
        u.test <- unique(dat_test[[i]])
        
        if (class(dat_train[[i]]) == "integer" & class(dat_test[[i]]) == "integer"){
            if(length(u.train) <= 2 & length(u.test) <= 2){
                dat_train[[i]] <- factor(dat_train[[i]])
                dat_test[[i]] <- factor(dat_test[[i]])
            }
        }
    }
    return(list(train=dat_train, test=dat_test))
}

preprocessed_data <- preprocess(dat_train, dat_test)

# save as a binary object for faster frequent loading
saveRDS(preprocessed_data$train, "../../data/preprocessed/train_preprocessed_full.rds")
saveRDS(preprocessed_data$test, "../../data/preprocessed/test_preprocessed_full.rds")