rm(list=ls())

# Reading the raw data
# however, might change to preprocessed in case of need
dat_train <- read.csv("../../data/train.csv", header = T, stringsAsFactors = F)
dat_test <- read.csv("../../data/test.csv", header = T, stringsAsFactors = F)

# apply pca analysis
merged_data <- rbind(dat_train[2:1777], dat_test)
pca_data <- scale(merged_data)
pca <- prcomp(pca_data, scale. = TRUE)

# let's have a look info about the components
summary(pca)

# let's make reduction to specified number of components
components_numb <- 1646
train_pca <- pca$x[1:nrow(dat_train), 1:components_numb] 
train_pca <- data.frame(Activity = dat_train$Activity, train_pca)
a <- nrow(dat_train)+1
b <- nrow(pca$x)
test_pca <- pca$x[a:b, 1:components_numb]


# save as a binary object for faster frequent loading
saveRDS(train_pca, "../../data/preprocessed/train_pca_1646_comp.rds")
saveRDS(test_pca, "../../data/preprocessed/test_pca_1646_comp.rds")
