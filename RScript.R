# Check all necessary libraries

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
library(funModeling)
library(corrplot)


#################################################
#  Breast Cancer Project Code e
################################################

#### Data Loading ####
# Wisconsin Breast Cancer Diagnostic Dataset
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2
# Loading the csv data file from my github account

data <- read.csv("https://raw.githubusercontent.com/gmineo/Breast-Cancer-Prediction-Project/master/data.csv")

data$diagnosis <- as.factor(data$diagnosis)
# the 33 column is not right
data[,33] <- NULL


# General Data Info

str(data)
summary(data)

## We have 569 observations with 32 variables. 
head(data)

# Check for missing values

map_int(data, function(.x) sum(is.na(.x)))
## no missing values

# Check proporton of data
prop.table(table(data$diagnosis))


# Distribution of the  Diagnosis COlumn
options(repr.plot.width=4, repr.plot.height=4)
ggplot(data, aes(x=diagnosis))+geom_bar(fill="blue",alpha=0.5)+theme_bw()+labs(title="Distribution of Diagnosis")

# Plotting Numerical Data
plot_num(data %>% select(-id), bins=10) 

# Correlation plot
correlationMatrix <- cor(data[,3:ncol(data)])
corrplot(correlationMatrix, order = "hclust", tl.cex = 1, addrect = 8)

# Find attributes that are highly corrected (ideally >0.90)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# Remove correlated variables
data2 <- data %>%select(-highlyCorrelated)
# Number of columns after removing correlated variables
ncol(data2)

pca_res_data <- prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
plot(pca_res_data, type="l")

# Summary of data after PCA
summary(pca_res_data)


# Reduce the number of variables
pca_res_data2 <- prcomp(data2[,3:ncol(data2)], center = TRUE, scale = TRUE)
plot(pca_res_data2, type="l")
summary(pca_res_data2)

# PC's in the transformed dataset2
pca_df <- as.data.frame(pca_res_data2$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=data$diagnosis)) + geom_point(alpha=0.5)

# Plot of pc1 and pc2
g_pc1 <- ggplot(pca_df, aes(x=PC1, fill=data$diagnosis)) + geom_density(alpha=0.25)  
g_pc2 <- ggplot(pca_df, aes(x=PC2, fill=data$diagnosis)) + geom_density(alpha=0.25)  
grid.arrange(g_pc1, g_pc2, ncol=2)


# Linear Discriminant Analysis (LDA)

# Data with LDA
lda_res_data <- MASS::lda(diagnosis~., data = data, center = TRUE, scale = TRUE) 
lda_res_data

#Data frame of the LDA for visualization purposes
lda_df_predict <- predict(lda_res_data, data)$x %>% as.data.frame() %>% cbind(diagnosis=data$diagnosis)
ggplot(lda_df_predict, aes(x=LD1, fill=diagnosis)) + geom_density(alpha=0.5)


### Model creation


# Creation of the partition 80% and 20%
set.seed(1815) #provare 1234
data3 <- cbind (diagnosis=data$diagnosis, data2)
data_sampling_index <- createDataPartition(data$diagnosis, times=1, p=0.8, list = FALSE)
train_data <- data3[data_sampling_index, ]
test_data <- data3[-data_sampling_index, ]


fitControl <- trainControl(method="cv",    #Control the computational nuances of thetrainfunction
                           number = 15,    #Either the number of folds or number of resampling iterations
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)


### Naive Bayes Model

# Creation of Naive Bayes Model
model_naiveb <- train(diagnosis~.,
                      train_data,
                      method="nb",
                      metric="ROC",
                      preProcess=c('center', 'scale'), #in order to normalize de data
                      trace=FALSE,
                      trControl=fitControl)

# Prediction
prediction_naiveb <- predict(model_naiveb, test_data)
# Confusion matrix
confusionmatrix_naiveb <- confusionMatrix(prediction_naiveb, test_data$diagnosis, positive = "M")
confusionmatrix_naiveb


plot(varImp(model_naiveb), top=10, main="Top variables NaiveBayes")


### Logistic Regression Model 

# Creation of Logistic Regression Model
model_logreg<- train(diagnosis ~., data = train_data, method = "glm",
                     metric = "ROC",
                     preProcess = c("scale", "center"),  # in order to normalize the data
                     trControl= fitControl)
# Prediction
prediction_logreg<- predict(model_logreg, test_data)

# Confusion matrix
confusionmatrix_logreg <- confusionMatrix(prediction_logreg, test_data$diagnosis, positive = "M")
confusionmatrix_logreg

# Plot of top important variables
plot(varImp(model_logreg), top=10, main="Top variables - Log Regr")


### Random Forest Model

# Creation of Random Forest Model
model_randomforest <- train(diagnosis~.,
                            train_data,
                            method="rf",  
                            metric="ROC",
                            preProcess = c('center', 'scale'),
                            trControl=fitControl)
# Prediction
prediction_randomforest <- predict(model_randomforest, test_data)

# Confusion matrix
confusionmatrix_randomforest <- confusionMatrix(prediction_randomforest, test_data$diagnosis, positive = "M")
confusionmatrix_randomforest

# Plot of top important variables
plot(varImp(model_randomforest), top=10, main="Top variables- Random Forest")


### K Nearest Neighbor (KNN) Model

# Creation of K Nearest Neighbor (KNN) Model
model_knn <- train(diagnosis~.,
                   train_data,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10, #The tuneLength parameter tells the algorithm to try different default values for the main parameter
                   #In this case we used 10 default values
                   trControl=fitControl)
# Prediction
prediction_knn <- predict(model_knn, test_data)

# Confusion matrix        
confusionmatrix_knn <- confusionMatrix(prediction_knn, test_data$diagnosis, positive = "M")
confusionmatrix_knn

# Plot of top important variables
plot(varImp(model_knn), top=10, main="Top variables - KNN")

### Neural Network with PCA Model

# Creation of Random Forest Model
model_nnet_pca <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
# Prediction
prediction_nnet_pca <- predict(model_nnet_pca, test_data)
# Confusion matrix
confusionmatrix_nnet_pca <- confusionMatrix(prediction_nnet_pca, test_data$diagnosis, positive = "M")
confusionmatrix_nnet_pca
# Plot of top important variables
plot(varImp(model_nnet_pca), top=8, main="Top variables - NNET PCA")

### Neural Network with LDA Model

# Creation of training set and test set with LDA modified data
train_data_lda <- lda_df_predict[data_sampling_index, ]
test_data_lda <- lda_df_predict[-data_sampling_index, ]

# Creation of Neural Network with LDA Mode
model_nnet_lda <- train(diagnosis~.,
                        train_data_lda,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
# Prediction
prediction_nnet_lda <- predict(model_nnet_lda, test_data_lda)
# Confusion matrix
confusionmatrix_nnet_lda <- confusionMatrix(prediction_nnet_lda, test_data_lda$diagnosis, positive = "M")
confusionmatrix_nnet_lda

# Results

# Creation of the list of all models
models_list <- list(Naive_Bayes=model_naiveb, 
                    Logistic_regr=model_logreg,
                    Random_Forest=model_randomforest,
                    KNN=model_knn,
                    Neural_PCA=model_nnet_pca,
                    Neural_LDA=model_nnet_lda)                                    
models_results <- resamples(models_list)
# Print the summary of models
summary(models_results)
# Plot of the models results
bwplot(models_results, metric="ROC")

# Confusion matrix of the models
confusionmatrix_list <- list(
  Naive_Bayes=confusionmatrix_naiveb, 
  Logistic_regr=confusionmatrix_logreg,
  Random_Forest=confusionmatrix_randomforest,
  KNN=confusionmatrix_knn,
  Neural_PCA=confusionmatrix_nnet_pca,
  Neural_LDA=confusionmatrix_nnet_lda)   
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results %>% knitr::kable()

# Discussion

# Find the best result for each metric
confusionmatrix_results_max <- apply(confusionmatrix_list_results, 1, which.is.max)
output_report <- data.frame(metric=names(confusionmatrix_results_max), 
                            best_model=colnames(confusionmatrix_list_results)
                            [confusionmatrix_results_max],
                            value=mapply(function(x,y) 
                            {confusionmatrix_list_results[x,y]}, 
                            names(confusionmatrix_results_max), 
                            confusionmatrix_results_max))
rownames(output_report) <- NULL
output_report

# Appendix - Enviroment
# Print system information
print("Operating System:")
version
