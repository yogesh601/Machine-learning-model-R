# Welcome to GitHub Desktop!

This project is for Machine Learning model development by using different methods.


Yogesh Kumar
## Step 1: Code for Model development by using different algorithms. 

# Install the packages (You will need these packages for further analysis)
# Call all the libraries
```{r}
library(tidyverse)  # data loading, manipulation and plotting
library(carData)    # Salary dataset
library(broom)      # tidy model output
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library("xlsx") # importing the XLSX files
library(MASS)
library(caTools)
```

# Read the data file from the source (CSV or XLSX) file may be also imported in TXT format.
mydata<- read.xlsx ("...Documents/......./file.xlsx", 1)

#replace NA witth 0 if you have NA in your table

mydata [is.na(mydata)] <- 0
mydata  <- as.data.frame(mydata)

# Normalization of the datasets

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#apply Min-Max normalization to first four columns in iris dataset
mydata_norm <- as.data.frame(lapply(mydata, min_max_norm))
#view first six rows of normalized iris dataset
head(mydata_norm)


# =============Generation correlation matrix======================
# ensure the results are repeatable
set.seed(100)
# load the library
library(mlbench)
library(caret)
# calculate correlation matrix
correlationMatrix <- cor(mydata)
# summarize the correlation matrix
print(correlationMatrix)

# Generating correlation plot where we can select the most dependent variables for our end point
library(corrplot)
correlations <- cor(mydata)
corrplot(correlations, method="circle")

## ========================GLM Logistic regression model fitting=====================
set.seed(222)
ind = sample.split(mydata, SplitRatio = .80)
trainingData = subset(mydata, ind == TRUE)
testData  = subset(mydata, ind == FALSE)

glm.fit <- lm(Endvariable ~., data = trainingData, family = binomial)
summary(glm.fit)
## ============================Neural Network model==============================
n <- neuralnet(Endvariable ~., data = trainingData,
               hidden = c(12,7),
               linear.output = F,
               lifesign = 'full',
               rep=1)
## ===============================Naive Bayes============================================        
 library(naivebayes)
 model_NB <- naive_bayes(Endvariable ~ ., data = trainingData)
  
## =================================SVM===============================================
library(e1071)
svmfit = svm(Endvariable ~., data = trainingData, kernel = "linear", cost = 10, scale = FALSE)
  
## ============================Random Forest============================================
library(mlbench)
library(caret)
library(randomForest)
rf60 <- randomForest(Endvariable ~., data = trainingData, ntrees= 500, mtry = 6, importance=TRUE) 

## ===============================Decision Tree========================================

library(rpart)
library(rpart.plot)
DT <- rpart (Endvariable ~., data = trainingData, method = 'class')

## =========Predicting the different model results============================
glm.predict <- predict(glm.fit,testData, type = "response")
glm.predict

ANN.predict <- predict(n,testData, type = "response")
ANN.predict

NB.predict <- predict(model_NB, testData)
NB.predict

svm.predict <- predict(svmfit, testData, type = "class")
print(svm.predict, 5)

RF.predict <- predict(rf60, testData, type = "class")
RF.predict

DT.predict <- predict(fit, testData, type = "class")
DT.predict

## ===========Calculating the ROC==========================================
PRROC_glm <- roc.curve(scores.class0 = glm.probs,  weights.class0=testData$NAS,curve=TRUE)
PRROC_SVM <- roc.curve(scores.class0 = glm.probs,  weights.class0=testData$NAS,curve=TRUE)
PRROC_RF <- roc.curve(scores.class0 = glm.probs,  weights.class0=testData$NAS,curve=TRUE)
PRROC_ANN <- roc.curve(scores.class0 = glm.probs,  weights.class0=testData$NAS,curve=TRUE)
PRROC_DT <- roc.curve(scores.class0 = glm.probs,  weights.class0=testData$NAS,curve=TRUE)

## ================Section to visualize all the AUC ROC curve using Plot Function======================

plot(PRROC_glm, colorize = F, lwd = 2, type = "l", col = 1) + abline(coef = c(0,1), col = c("grey"),lwd = 2,lty = 2:3,main = "ROC") 
plot(PRROC_SVM, col = 2, lty = 1, lwd = 2, type = "l", add = TRUE, print.auc.adj=c(0,2))
plot(PRROC_RF, col = 3, lty = 1,  lwd = 2,type = "l", add = TRUE, print.auc.adj=c(0,3))
plot(PRROC_ANN, col = 4, lty = 1, lwd = 2,type = "l", add = TRUE, print.auc.adj=c(0,4)) 
plot(PRROC_DT, col = 5, lty = 1, lwd = 2,type = "l", add = TRUE, print.auc.adj=c(0,5))
legend(0.6,0.45, c('AUC (Logistic Regression) = 0.69','AUC (Support Vector Machine) = 0.70','AUC (Random Forest) = 0.63', 'AUC (Neural Network) = 0.66', 'AUC (Decision Tree) = 0.55'),lty=1:1, cex=0.8,
lwd=c(2,2),col=c('1','2','3','4','5'))

##=====================================================================================================

# Enjoy the code and comment if you have any query.

## Thanks.
