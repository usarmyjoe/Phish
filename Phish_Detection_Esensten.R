#Hello,
#This script is part of the EdX Data Science 125.9x Capstone Project for Joseph Esensten
#
#The purpose of the script is to predict phishing emails using machine learning algorythms.
#Phishing is a type of attack used in computing in which malicious emails masquerade as legitimate to fool users into taking an action which compromises security.
#Emails are the method used for phishing attacks. Emails use well-known protocols which date back to 1982.Sending and recieving emails leaves electronic trails which
#can be used to classify them. These classifications are derived from the email itself which contains metadata to ensure delivery.
#The algorythm with the highest accuracy will be selected after trying several with the data set.
#
#Written in R 3.6

#############################
#Load Data                  #
#############################

#Data Set Metadata:

#Author: colin89lin@gmail.com
#Phishing Features Dataset	This dataset contains comprehensive and up-to-date features of phishing and legitimate webpages.	
#This dataset contains 48 features extracted from 5000 phishing webpages and 5000 legitimate webpages. 
#The webpages were downloaded from January to May 2015 and from May to June 2017. 
#Phishing webpages were sourced from PhishTank and OpenPhish, while legitimate webpages were sourced from Alexa and Common Crawl. 
#This dataset is WEKA-ready and can be used for machine learning-based phishing detection, phishing features analysis or rapid proof of concept experiments.	
#ARFF 	1.30 MB
#Repositary name: 	Open Data
#Persistent identifier of the data set: 	http://dx.doi.org/10.17632/h3cgnj8hft.1
#Deep-link URL to the data set: 	https://data.mendeley.com/datasets/h3cgnj8hft/1

#Load libraries if required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(farff)) install.packages("farff", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(ElemStatLearn)) install.packages("ElemStatLearn", repos = "http://cran.us.r-project.org")
if(!require(monmlp)) install.packages("monmlp", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(pander)) install.packages("pander", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

#Package housekeeping
confusionMatrix <- caret::confusionMatrix
train <- caret::train

#Data Set
#The Data set....
cat("Downloading Dataset\n")
#Download the data from source "https://data.mendeley.com/datasets/h3cgnj8hft/1/files/286768bb-83f2-4e59-9210-6fed84e3c7fd/Phishing_Legitimate_full.arff?dl=1"
rawData <- tempfile()
download.file("https://data.mendeley.com/datasets/h3cgnj8hft/1/files/286768bb-83f2-4e59-9210-6fed84e3c7fd/Phishing_Legitimate_full.arff?dl=1",rawData)

#Read arff file into phishData
phishData <- farff::readARFF(rawData)

##############################
#Validate Data               #
##############################
#glimpse(phishData)

#Check for NAs
if(any(apply(phishData, 2, is.na))){
   cat("ERROR IN SOURCE DATA...Stopping")
   stop()
}
dim(phishData)

##############################
#Data Exploration            #
##############################
#Fix the key column of CLASS_LABEL to reflect what it is "Phish"
setnames(phishData, old=c("CLASS_LABEL"), new=c("Phish"))
#Use R valid names
#levels(phishData$Phish) <- list(No="0",Yes="1")
#Phish also has the problem of being a factor with 2 levels. Really just want numeric.
#phishData[49] <- as.numeric(as.character(unlist(phishData[49])))
#Create an easy to reference data model for the report and my own use
phishModel <- data.frame(names(phishData), typeof(phishData[,2]))
#There was no description of these data so I added as many as I could resolve.
phishModel["Description"] <- c("Number of dots in url", "# Levels of subdomains in URL", "# Levels of path in URL", "URL Total Length", "# of - symbols", "# of - symbols in hostname", "# of @ symbols", "# of ~ symbols","# of _ symbols", "# of % symbols","Number of Queries =?","# of & Symbols","# of # symbols","# of Numbers","Non HTTPS Page","Contains Random String","URL is an IP Address","Domain Name contained in Subdomains (google.google.com)","Domain Name contained in paths (www.google.com/google)","HTTPS in name (https.google.com)","Length of hostname","Length of path","Length of query","Path contains double slash","# of sensitive words","Inc Brand Name (Bank of America)","% External Hyperlinks","% External resource URLs","External icon","Insecure forms submission","Form action (submit) is relative to the page (../action)","External form action","Abnormal form action","_target can be a trick to get a new blank page to open yet still have some control over the original page. % of links that do this.", "Common domain name does not match certificate (google.com)", "onMouseOver: Fake URL shown in status bar to trick users", "Code disables right click", "Code contains a popup", "Code contains a mailto:", "Frames used", "Title of page missing", "Not sure here. Guessing images in forms are not good.", "","","","","", "Null _target redirect attack method to an external site","Phishing = 1, Not = 0")
#Name Column Headers
names(phishModel) <- c("Attribute","Type","Description")

#The source data is split between 50% legitimate and 50% phishing emails.
sum(phishData$Phish == 0) #Legitimate
sum(phishData$Phish == 1) #Phishing
phishN <- nrow(phishData) #total

#The data column "Phish" is a designator 1, or 0. 1 meaning a phishing email, 0 meaning a legitimate email.

##############################
#PREPARE FOR SCIENCE!!!!!!!!!#
##############################
#Set sequence for random number generation
set.seed(66, sample.kind = "Rounding")


##############################
#PreProcessing               #
##############################
cat("Preprocessing\n")
#ID columns with no Variance
phishNZVn <- nearZeroVar(phishData, names=TRUE)
phishNZV <- nearZeroVar(phishData)


#So lets remove non-varying data.This is not helpful
phishData <- phishData[,-phishNZV]

#Check for correlation between data
phishCory <- ifelse(phishData$Phish == "1", 1,0)
phishCorData <- phishData
phishCorData$Phish <- phishCory
phishCor <- cor(as.vector(phishCorData),use="complete.obs")
#Sort PhishCor for report
phishCorSort <- phishCor[order(phishCor[,1], decreasing=TRUE),]
#Search for highly correlated variables.
phishHighCor <- findCorrelation(phishCor, cutoff = 0.9, names=TRUE)

#Get ranges of data
phishRange <- apply(phishData, 2, range)
#Much of the data is binary or ternary.

#Dummy Data for Categorical Values (Phish)
phishDumy <- dummyVars(Phish ~., data=phishData)

#Removal of outliers identified during logistic regression:
#phishData <- phishData[-c(3792,3012,4278,2394,3917,1973,3083,4719,3788,3791,4721,3337,3016,3419,3084,918,4715),]

#Lets shuffle the data to remove any bias
shuffle <- sample(nrow(phishData))
phishData <- phishData[shuffle, ]

#Split the data into training and test sets
#We will create a 20% test set, leaving 80% for training.
phishIndex <- createDataPartition(y=phishData$Phish, times=1, p=.2, list=FALSE)
phishTrain <- phishData[-phishIndex,] #training set
phishTest <- phishData[phishIndex,] #test set

#Specific Data Sets for Algorythms
phishTrainTarget <- phishTrain$Phish
phishTestTarget <- phishTest$Phish
phishTrain2 <- phishTrain[,-32]
phishTest2 <- phishTest[,-32]
summary(phishTrain)
summary(phishTest)

#For comparison
phishTrainB4 <- phishTrain
#We need to make our data ready for Prediction models. 
#
#Preprocess
phishPre <- preProcess(phishTrain, method=c("range"))
#Scale, center and normalize the training data with the training data parameters
phishTrain <- predict(phishPre, phishTrain)
#Scale, center and normalize the test data with the training data parameters
phishTest <- predict(phishPre, phishTest)

##############################
#Prediction Hocus Pocus      #
##############################
#Set up some things

#I considered various machine learning algorythms. Knowing a bit about email security I was able to select methods which I 
#believed would produce quality results. For instance, Naive Bayes equally weighs data features. 
#I know from experience that this would produce skewed results given that all features are not created equal. 
#Therefore, naive bayes will not be used.
#I selected the following algorythms to test their accuracy with predicting phisihing emails:
#K-Nearest Neighbors
#Random Forests
#Logistic Regression
#Penalized Logistic Regression
#
#The results of each will be tracked for comparison.

#Create tracker for results
#Model
#Accuracy
#Root Mean Squared Error to measure how far the data points are from the regression line.
#In human terms: how spread out is the prediction from the actuals.
phishTrack <- data.frame("Model" = character(0),"Accuracy"=integer(0), "RMSE"=integer(0), stringsAsFactors = FALSE)


#Random Forest########
cat("Starting Random Forest\n")
#Random Forest########
#Random Forest Start

#Random Forests combines multiple decision trees.
#Tuning involves changing ntree, mtry for the best accuracy.
#I will be using the data with the target split out.
#Ballpark tuning using tuneRF using OOBError
phishFTuneRF <- randomForest::tuneRF(
  x = phishTrain2,
  y = phishTrainTarget,
  ntreeTry = 500,
  mtryStart = 4,
  stepFactor = 1.5,
  improve = 0.001,
  trace=FALSE
)
#From TuneRF, it looks like 6 is the best value of mtry which minizes OOB Error. I will revalidate later.
#Take a look at trees
phishFRF <- randomForest::randomForest(formula = Phish ~ ., data=phishTrain)
#Classes=Colored and OOB=Black

#For this effort, ranger will be used over randomforest for efficiency.
#Create a grid for tuning parameters for ranger
phishFParam <- expand.grid(
  mtry=seq(2,8),
  node_size=c(4,5,6,7,8),
  samp_size=c(.6,.7,.8),
  splitrule=c("gini","extratrees")
)
#Validate Tuning Parameters
for (i in 1:nrow(phishFParam)) {
  Model <- ranger(
    formula = Phish ~ .,
    data = phishTrain,
    num.trees=500,
    mtry=phishFParam$mtry[i],
  )
  #Grab prediction error and add back to the parameter tracker
  phishFParam$RMSE[i] <- sqrt(Model$prediction.error)
  #Get Accuracy and add back to the parameter tracker
  phishFParam$Accuracy[i] <- confusionMatrix(Model$predictions,phishTrainTarget)$overall["Accuracy"]
}
#Report the max accuracy value:
phishFParam %>% filter(Accuracy == max(phishFParam$Accuracy))
#Build the final model off the best tuning parameters
phishFModel <- ranger(
  formula = Phish ~ .,
  data = phishTrain,
  num.trees=500,
  mtry=4,
  min.node.size=1,
  sample.fraction=.8,
  importance="impurity",
  splitrule="gini"
)
#Get predictions
phishFPred <- predict(phishFModel, data=phishTest)

#DEFUNCT CODE BELOW
#Settings for Random Forest. Crossvalidation and random search for efficiency. Defaults for number and repeats
#phishFCtrl1 <- trainControl(method="cv",number=10, search="random")
#Picking random predictors should not exceed predictor count. 
#phishFGrid1 <- expand.grid(mtry=seq(2,6)) #revalidate mtry value for accuracy but evaluating left and right of best mtry.
#phishFTrain1 <- train(phishTrain2,phishTrainTarget,method="rf", nTree= 500, trControl = phishFCtrl1, tuneGrid = phishFGrid1) 
#Lets tune ntree 

#I commented this out due to the time it took to run. But the results are below.
#ntreeTune <- list()
#for (ntree in c(250,500,1000,2000)){
#  fit <- train(phishTrain2,phishTrainTarget,
#               method = 'rf',
#               metric = 'Accuracy',
#               tuneGrid = phishFGrid,
#               trControl = phishFCtrl,
#               ntree = ntree)
#  moo <- toString(ntree)
#  ntreeTune[[moo]] <- fit
#}
#ntreeResults <- resamples(ntreeTune)
#ntreeSum <- summary(ntreeResults)
#ntreePlot <- data.frame(ntreeSum$statistics$Accuracy)
#ntreePlot$ntree <- rownames(ntreePlot)
#1000 trees was the sweet spot with mtry of 4.
#phishFTree <- 1000
#phishFCtrl <- trainControl(method="cv",number=10, search="random")
#phishFGrid <- expand.grid(mtry=phishFTrain1$bestTune$mtry)
#phishFTrain <- train(phishTrain2,phishTrainTarget,method="rf", nTree=phishFTree, trControl = phishFCtrl, tuneGrid = phishFGrid)
#phishFPred <- predict(phishFTrain,phishTest2)
#DEFUNCT CODE ABOVE

phishFCM <- confusionMatrix(phishFPred$predictions,phishTestTarget)
phishFCM
#Log the variables importance
phishFImp <- as.data.frame(phishFModel$variable.importance) %>% mutate(variable=row.names(.)) %>% arrange(-phishFModel$variable.importance)
names(phishFImp)[1] <- "importance"
phishFRMSE <- RMSE(as.numeric(as.character(unlist(phishFPred$predictions))),(as.numeric(as.character(unlist(phishTestTarget)))))
#Track the results
phishTrack <- bind_rows(phishTrack, data.frame(Model = "RF", Accuracy=phishFCM$overall["Accuracy"], RMSE=phishFRMSE, stringsAsFactors = FALSE))

#Random Forest########
#Random Forest########
#Random Forest Complete

#KNN########
cat("Starting K-Nearest Neighbors\n")
#KNN########
#K-Nearest Neighbors Start

#Settings for KNN. Crossvalidation. 10 resampling iterations.
phishKCtrl <- trainControl(method="cv",number=10)

#Create our model
#Using tuneGrid as we want to specify values of K to tune. 
phishKFit <- caret::train(Phish ~ ., data=phishTrain, method="knn", trControl=phishKCtrl, tuneGrid=data.frame(k = c(2,3,4,5,7,9)))

#Plot of Number of Neighbors versus Accuracy
#plot(phishKFit, print.thres = 0.5)
#3 is the winner
phishKNN1<-knn(phishTrain,phishTest,phishTrain$Phish,k=3,prob=TRUE)
phishKProb <- as.vector(attr(phishKNN1,"prob"))
#Lets see how close our model got to the data
phishKnn <- predict(phishKFit, newdata=phishTest)
#Build a confusion matrix
phishKCM <- confusionMatrix(phishKnn, phishTest$Phish)
#Mean Squared Error
phishKRMSE <- RMSE(as.numeric(as.character(unlist(phishKnn))),as.numeric(as.character(unlist(phishTest$Phish))))

#Check for overtraining
phishKTKnn <- predict(phishKFit, newdata=phishTest)
phishKTCM <- confusionMatrix(phishKTKnn, phishTest[,32])

#Add result to Tracker
phishTrack <- bind_rows(phishTrack, data.frame(Model = "KNN", Accuracy=phishKCM$overall["Accuracy"], RMSE=phishKRMSE,stringsAsFactors = FALSE))

#KNN########
#KNN########
#K-Nearest Neighbors Complete

#Logistic Regression##
cat("Starting LogR\n")
#Logistic Regression##
#Thanks Francis Galton
#Logistic Regression##

#Binomial technique is at first glance a good measure. 
#Allows prediction of a qualitative response such as phishing in our case
#Use Logistric Regression to model Phish as a function of the predictors

################
#Using binomial model as our response is a 1 or a 0. Phish or no phish.
phishLRFit <- glm(Phish~.,data=phishTrain, family="binomial")
#See how the fit looks
#summary(phishLRFit)
#confint(phishLRFit)
#Predict Probabiity of events "P Hat"
phishLRPhat <- predict(phishLRFit,phishTest, type="response")
#Check for multicolinearity
car::vif(phishLRFit)
#Get model data
phishLRModelData <- augment(phishLRFit) %>% mutate(index= 1:n())
phishLRModelDataOut <- phishLRModelData %>% filter(.std.resid > 3 & .cooksd > 4*mean(.cooksd)) %>% select(.rownames,.std.resid,.cooksd)
#Best fit "Y Hat"
phishLRYhat <- factor(ifelse(phishLRPhat > 0.5,1,0))
#Make a nice confusion matrix, get Accuracy.
phishLRCM <- confusionMatrix(data=phishLRYhat, reference = phishTest$Phish)
#Root Mean Squared Error
phishLRRMSE <- RMSE(as.numeric(as.character(unlist(phishLRYhat))),as.numeric(as.character(unlist(phishTest$Phish))))
#Add to Tracker
phishTrack <- bind_rows(phishTrack, data.frame(Model = "LogR", Accuracy=phishLRCM$overall['Accuracy'],  RMSE=phishLRRMSE, stringsAsFactors = FALSE))

#Errors due to probabilities 0 or 1. Need to use penalized regression.
#Penalized Logistic Regression
#Additional Data Preparation
#Predictor placeholder Variables.
phishPLRx <- model.matrix(Phish~., phishTrain)[,-1]
#Convert class factor to num
phishPLRy <- ifelse(phishTrain$Phish == "1", 1,0)

#Which model is best? We desire to reduce model complexity.
#Starting with lasso and alpha = 1
lambdas <- 10^seq(-3, 5, length.out = 100)
phishPLRLCV <- cv.glmnet(phishPLRx,phishPLRy, alpha=1, family="binomial", lambda = lambdas, nfolds=10)
#plot(phishPLRLasso)
#The plot using lasso shows that the optimal log of lamda is somewhere close to -9.
#This minimizes the cross-validation error.
#Fit the best model on the training data. Min lambda v. simplest lambda 1se.
#Check CV with a plot using without discrete lambda 
phishPLRLFit<- glmnet(phishPLRx,phishPLRy, alpha=1, family="binomial")
#Model with static min lambda
phishPLRL <- glmnet(phishPLRx,phishPLRy, alpha=1, family="binomial", lambda=phishPLRLCV$lambda.min) #min lambda
#Regression Coefficients
coef(phishPLRLCV, phishPLRLCV$lambda.min)
#Predict off the test data.
phishPLRxTest <- model.matrix(Phish~., phishTest)[,-1]
#Get probabilities
phishPLRProb1 <- phishPLRL %>% predict(newx=phishPLRxTest,type="response")
#Convert back to factor
phishPLRPred1 <- as.factor(ifelse(phishPLRProb1> 0.5, 1,0))
#ConfusionMatrix
phishPLRCM1 <- confusionMatrix(phishPLRPred1,phishTest[,32])
#Root Mean Squared Error
phishPLRRMSE <- RMSE(as.numeric(as.character(unlist(phishPLRPred1))),(as.numeric(as.character(unlist(phishTest$Phish)))))
#Add to Tracker
phishTrack <- bind_rows(phishTrack, data.frame(Model = "Lasso-PenalizedLogR", Accuracy=phishPLRCM1$overall["Accuracy"], RMSE=phishPLRRMSE, stringsAsFactors = FALSE))

#Elastic Net and alpha = .5
phishPLRELCV <- cv.glmnet(phishPLRx,phishPLRy, alpha=0.5, family="binomial")
#Check CV with a plot using without discrete lambda 
phishPLRELFit<- glmnet(phishPLRx,phishPLRy, alpha=0.5, family="binomial")
#Model with static min lambda
phishPLREL <- glmnet(phishPLRx,phishPLRy, alpha=0.5, lambda = phishPLRELCV$lambda.min)
#Regression Coefficients
coef(phishPLREL)
#Get probabilities
phishPLRProb2 <- phishPLREL %>% predict(newx=phishPLRxTest, type="response")
#Convert back to factor
phishPLRPred2 <- as.factor(ifelse(phishPLRProb2 > 0.5,1,0))
#ConfusionMatrix
phishPLRCM2 <- confusionMatrix(phishPLRPred2,phishTest[,32])
#Root Mean Squared Error
phishPLR2RMSE <- RMSE(as.numeric(as.character(unlist(phishPLRPred2))),as.numeric(as.character(unlist(phishTest$Phish))))
#Add to Tracker
phishTrack <- bind_rows(phishTrack, data.frame(Model = "ElasticNet-PenalizedLogR", Accuracy=phishPLRCM2$overall["Accuracy"], RMSE=phishPLR2RMSE, stringsAsFactors = FALSE))

#Ridge and alpha = 0
phishPLRRidgeCV <- cv.glmnet(phishPLRx,phishPLRy, alpha=0, family="binomial")
#Check CV with a plot using without discrete lambda 
phishPLRRFit<- glmnet(phishPLRx,phishPLRy, alpha=0.5, family="binomial")
#Model with static min lambda
phishPLRRidge <- glmnet(phishPLRx,phishPLRy, alpha=0, lambda = phishPLRRidgeCV$lambda.min)
#Regression Coefficients
coef(phishPLRRidge)
phishPLRProb3 <- phishPLRRidge %>% predict(newx=phishPLRxTest, type="response")
#convert back to factor
phishPLRPred3 <- as.factor(ifelse(phishPLRProb3 > 0.5, 1,0))
#ConfusionMatrix
phishPLRCM3 <- confusionMatrix(phishPLRPred3,phishTest[,32])
#Root Mean Squared Error
phishPLR3RMSE <- RMSE(as.numeric(as.character(unlist(phishPLRPred3))),as.numeric(as.character(unlist(phishTest$Phish))))
#Add to Tracker
phishTrack <- bind_rows(phishTrack, data.frame(Model = "Ridge-PenalizedLogR", Accuracy=phishPLRCM3$overall["Accuracy"], RMSE=phishPLR3RMSE, stringsAsFactors = FALSE))

#Logistic Regression##
#Logistic Regression##
#Thanks Francis Galton
#Logistic Regression Complete

##ENSEMBLE/Stack START
##
##
cat('Starting Ensemble and Stack Methods\n')
#Because we will be using classProbs=TRUE, we have to make the factors of Phish valid names versus 0,1. Changing to "No","Yes".
phishTrainE <- phishTrain
phishTestE <- phishTest
levels(phishTrainE$Phish) <- list(Yes="1",No="0")
#Update the Test data for our "Yes","No" scheme
levels(phishTestE$Phish) <- list(Yes="1",No="0")
#knn, logR, ranger
phishEAlgs <- c("knn", "ranger","glm")
#Make it faster
registerDoParallel(4)
getDoParWorkers()
#Train again with similar values across algorythms for comparison
phishEStackTrain <- trainControl(method="cv", number=10, savePredictions = 'final', allowParallel = TRUE, classProbs=TRUE)
phishEList <- caretEnsemble::caretList(phishTrainE[-32],phishTrainE$Phish, trControl=phishEStackTrain, methodList=phishEAlgs, continue_on_fail = FALSE)
#Extract just the accuracy values
#phishEListAccuracy <- data.frame(RF=max(phishEList$ranger$results$Accuracy), KNN=max(phishEList$knn$results$Accuracy),GLM=max(phishEList$glm$results$Accuracy),GLMNET=max(phishEList$glmnet$results$Accuracy))
phishEResamples <- resamples(phishEList)
#dotplot(phishEResamples)
#Check for model correlation.
phishECor<- modelCor(phishEResamples)
#Stack algorythms
phishStack <- caretEnsemble::caretStack(phishEList,method="svmPoly",metric="Accuracy",trConrol=phishEStackTrain)
#summary(phishEnsemble1)
#Results from manual algorythm testing
#svmPoly = .982
#multinom = .981
#extratrees = .981
#pda=.981
#parRF=.981
#rFerns = .981
#lda=.981
#pls=.981
#penalized=.981
#LogitBoost=.980
#svmRadial=.95
phishEnsemble1 <- caretEnsemble::caretEnsemble(phishEList,metric="Accuracy",trControl=phishEStackTrain)
#Try out our model on test data
#Get predictions for ensemble and stack methods against test data
phishEPredE1<-predict(phishEnsemble1, newdata=phishTestE)
phishEPredS1<-predict(phishStack, newdata=phishTestE)

#Convert No and Yes back to 0,1 so that we get calculate Confusion Matrix and RMSE
levels(phishEPredE1) <- list("0"="No","1"="Yes")
#RMSE
phishEERMSE <- RMSE(as.numeric(as.character(unlist(phishEPredE1))),as.numeric(as.character(unlist(phishTest$Phish))))
#Confusion Matrix
phishEEA <- confusionMatrix(phishEPredE1,phishTest$Phish)
#Convert No and Yes back to 0,1 so that we get calculate Confusion Matrix and RMSE
levels(phishEPredS1) <- list("0"="No","1"="Yes")
#RMSE
phishESRMSE <- RMSE(as.numeric(as.character(unlist(phishEPredS1))),as.numeric(as.character(unlist(phishTest$Phish))))
#Confusion Matrix
phishESA <- confusionMatrix(phishEPredS1,phishTest$Phish)

#Add Results to Tracker
phishTrack <- bind_rows(phishTrack, data.frame(Model = "Ensemble", Accuracy=phishEEA$overall["Accuracy"], RMSE=phishEERMSE, stringsAsFactors = FALSE))
phishTrack <- bind_rows(phishTrack, data.frame(Model = "Stack", Accuracy=phishESA$overall["Accuracy"], RMSE=phishESRMSE, stringsAsFactors = FALSE)) 

##
##
##ENSEMBLE/Stack END

#Print out Accuracy
print(phishTrack)

cat('Saving Image\n')
#Store the data so that the report Rmd file can pick it up and read it.
save.image(file="Phish-Save.RData")

cat('Complete\n')

#References
#  https://rafalab.github.io/dsbook/
#RF
#  https://rpubs.com/phamdinhkhanh/389752
#  https://uc-r.github.io/random_forests
#  https://www4.stat.ncsu.edu/~post/josh/LASSO_Ridge_Elastic_net_-_Examples.html
#  https://www.datacamp.com/community/tutorials/logistic-regression-R
#  http://rstudio-pubs-static.s3.amazonaws.com/16444_caf85a306d564eb490eebdbaf0072df2.html
#  https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/
#  http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/153-penalized-regression-essentials-ridge-lasso-elastic-net/#ridge-regression
#  Ensemble
#  https://towardsdatascience.com/a-comprehensive-machine-learning-workflow-with-multiple-modelling-using-caret-and-caretensemble-in-fcbf6d80b5f2
#  https://github.com/zachmayer/caretEnsemble/issues/189
#  https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html
#  Feature Plot
#  https://www.machinelearningplus.com/machine-learning/caret-package/
#  glmnet
#  https://rstudio-pubs-static.s3.amazonaws.com/133416_8bc14091dac24831a1ad08c1b1d0e38f.html
#  glm
#  http://www.sthda.com/english/articles/36-classification-methods-essentials/148-logistic-regression-assumptions-and-diagnostics-in-r/