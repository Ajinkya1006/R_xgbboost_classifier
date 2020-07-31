#reading training and testing data
train_data_lp2 <- read.csv("D:/predective_analysis/project/project2/train_ctrUa4K.csv",na.strings = c("",NA))
test_data_lp2 <- read.csv("D:/predective_analysis/project/project2/test_lAUu6dG.csv",na.strings = c("",NA))

#analysisng data
str(train_data_lp2)
head(train_data_lp2)
View(train_data_lp2)
summary(train_data_lp2)
#checking for NA values
colSums(is.na(train_data_lp2))
colSums(is.na(test_data_lp2))


#Explorative data analysis
table(train_data_lp2$Gender,train_data_lp2$Loan_Status)
table(train_data_lp2$Married,train_data_lp2$Loan_Status)
table(train_data_lp2$Education,train_data_lp2$Loan_Status)
table(train_data_lp2$Self_Employed,train_data_lp2$Loan_Status)
table(train_data_lp2$Credit_History,train_data_lp2$Loan_Status)

target <- table(train_data_lp2$Loan_Status)
lbls <- paste(names(target),"\n",target, sep="")
pie(target,labels = lbls,main = "Loan status")
library(ggplot2)
ggplot(train_data_lp2,aes(x=Gender,fill=Loan_Status))+geom_bar(position = "stack")
ggplot(train_data_lp2,aes(x=Married,fill=Loan_Status))+geom_bar(position = "stack")
ggplot(train_data_lp2,aes(x=Education,fill=Loan_Status))+geom_bar(position = "stack")
ggplot(train_data_lp2,aes(x=Self_Employed,fill=Loan_Status))+geom_bar(position = "stack")
ggplot(train_data_lp2,aes(x=Credit_History,fill=Loan_Status))+geom_bar(position = "stack")



#using mice method to predict the NA values
train_data_mice <- as.data.frame(train_data_lp2[ ,-1])
library(mice)
library(lattice)
md.pattern(train_data_mice)
impute_data <- mice(train_data_mice,m=2,maxit = 4,method = 'pmm')
impute_gender <- as.data.frame(impute_data$imp$Gender)
impute_married <- as.data.frame(impute_data$imp$Married)
impute_dependents <- as.data.frame(impute_data$imp$Dependents)
impute_self_employed <- as.data.frame(impute_data$imp$Self_Employed)
impute_loanAmount <- as.data.frame(impute_data$imp$LoanAmount)
impute_loanTerm <- as.data.frame(impute_data$imp$Loan_Amount_Term)
impute_creditHistory <- as.data.frame(impute_data$imp$Credit_History)


#updating variables for training dataset
train_data_lp2[which(is.na(train_data_lp2$Gender)),]$Gender <- impute_gender$`1`
train_data_lp2[which(is.na(train_data_lp2$Married)),]$Married <- impute_married$`1`
train_data_lp2[which(is.na(train_data_lp2$Dependents)),]$Dependents <- impute_dependents$`1`
train_data_lp2[which(is.na(train_data_lp2$Self_Employed)),]$Self_Employed <- impute_self_employed$`1`
train_data_lp2[which(is.na(train_data_lp2$LoanAmount)),]$LoanAmount <- impute_loanAmount$`1`
train_data_lp2[which(is.na(train_data_lp2$Loan_Amount_Term)),]$Loan_Amount_Term <- impute_loanTerm$`1`
train_data_lp2[which(is.na(train_data_lp2$Credit_History)),]$Credit_History <- impute_creditHistory$`1`


colSums(is.na(train_data_lp2))


#imputing missing values for test data
test_data_mice <- as.data.frame(test_data_lp2[ ,-1])
md.pattern(test_data_mice)
impute_data <- mice(test_data_mice,m=2,maxit = 4,method = 'pmm')
impute_gender <- as.data.frame(impute_data$imp$Gender)
impute_married <- as.data.frame(impute_data$imp$Married)
impute_dependents <- as.data.frame(impute_data$imp$Dependents)
impute_self_employed <- as.data.frame(impute_data$imp$Self_Employed)
impute_loanAmount <- as.data.frame(impute_data$imp$LoanAmount)
impute_loanTerm <- as.data.frame(impute_data$imp$Loan_Amount_Term)
impute_creditHistory <- as.data.frame(impute_data$imp$Credit_History)


#updating variables for testing dataset
test_data_lp2[which(is.na(test_data_lp2$Gender)),]$Gender <- impute_gender$`1`
test_data_lp2[which(is.na(test_data_lp2$Dependents)),]$Dependents <- impute_dependents$`1`
test_data_lp2[which(is.na(test_data_lp2$Self_Employed)),]$Self_Employed <- impute_self_employed$`1`
test_data_lp2[which(is.na(test_data_lp2$LoanAmount)),]$LoanAmount <- impute_loanAmount$`1`
test_data_lp2[which(is.na(test_data_lp2$Loan_Amount_Term)),]$Loan_Amount_Term <- impute_loanTerm$`1`
test_data_lp2[which(is.na(test_data_lp2$Credit_History)),]$Credit_History <- impute_creditHistory$`1`




##Building simple logistic regression model
ap_predict <- glm(Loan_Status~Gender+Married+Dependents+Education+Self_Employed+LoanAmount+Loan_Amount_Term+Credit_History+Property_Area,data = train_data_lp2,family = binomial)
vp_predict <- predict(ap_predict,type = "response")
summary(ap_predict)
table(train_data_lp2$Loan_Status,vp_predict>0.6)
table(train_data_lp2$Loan_Status)

library(ROCR)
ROCPred <- prediction(vp_predict,train_data_lp2$Loan_Status)
ROCPref <- performance(ROCPred,"tpr","fpr")

#plot ROC curve
plot(ROCPref,colorize=TRUE)
#ADD THRESHOLD LEVELS
plot(ROCPref,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1),tex.adj=c(-0.2,0.7))
loan_predict <- predict(ap_predict,type="response",newdata=test_data_lp2)
loan_predict <- as.data.frame(loan_predict)


#using xgboost for model building
train_trail_num <- model.matrix(~.-1,data=train_data_lp2[ ,2:12])
dim(train_trail_num)
y <- train_data_lp2[ ,13]
y <- as.character(y)
y[which(is.na(y))] <- 0
y[which(y=="Y")] <- 1
y <- as.numeric(y)
library(xgboost)
param <- list(booster="gbtree",eta=0.3,objective="binary:logistic")
help("xgboost")
xgb_cross <- xgboost(params=param,data = train_trail_num,label = y,nround=500,nfold =3)
xgb_predict <- predict(xgb_cross,type="response",newdata = train_trail_num)
xgb_predict <- as.data.frame(xgb_predict)
View(xgb_predict)
table(train_data_lp2$Loan_Status,xgb_predict>0.5)

test_trail_num <- model.matrix(~.-1,data=test_data_lp2[ ,-1])
xgb_test_predict <- predict(xgb_cross,type="response",newdata = test_trail_num)
xgb_test_predict <- as.data.frame(xgb_test_predict)
View(xgb_test_predict)
xgb_test_predict <- as.character(xgb_test_predict)
xgb_test_predict[which(xgb_test_predict==1),] <- "Y"
xgb_test_predict[which(xgb_test_predict==0),] <- "N"
test_data_lp2 <- cbind(test_data_lp2,xgb_test_predict)
names(test_data_lp2)[13] <- "Loan_Status"


LOAN_PREDICTION <- as.data.frame(test_data_lp2[,c(1,13)])
View(LOAN_PREDICTION)
write.csv(LOAN_PREDICTION,"LOAN_PREDICTION.csv")
