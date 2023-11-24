library(ROSE)
library(caret)
library(party)
library(MASS)
library(randomForest)
library(psych)
library(FactoMineR)
library(factoextra)
library(corrplot)
library(lattice)
library(ggplot2)
library(pROC) 
library(kknn)
library(class)
library(e1071)
library(FNN)

##########with PCA###############
###############1. Data Preprocessing###############
#read data
credit<-read.csv('C:/Users/Admin/Desktop/????/BA/Risk Analysis/credit_risk_dataset.csv')
credit<- credit[1:32574,c(1,2,4,5,6,9,10,11,12,13,14,16)]
names(credit)<-c("age","income","home",'emp','intent','grade','amnt','rate','default','percent','cbdefault','len')

#dropna
credit<-credit[-which(credit$rate == ""), ]
credit<-credit[-which(credit$emp == ""), ]

#deal with unbalanced data
table(credit$default) ##check
#transform data type
credit$default<- as.factor(credit$default)
credit$age<- as.factor(credit$age)
credit$income<- as.numeric(credit$income)
credit$home<- as.factor(credit$home)
credit$emp<- as.factor(credit$emp)
credit$intent<- as.factor(credit$intent)
credit$grade<- as.factor(credit$grade)
credit$amnt<- as.numeric(credit$amnt)
credit$rate<- as.numeric(credit$rate)
credit$percent<- as.numeric(credit$percent)
credit$cbdefault<- as.factor(credit$cbdefault)
credit$len<- as.factor(credit$len)
credit<- ROSE(default ~ ., data = credit, seed = 1)$data ##Synthetic Data Generation
table(credit$default)

#PCA
cre<-credit[,-6] ##don't consider grade when doing classification
#transform data type
cre$default<- as.numeric(as.character(credit$default))
cre$age<- as.numeric(cre$age)
cre$home<- as.numeric(cre$home)
cre$emp<- as.numeric(cre$emp)
cre$intent<- as.numeric(cre$intent)
cre$cbdefault<- as.numeric(cre$cbdefault)
cre$len<- as.numeric(cre$len)
cre.scale <- scale(cre[, -8]) ##exclude default
nhl.cor <- cor(cre.scale) ##correlation matrix
corrplot(nhl.cor,tl.col = "black")
#visualization
cre.pca <- PCA(cre.scale, graph = FALSE)
eig.val <- get_eigenvalue(cre.pca)
eig.val ##view eigenvalue, choose 6 PC
fviz_eig(cre.pca, addlabels = TRUE, ylim = c(0, 20))
var <- get_pca_var(cre.pca)
corrplot(var$cos2, is.corr=FALSE,tl.col = "black") #view the composition of PC
#extract 6 PC
pca.rotate <- principal(cre.scale, nfactors = 6, rotate ="varimax")
pca.scores <- data.frame(pca.rotate$scores)
pca.scores$default <- cre$default

###############2.Model Analysis###############
#split the data
train_row<-sample(1:nrow(pca.scores),round(nrow(pca.scores)*0.8))
train<-pca.scores[train_row,]
test<-pca.scores[-train_row,]

###############2.1 Multiple Logistic Regression###############
#fit the model
cre_lr<-glm(default~RC1+RC2+RC3+RC4+RC5+RC6,family=binomial(link='logit'),data=train)
summary(cre_lr)
#MSE
train_pred_lr<-predict(cre_lr,train,type='resp')
test_pred_lr<-predict(cre_lr,test,type='resp')
train_res_lr<-train$default-train_pred_lr
test_res_lr<-test$default-test_pred_lr
print(mean((train_res_lr)^2))
print(mean((test_res_lr)^2))
result<-pca.scores
result$pred_lr<-0
result$pred_lr[train_row]<-train_pred_lr
result$pred_lr[-train_row]<-test_pred_lr
#roc
test_pred_dt<-as.numeric(as.character(test_pred_lr))
roc_lr <- roc(test$default,test_pred_lr)
plot(roc_lr, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main = "Multiple Logistic Regression")

###############2.2 KNN#########################
#fit the model k=15 distance=2
train$default=as.factor(train$default)
test$default=as.factor(test$default)
dt_kknn=kknn(default~.,train,test[,-7],k=15,distance=2)
#predict
pre_kknn=fitted(dt_kknn)
train_pred_knn<-fitted(dt_kknn,train)
test_pred_knn<-fitted(dt_kknn,test)
result$pred_knn<-0
result$pred_knn[train_row]<-as.numeric(as.character(train_pred_knn))
result$pred_knn[-train_row]<-as.numeric(as.character(test_pred_knn))
#confusion matrix
table(test$default,pre_kknn,dnn=c("??ʵֵ","Ԥ??ֵ"))
#roc
kknn_roc=roc(test$default, as.numeric(pre_kknn))
plot(kknn_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1,0.2), grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE, main='K-nearest-neighbourhood')

###############2.3 SVM#########################
#########2.3.1 svm sigmoid###########
#fit the model kernel=sigmoid
train$default = as.factor(train$default)
test$default = as.factor(test$default)
svm_dt<- svm(default~., 
               data = train,
               type = 'C',kernel = 'sigmoid' )
#prediction
pre_svm <- predict(svm_dt,newdata = test)
obs_p_svm = data.frame(prob=pre_svm,obs=test$default)
#confusion matrix
table(test$default,pre_svm,dnn=c("??ʵֵ","Ԥ??ֵ"))
#roc
svm_roc <- roc(test$default,as.numeric(pre_svm))
plot(svm_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='SVM kernel = sigmoid')
############2.3.2 svm radial#############
#fit the model kernel=radial
svm_dt<- svm(default~., 
             data = train,
             type = 'C',kernel = 'radial' )
#prediction
pre_svm <- predict(svm_dt,newdata = test)
obs_p_svm = data.frame(prob=pre_svm,obs=test$default)
train_pred_svm<-predict(svm_dt,train)
test_pred_svm<-predict(svm_dt,test)
result$pred_svm<-0
result$pred_svm[train_row]<-as.numeric(as.character(train_pred_svm))
result$pred_svm[-train_row]<-as.numeric(as.character(test_pred_svm))
#confusion matrix
table(test$default,pre_svm,dnn=c("??ʵֵ","Ԥ??ֵ"))
confusionMatrix(table(train_pred_svm,train$default))
confusionMatrix(table(test_pred_svm,test$default))
#roc
svm_roc <- roc(test$default,as.numeric(pre_svm))
plot(svm_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='SVM kernel = radial')
############2.3.3 svm polynomial#############
#fit the model kernel=radial
svm_dt<- svm(default~., 
             data = train,
             type = 'C',kernel = 'polynomial' )
#prediction
pre_svm <- predict(svm_dt,newdata = test)
obs_p_svm = data.frame(prob=pre_svm,obs=test$default)
#confusion matrix
table(test$default,pre_svm,dnn=c("??ʵֵ","Ԥ??ֵ"))
#roc
svm_roc <- roc(test$default,as.numeric(pre_svm))
plot(svm_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='SVM kernel = polynomial')
############2.3.4 svm linear#############
#fit the model kernel=radial
svm_dt<- svm(default~., 
             data = train,
             type = 'C',kernel = 'linear' )
#prediction
pre_svm <- predict(svm_dt,newdata = test)
obs_p_svm = data.frame(prob=pre_svm,obs=test$default)
#confusion matrix
table(test$default,pre_svm,dnn=c("??ʵֵ","Ԥ??ֵ"))
#roc
svm_roc <- roc(test$default,as.numeric(pre_svm))
plot(svm_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='SVM kernel = linear')


###############2.4 Decision Tree###############
#fit the model
train$default<-as.factor(train$default)
test$default<-as.factor(test$default)
cre_dt<-ctree(default~RC1+RC2+RC3+RC4+RC5+RC6, data = train,controls=ctree_control(maxdepth=5))
plot(cre_dt)
#prediction
train_pred_dt<-predict(cre_dt,train)
test_pred_dt<-predict(cre_dt,test)
result$pred_dt<-0
result$pred_dt[train_row]<-as.numeric(as.character(train_pred_dt))
result$pred_dt[-train_row]<-as.numeric(as.character(test_pred_dt))
#confusion matrix
confusionMatrix(table(train_pred_dt,train$default))
confusionMatrix(table(test_pred_dt,test$default))
#roc
test_pred_dt<-as.numeric(as.character(test_pred_dt))
roc_dt <- roc(test$default,test_pred_dt)
plot(roc_dt, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE, main = "Decision Tree")

###############2.5 Random Forest###############
#fit the model
cre_rf=randomForest(default~., data =train,ntree=100,importance=TRUE)
#importance
cre_rf$importance
varImpPlot(cre_rf, main = "variable importance")
#prediction
train_pred_rf<-predict(cre_rf,train)
test_pred_rf<-predict(cre_rf,test)
result$pred_rf<-0
result$pred_rf[train_row]<-as.numeric(as.character(train_pred_rf))
result$pred_rf[-train_row]<-as.numeric(as.character(test_pred_rf))
#confusion matrix
confusionMatrix(table(train_pred_rf,train$default))
confusionMatrix(table(test_pred_rf,test$default))
#roc
test_pred_rf<-as.numeric(as.character(test_pred_rf))
roc_rf <- roc(test$default,test_pred_rf)
plot(roc_rf, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE, main = "Random Forest")

###############2.6 Model Comparison###############
#sensitivity, specificity, accuracy
per<-data.frame(sensitivity=c(0,0,0,0),specificity=c(0,0,0,0), accuracy=c(0,0,0,0))
row.names(per)<-c("KNN","SVM","DT","RF")
table(test$default,pre_kknn,dnn=c("??ʵֵ","Ԥ??ֵ"))

sensitivity <- function(table,n=2){
  if(!all(dim(table)==c(2,2)))
    stop('Must be a 2??2 table')
  tn=table[2,2]
  fp=table[2,1]
  fn=table[1,2]
  tp=table[1,1]
  sensitivity = tp/(tp+fn)
  return(round(sensitivity,n))
}
specificity <- function(table,n=2){
  if(!all(dim(table)==c(2,2)))
    stop('Must be a 2??2 table')
  tn=table[2,2]
  fp=table[2,1]
  fn=table[1,2]
  tp=table[1,1]
  specificity = tn/(tn+fp)
  return(round(specificity,n))
}
accuracy <- function(table,n=2){
  if(!all(dim(table)==c(2,2)))
    stop('Must be a 2??2 table')
  tn=table[2,2]
  fp=table[2,1]
  fn=table[1,2]
  tp=table[1,1]
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  return(round(accuracy,n))
}
per[1,1]<-sensitivity(table(test$default,pre_kknn,dnn=c("??ʵֵ","Ԥ??ֵ")))
per[2,1]<-sensitivity(table(test_pred_svm,test$default))
per[3,1]<-sensitivity(table(test_pred_dt,test$default))
per[4,1]<-sensitivity(table(test_pred_rf,test$default))
per[1,2]<-specificity(table(test$default,pre_kknn,dnn=c("??ʵֵ","Ԥ??ֵ")))
per[2,2]<-specificity(table(test_pred_svm,test$default))
per[3,2]<-specificity(table(test_pred_dt,test$default))
per[4,2]<-specificity(table(test_pred_rf,test$default))
per[1,3]<-accuracy(table(test$default,pre_kknn,dnn=c("??ʵֵ","Ԥ??ֵ")))
per[2,3]<-accuracy(table(test_pred_svm,test$default))
per[3,3]<-accuracy(table(test_pred_dt,test$default))
per[4,3]<-accuracy(table(test_pred_rf,test$default))
#plot
plot(c(1,2,3,4),per[ ,1],pch=15,col='cornflowerblue',axes=FALSE,ylim=c(0.7,0.9),xlab='Model',ylab='Performance')
points(c(1,2,3,4),per[ ,2],pch=16,col='lightcoral',axes=FALSE,ylim=c(0.7,0.9))
points(c(1,2,3,4),per[ ,3],pch=17,col='darkseagreen',axes=FALSE,ylim=c(0.7,0.9))
box()
axis(1,at=c(1,2,3,4),labels=c('KNN','SVM','DT','RF'))
axis(2)
legend("topleft",legend=c("sensitivity","specificity","accuracy"),col=c("cornflowerblue","lightcoral","darkseagreen"),pch=c(15,16,17),lty=1,cex=0.7)

#roc$auc
plot(roc_lr,  auc.polygon=TRUE, max.auc.polygon=TRUE,col="darkseagreen")
lines(roc_dt, auc.polygon=TRUE, max.auc.polygon=TRUE,col="cornflowerblue")
lines(roc_rf, auc.polygon=TRUE, max.auc.polygon=TRUE,col="lightslateblue")
lines(kknn_roc, auc.polygon=TRUE, max.auc.polygon=TRUE,col="lightcoral")
lines(svm_roc, auc.polygon=TRUE, max.auc.polygon=TRUE,col="gold")
legend("bottomright",legend=c("Logistic Regression","KNN","SVM","Decision Tree","Random Forest"),col=c("darkseagreen","lightcoral","gold","cornflowerblue","lightslateblue"),lty=1,cex=0.7,bty="n")

###############3.without PCA###############
#split the data
credit<-credit[,-6]
credit$default<- as.numeric(as.character(credit$default))
credit$age<- as.numeric(credit$age)
credit$home<- as.numeric(credit$home)
credit$emp<- as.numeric(credit$emp)
credit$intent<- as.numeric(credit$intent)
credit$cbdefault<- as.numeric(credit$cbdefault)
credit$len<- as.numeric(credit$len)
credit$grade<- as.numeric(credit$grade)
train_row2<-sample(1:nrow(credit),round(nrow(credit)*0.8))
train2<-credit[train_row2,]
test2<-credit[-train_row2,]

###############3.1 Multiple Logistic Regression###############
#fit the model
cre_lr2<-glm(default~.,family=binomial(link='logit'),data=train2)
summary(cre_lr2)
#MSE
train_pred_lr2<-predict(cre_lr2,train2,type='resp')
test_pred_lr2<-predict(cre_lr2,test2,type='resp')
train_res_lr2<-train2$default-train_pred_lr2
test_res_lr2<-test2$default-test_pred_lr2
print(mean((train_res_lr2)^2))
print(mean((test_res_lr2)^2))
result2<-credit
result2$pred_lr<-0
result2$pred_lr[train_row2]<-train_pred_lr2
result2$pred_lr[-train_row2]<-test_pred_lr2
#roc
test_pred_dt2<-as.numeric(as.character(test_pred_lr2))
roc_lr2 <- roc(test2$default,test_pred_lr2)
plot(roc_lr2, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main = "Multiple Logistic Regression")

###############3.2 KNN#########################
#fit the model k=15 distance=2
train2$default=as.factor(train2$default)
test2$default=as.factor(test2$default)
dt_kknn2=kknn(default~.,train2,test2[,-8],k=15,distance=2)
#predict
pre_kknn2=fitted(dt_kknn2)
train_pred_knn2<-fitted(dt_kknn2,train2)
test_pred_knn2<-fitted(dt_kknn2,test2)
result2$pred_knn<-0
result2$pred_knn[train_row2]<-as.numeric(as.character(train_pred_knn2))
result2$pred_knn[-train_row2]<-as.numeric(as.character(test_pred_knn2))
#confusion matrix
table(test2$default,pre_kknn2,dnn=c("??ʵֵ","Ԥ??ֵ"))
#roc
kknn_roc2=roc(test2$default, as.numeric(pre_kknn2))
plot(kknn_roc2, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1,0.2), grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE, main='K-nearest-neighbourhood')

###############3.3 SVM#########################
#fit the model kernel=radial
svm_dt2<- svm(default~., 
             data = train2,
             type = 'C',kernel = 'radial' )
#prediction
pre_svm2 <- predict(svm_dt2,newdata = test2)
obs_p_svm2 = data.frame(prob=pre_svm2,obs=test2$default)
train_pred_svm2<-predict(svm_dt2,train2)
test_pred_svm2<-predict(svm_dt2,test2)
result2$pred_svm<-0
result2$pred_svm[train_row2]<-as.numeric(as.character(train_pred_svm2))
result2$pred_svm[-train_row2]<-as.numeric(as.character(test_pred_svm2))
#confusion matrix
table(test2$default,pre_svm2,dnn=c("??ʵֵ","Ԥ??ֵ"))
confusionMatrix(table(train_pred_svm2,train2$default))
confusionMatrix(table(test_pred_svm2,test2$default))
#roc
svm_roc2 <- roc(test2$default,as.numeric(pre_svm2))
plot(svm_roc2, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='SVM kernel = radial')

###############3.4 Decision Tree###############
#fit the model
train2$default<-as.factor(train2$default)
test2$default<-as.factor(test2$default)
cre_dt2<-ctree(default~., data = train2,controls=ctree_control(maxdepth=5))
plot(cre_dt2)
#prediction
train_pred_dt2<-predict(cre_dt2,train2)
test_pred_dt2<-predict(cre_dt2,test2)
result2$pred_dt<-0
result2$pred_dt[train_row2]<-as.numeric(as.character(train_pred_dt2))
result2$pred_dt[-train_row2]<-as.numeric(as.character(test_pred_dt2))
#confusion matrix
confusionMatrix(table(train_pred_dt2,train2$default))
confusionMatrix(table(test_pred_dt2,test2$default))
#roc
test_pred_dt2<-as.numeric(as.character(test_pred_dt2))
roc_dt2 <- roc(test2$default,test_pred_dt2)
plot(roc_dt2, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE, main = "Decision Tree")

###############3.5 Random Forest###############
#fit the model
cre_rf2=randomForest(default~., data =train2,ntree=100,importance=TRUE)
#importance
cre_rf2$importance
varImpPlot(cre_rf2, main = "variable importance")
#prediction
train_pred_rf2<-predict(cre_rf2,train2)
test_pred_rf2<-predict(cre_rf2,test2)
result2$pred_rf<-0
result2$pred_rf[train_row2]<-as.numeric(as.character(train_pred_rf2))
result2$pred_rf[-train_row2]<-as.numeric(as.character(test_pred_rf2))
#confusion matrix
confusionMatrix(table(train_pred_rf2,train2$default))
confusionMatrix(table(test_pred_rf2,test2$default))
#roc
test_pred_rf2<-as.numeric(as.character(test_pred_rf2))
roc_rf2 <- roc(test2$default,test_pred_rf2)
plot(roc_rf2, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE, main = "Random Forest")

###############3.6 Model Comparison###############
#sensitivity, specificity, accuracy
per2<-data.frame(sensitivity=c(0,0,0,0),specificity=c(0,0,0,0), accuracy=c(0,0,0,0))
row.names(per2)<-c("KNN","SVM","DT","RF")

per2[1,1]<-sensitivity(table(test2$default,pre_kknn2,dnn=c("??ʵֵ","Ԥ??ֵ")))
per2[2,1]<-sensitivity(table(test_pred_svm2,test2$default))
per2[3,1]<-sensitivity(table(test_pred_dt2,test2$default))
per2[4,1]<-sensitivity(table(test_pred_rf2,test2$default))
per2[1,2]<-specificity(table(test2$default,pre_kknn2,dnn=c("??ʵֵ","Ԥ??ֵ")))
per2[2,2]<-specificity(table(test_pred_svm2,test2$default))
per2[3,2]<-specificity(table(test_pred_dt2,test2$default))
per2[4,2]<-specificity(table(test_pred_rf2,test2$default))
per2[1,3]<-accuracy(table(test2$default,pre_kknn2,dnn=c("??ʵֵ","Ԥ??ֵ")))
per2[2,3]<-accuracy(table(test_pred_svm2,test2$default))
per2[3,3]<-accuracy(table(test_pred_dt2,test2$default))
per2[4,3]<-accuracy(table(test_pred_rf2,test2$default))
#plot
plot(c(1,2,3,4),per2[ ,1],pch=15,col='cornflowerblue',axes=FALSE,ylim=c(0.75,0.95),xlab='Model',ylab='Performance')
points(c(1,2,3,4),per2[ ,2],pch=16,col='lightcoral',axes=FALSE,ylim=c(0.75,0.95))
points(c(1,2,3,4),per2[ ,3],pch=17,col='darkseagreen',axes=FALSE,ylim=c(0.75,0.95))
box()
axis(1,at=c(1,2,3,4),labels=c('KNN','SVM','DT','RF'))
axis(2)
legend("topleft",legend=c("sensitivity","specificity","accuracy"),col=c("cornflowerblue","lightcoral","darkseagreen"),pch=c(15,16,17),lty=1,cex=0.7)

#roc$auc
plot(roc_lr2,  auc.polygon=TRUE, max.auc.polygon=TRUE,col="darkseagreen")
lines(roc_dt2, auc.polygon=TRUE, max.auc.polygon=TRUE,col="cornflowerblue")
lines(roc_rf2, auc.polygon=TRUE, max.auc.polygon=TRUE,col="lightslateblue")
lines(kknn_roc2, auc.polygon=TRUE, max.auc.polygon=TRUE,col="lightcoral")
lines(svm_roc2, auc.polygon=TRUE, max.auc.polygon=TRUE,col="gold")
legend("bottomright",legend=c("Logistic Regression","KNN","SVM","Decision Tree","Random Forest"),col=c("darkseagreen","lightcoral","gold","cornflowerblue","lightslateblue"),lty=1,cex=0.7,bty="n")

###############4.Model Combination###############
##logistic
train$label_rf<-train$default-train_pred_lr
test$label_rf<-test$default-test_pred_lr
##rf
rf<-randomForest(label_rf~RC1+RC2+RC3+RC4+RC5+RC6,data=train,importance=TRUE, ntree=100)
train$label_knn<-train$label_rf-predict(rf,train)
test$label_knn<-test$label_rf-predict(rf,test)
##knn
#fit the model k=15 distance=2
knn = knn.reg(train = train[,-c(7,8)], test = test[,-c(7,8)], y=train$label_knn,k = 15)
test$combination<-knn$pred+predict(rf,test)+test_pred_lr
#mse
test_res_com<-test$default-test$combination
print(mean((test_res_com)^2))
#0-1
test$default_com<-0
for (i in 1:nrow(test)){
  if (test[i,10]>0.5){test[i,11]=1}
  else {test[i,11]=0}
}
#confusion matrix
confusionMatrix(table(test$default_com,test$default))
#roc
combination_roc=roc(test$default, test$combination)
plot(combination_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1,0.2), grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE, main='Model Combination_Boosting')
#sensitivity, accuracy, specificity
sensitivity(table(test$default_com,test$default))
accuracy(table(test$default_com,test$default))
specificity(table(test$default_com,test$default))