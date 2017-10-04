library(caret)
library(data.table)
library(ggplot2)

### data cleaning: NA interpolate or ignore, outliers
data=rbind(fread("data/train.csv"),fread("data/test.csv"),fill=T)
## NA interpolate
sapply(data,FUN = function(x) length(which(x=="")))
colSums(sapply(data,is.na))
# Embarked
data[Embarked==""]
ggplot(data,aes(x=Pclass,y=Fare,group=Pclass,fill=as.factor(Pclass)))+geom_boxplot()
data[Embarked=="",Embarked:="C"]
# Fare 
data[is.na(Fare),Fare:=data[Pclass==data[is.na(Fare),Pclass]&&Embarked==data[is.na(Fare),Embarked],mean(Fare,na.rm=T)]]
# Age knn
preObj=preProcess(data[,!c("Survived")],method="knnImpute") 
data=predict(preObj,data)
# Age group
getTitle=function(s){
  reg=sapply(s,FUN = function(x) return(gregexpr("\\s\\S+\\.",x)))
  return(sapply(1:length(s),FUN=function(x) substr(s[x],reg[x][[1]]+1,reg[x][[1]]+attr(reg[x][[1]],"match.length")-1)))
}
data[,`:=`(Title=getTitle(Name))]
data[,gpAge:=mean(Age,na.rm=T),by=Title]
data[is.na(Age),Age:=gpAge,]

### exploration: variables choose, correlated variables, create new variables
data[,table(Survived,Pclass,Sex)]
ggplot(data=data[!is.na(Survived)],aes(x=Age,colour=as.factor(Survived)))+geom_density()
data[,summary(Age,na.rm=T),by=Survived]
table(data[(!is.na(Survived))][Age<18,Survived])
table(data[(!is.na(Survived))][Age>=18,Survived])
data[,`:=`(FamilySize=SibSp+Parch,Child=rep(0,.N))]
data[Age<18,Child:=1]
data[,c("Survived","Pclass","SibSp","Parch","FamilySize","Child")]=lapply(data[,c("Survived","Pclass","SibSp","Parch","FamilySize","Child")], function(x) as.factor(x))

###data slicing & preprocess
train=data[!is.na(Survived)]
test=data[is.na(Survived)]
set.seed(1)
inTrain=createDataPartition(y=train[,Survived],p=0.75,list=FALSE)
training=train[inTrain,]
testing=train[-inTrain,]
# preObj=preProcess(training[,!c("Survived")],method="knnImpute")
# predict(preObj,training[,!c("Survived")])

### model fit
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControl <- trainControl(allowParallel = TRUE)
set.seed(1)
system.time({
  modelFit=train(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+FamilySize+Child,
                 method="rf",data=training,trControl = fitControl)
})
stopCluster(cluster)
registerDoSEQ()

### model test
confusionMatrix(testing$Survived,predict(modelFit,testing))

### submit
submit=test[,.(PassengerId)]
submit[,`:=`(Survived=predict(modelFit,test))]
fwrite(submit,"data/submit5.csv")


### model save
# 1 
# preprocess: Age NA mean
# variables: Pclass, Sex, Age, SibSp, Parch, Fare, SibSp+Parch
# method: randomforest
save(modelFit,file="model/model1")

# 2
# preprocess: Age NA knn
# variables: Pclass, Sex, Age, SibSp, Parch, Fare, SibSp+Parch
# method: randomforest
save(modelFit,file="model/model2")

# 3
# preprocess: Age NA knn
# variables: Pclass, Sex, Age, SibSp, Parch, Fare, SibSp+Parch, Embarked
# method: randomforest
save(modelFit,file="model/model3")

# 4
# preprocess: Age NA mean by group of Title, Fare NA in test mean by Embarked & Pclass
# variables: Pclass, Sex, Age, SibSp, Parch, Fare, SibSp+Parch, Embarked
# method: randomforest
save(modelFit,file="model/model4")

# 5
# preprocess: Embarked NA "C", Fare NA in test mean by Embarked & Pclass, Age NA mean by group of Title,
# variables: Pclass, Sex, Age, SibSp, Parch, Fare, SibSp+Parch, Embarked, Child(Age<18)
# method: randomforest
save(modelFit,file="model/model5")

