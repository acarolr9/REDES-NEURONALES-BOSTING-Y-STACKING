#setwd('C:\Users\ASUS\OneDrive - Pontificia Universidad Javeriana\POSTGRADO\(2) Segundo Semestre\Analitica II\Caso 1')
setwd('C:/Users/ASUS/OneDrive - Pontificia Universidad Javeriana/POSTGRADO/(2) Segundo Semestre/Analitica II/Caso 1')

datos <- read.csv("BaseCompleta.csv",sep = ",")

datos$C.MANIPULATOR <- as.factor(datos$C.MANIPULATOR)
str(datos)

#################################################################################################
#EXPLORACIÓN DE DATOS
#################################################################################################


#Limpieza de Nulos
sapply(datos, function(x) sum(is.na(x)))

datos_num <- datos[,2:9]
#ESTADISDICOS 
library(psych)
CV <- function(var){(sd(var)/mean(var))*100}
cvs<-apply(datos_num,2,CV)
numdescriptive<-describe(datos_num)
numdescriptive<-cbind(numdescriptive,cvs)
numdescriptive

#TABLA DE CORRELACIÓN 
library("corrplot")
base<-cor(datos_num)
corrplot(base, type="upper")

#################################################################################################
#LIMPIEZA Y SELECCION DE DATOS
#################################################################################################

#ACCR
datos_modelo <- as.data.frame(datos[datos$ï..Company.ID != 504,])
#GMI
datos_modelo <- as.data.frame(datos_modelo[datos_modelo$ï..Company.ID != 26,])
datos_modelo <- as.data.frame(datos_modelo[datos_modelo$ï..Company.ID != 211,])


datos_modelo$Manipulater<-NULL
datos_modelo$ï..Company.ID <-NULL

#################################################################################################
#TRANSFORMACION DE DATOS
#################################################################################################
#Aplicación de log
datos_trans <- as.data.frame(datos_modelo[,4:6])
datos_trans <- cbind(datos_modelo$DSRI,datos_trans,datos_modelo$LEVI)
colnames(datos_trans)<- c('DSRI',	'SGI',	'DEPI',	'SGAI','LEVI')
numdescriptive<-describe(datos_trans)

datos_trans <- as.data.frame(sapply(datos_trans, function(x) log(x - min(x) + 1)))
numdescriptive1<-describe(datos_trans)

datos_finales <- as.data.frame( cbind(datos_trans,datos_modelo$ACCR,datos_modelo$GMI,datos_modelo$AQI))
colnames(datos_finales)<- c('DSRI',	'SGI',	'DEPI',	'SGAI','LEVI','ACCR', 'GMI' , 'AQI')
numdescriptiveF<-describe(datos_finales)

datos_finales <- as.data.frame(cbind(datos_finales,datos_modelo$C.MANIPULATOR))
colnames(datos_finales)<- c('DSRI',	'SGI',	'DEPI',	'SGAI','LEVI','ACCR', 'GMI' , 'AQI','MANIPULATOR')
str(datos_finales)

#################################################################################################
#NORMALIZAR
#################################################################################################
prop.table(table(datos$C.MANIPULATOR))

#Normalizar 
normalizar <- function(w) {
  return((w-min(w))/(max(w)-min(w)))
}
datos_norm <- as.data.frame(lapply(datos_finales[,1:8], normalizar))
datos_norm <- cbind(datos_norm,datos_finales$MANIPULATOR)

colnames(datos_norm) <- c('DSRI',	'SGI',	'DEPI',	'SGAI','LEVI','ACCR', 'GMI' , 'AQI','MANIPULATOR')

####################################################################################
#GENERA 220 Y DIVIDE EN ENTRENAMIENTO Y PRUEBA
####################################################################################

library('dplyr')

datos_man<-filter(datos_norm, MANIPULATOR == "1")
datos_no_mani<-filter(datos_norm, MANIPULATOR == "0")

a<-sample(1:1198,182,replace=F)
datos_no_mani<-datos_no_mani[a,]

datos_norm_p<-rbind(datos_man,datos_no_mani)

prop.table(table(datos_norm_p$MANIPULATOR))

set.seed(15000)
sample <- sample.int(nrow(datos_norm_p), floor(.7*nrow(datos_norm_p)))
modelo.train <- datos_norm_p[sample, ]
modelo.test <- datos_norm_p[-sample, ]

prop.table(table(modelo.train$MANIPULATOR))
prop.table(table(modelo.test$MANIPULATOR))

#################################################################################################
#REDES NEURONALES
#################################################################################################

library(neuralnet)

#MODELO REDES NEURONALES
set.seed(150)
Mod2 <- neuralnet(MANIPULATOR ~ ., data = modelo.train,
                  hidden= 3,
                  act.fct = "logistic", lifesign = "full",
                  stepmax = 8000000)

plot(Mod2, rep="best")

######## PREDECIR ENTRENAMIENTO                   

Mod1.pred <- predict(Mod2, newdata=modelo.train[,-9])
res.train <- ifelse(Mod1.pred[,1] > Mod1.pred[,2],0,1)
res.train<-as.factor(res.train)

# Matriz de confusiÃ³n
c<-confusionMatrix(res.train,
                modelo.train$MANIPULATOR, positive = "1")
c
#accuracy-sensitivity-specificity-f1

dat<-cbind(c$overall[1],c$byClass[1],c$byClass[2],c$byClass[7])

Ind.train<-rbind(Ind.train,dat)

# Curva ROC
pr<-prediction(as.numeric(res.train),as.numeric(modelo.train$MANIPULATOR))

curvaROC<-performance(pr,measure="tpr",x.measure="fpr")

plot(curvaROC)
auc1 = performance(pr, "auc")
auc1.area <- as.numeric(auc1@y.values)

Roc.train<-rbind(Roc.train,auc1.area)

######## PREDECIR PRUEBA                         


Mod1.pred.p <- predict(Mod2, newdata=modelo.test[-9])
res.test <- ifelse(Mod1.pred.p[,1] > Mod1.pred.p[,2],0,1)
res.test<-as.factor(res.test)

# Matriz de confusiÃ³n
c<-confusionMatrix(res.test,
                modelo.test$MANIPULATOR, positive = "1")
c
#accuracy-sensitivity-specificity-f1

dat<-cbind(c$overall[1],c$byClass[1],c$byClass[2],c$byClass[7])

Ind.test<-rbind(Ind.test,dat)

# Curva ROC
pr<-prediction(as.numeric(res.test),as.numeric(modelo.test$MANIPULATOR))

curvaROC<-performance(pr,measure="tpr",x.measure="fpr")

plot(curvaROC)
auc1 = performance(pr, "auc")
auc1.area <- as.numeric(auc1@y.values)

Roc.test<-rbind(Roc.test,auc1.area)


#################################################################################################
#BOOSTING
#################################################################################################


library(gbm)
library(DiagrammeR)

training.x <- model.matrix(MANIPULATOR~., data=modelo.train)
testing.x <- model.matrix(MANIPULATOR~., data=modelo.test)

set.seed(100)
Mod1 <- xgboost(data = data.matrix(training.x[,-1]),
                         label = as.numeric(as.character(modelo.train$MANIPULATOR)),
                         eta=0.1,
                         max_deph = 10,
                         min_child_weight =5,
                         nround =  120,
                         verbose = 0,
                         objective = "binary:logistic")


importance_matrix <- xgb.importance( model = Mod1)
xgb.ggplot.importance(importance_matrix)
xgb.plot.deepness(model = Mod1)  
xgb.plot.multi.trees(model = Mod1,  features.keep = 3)

####RESULTADOS PRUEBA

pred <- predict(Mod1, newdata = testing.x[,-1],
                type="response")

#Matrix de confusión
c<-confusionMatrix(as.factor(ifelse(pred > 0.5, 1,0)),
                   modelo.test$MANIPULATOR, positive = "1")

c
#accuracy-sensitivity-specificity-f1

dat<-cbind(c$overall[1],c$byClass[1],c$byClass[2],c$byClass[7])
dat


Ind.test<-rbind(Ind.test,dat)

# Curva ROC
pred.tr <- prediction(pred, modelo.test$MANIPULATOR)
pred.roc <- performance(pred.tr, "tpr","fpr")
plot(pred.roc)

# AUC
AUC.tmp <- performance(pred.tr, "auc")
ROC <- as.numeric(AUC.tmp@y.values)

Roc.test<-rbind(Roc.test,ROC)


####RESULTADOS ENTRENAMIENTO

pred <- predict(Mod1, newdata = training.x[,-1],
                              type="response")

#Matrix de confusión
c<-confusionMatrix(as.factor(ifelse(pred > 0.5, 1,0)),
                modelo.train$MANIPULATOR, positive = "1")

c
#accuracy-sensitivity-specificity-f1

dat<-cbind(c$overall[1],c$byClass[1],c$byClass[2],c$byClass[7])
Ind.train<-rbind(Ind.train,dat)

# Curva ROC
pred.tr <- prediction(pred, modelo.train$MANIPULATOR)
pred.roc <- performance(pred.tr, "tpr","fpr")
plot(pred.roc)

# AUC
AUC.tmp <- performance(pred.tr, "auc")
ROC <- as.numeric(AUC.tmp@y.values)

Roc.train<-rbind(Roc.train,ROC)



write.xlsx(Roc.train, "C:/Users/ASUS/Documents/Roc.train.xlsx")
write.xlsx(Ind.train, "C:/Users/ASUS/Documents/Ind.train.xlsx")

write.xlsx(Roc.test, "C:/Users/ASUS/Documents/Roc.test.xlsx")
write.xlsx(Ind.test, "C:/Users/ASUS/Documents/Ind.test.xlsx")




