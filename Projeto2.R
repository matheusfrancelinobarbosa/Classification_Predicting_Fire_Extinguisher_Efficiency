#Package for reading xlsx files
install.packages("readxl")
library(readxl)

library("ggplot2")
library(dplyr)
library(plyr)

#read the data
dados <- read_excel("Acoustic_Extinguisher_Fire_Dataset.xlsx")

#DataFrame dimensions
dim(dados)
#Visualize the Dataframe
View(dados)
#Quantity per Class
table(dados$FUEL)
table(dados$SIZE)
length(dados)
str(dados)

################### Checking for Null Data ##################
not_complete_cases <- sum(!complete.cases(dados))
not_complete_cases

#Dados nulos
nulos <- colwise(function(x){ sum(is.na(x))})
nulos(dados)
#We don't have null data

################## Exploring Numerical Data ###################
dim(dados)

#Colunas numericas
colunas_numericas <- sapply(dados, is.numeric)
colunas_numericas

numerical_data <- dados[colunas_numericas]
numerical_data

# Histogram
labels <- list("Histograma: Size",
               "Histograma: Distance",
               "Histograma: Desibel",
               "Histograma: Airflow",
               "Histograma: Frequency",
               "Histograma: Status")
labels

xAxis <- list("SIZE",
              "DISTANCE",
              "DESIBEL",
              "AIRFLOW",
              "FREQUENCY",
              "STATUS")

xAxis

library(plyr)

#SIZE
ggplot(dados, aes(x=SIZE)) +
  geom_histogram(fill="#E69F00", color="black", bins = 5)+
  geom_vline(aes(xintercept=mean(SIZE)), color="blue",
             linetype="dashed")+
  labs(title="Size histogram plot",x="Size", y = "Count")+
  theme_classic()

#DISTANCE
ggplot(dados, aes(x=DISTANCE)) +
  geom_histogram(fill="#E69F00", color="black", bins = 5)+
  geom_vline(aes(xintercept=mean(DISTANCE)), color="blue",
             linetype="dashed")+
  labs(title="Distance histogram plot",x="Distance", y = "Count")+
  theme_classic()

#DESIBEL
ggplot(dados, aes(x=DESIBEL)) +
  geom_histogram(fill="#E69F00", color="black", bins = 5)+
  geom_vline(aes(xintercept=mean(DESIBEL)), color="blue",
             linetype="dashed")+
  labs(title="Desibel histogram plot",x="Desibel", y = "Count")+
  theme_classic()

#AIRFLOW
ggplot(dados, aes(x=AIRFLOW)) +
  geom_histogram(fill="#E69F00", color="black", bins = 5)+
  geom_vline(aes(xintercept=mean(AIRFLOW)), color="blue",
             linetype="dashed")+
  labs(title="AirFlow histogram plot",x="Airflow", y = "Count")+
  theme_classic()

#FREQUENCY
ggplot(dados, aes(x=FREQUENCY)) +
  geom_histogram(fill="#E69F00", color="black", bins = 5)+
  geom_vline(aes(xintercept=mean(FREQUENCY)), color="blue",
             linetype="dashed")+
  labs(title="Frequency histogram plot",x="Frequency", y = "Count")+
  theme_classic()

colnames(dados)

## CHECKING OUTLIERS WITH BOXPLOTS ##
boxplot(dados$SIZE, main = "Boxplot para os tamanhos dos Extintores", ylab = "Size")
boxplot(dados$DISTANCE, main = "Boxplot para a Distancia", ylab = "Distance (cm)")
boxplot(dados$DESIBEL, main = "Boxplot para o Desibel dos Extintores", ylab = "Desibel")
boxplot(dados$AIRFLOW, main = "Boxplot para o Fluxo de Ar dos Extintores", ylab = "Airfolw")
boxplot(dados$FREQUENCY, main = "Boxplot para a frequencia dos Extintores", ylab = "Preço (R$)")


################## Exploring Categorical Data ###################
table(dados$FUEL)

### Applying One Hot Conding to the FUEL Variable ###
library(caret)

dummy <- dummyVars(" ~ .", data=dados)
df_novo <- data.frame(predict(dummy, newdata = dados)) 

dim(df_novo)
View(df_novo)

table(dados$FUEL)
table(df_novo$FUELgasoline)

#Renaming the Dummy variables created by one hot enconding
MyColumns <- colnames(df_novo)

MyColumns[2] <- "GASOLINE"
MyColumns[3] <- "KEROSE"
MyColumns[4] <- "LPG"
MyColumns[5] <- "THINNER"

colnames(df_novo) <- MyColumns
colnames(df_novo)

###### BALANCING THE CLASSES #######
table(df_novo$STATUS)
#No need for class balancing


################################################# LOGISTIC ##################################################
df_novo

library(readr)
install.packages("tidymodels")
library(tidymodels)
install.packages("glmnet")
library(glmnet)

#The target variable must be of type factor
df_novo$STATUS = as.factor(df_novo$STATUS)

#Separating the data into training and test data
split <- initial_split(df_novo, prop = 0.70, strata = STATUS)
train <- split %>% 
  training()
test <- split %>% 
  testing()

train

#training the model
model <- logistic_reg(mixture = double(1), penalty = double(1)) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(STATUS ~ ., data = train)

# Model summary
tidy(model)

# Class Predictions
pred_class <- predict(model,
                      new_data = test,
                      type = "class")

# Class Probabilities
pred_proba <- predict(model,
                      new_data = test,
                      type = "prob")

results <- test %>%
  select(STATUS) %>%
  bind_cols(pred_class, pred_proba)

#Accuracy
accuracy(results, truth = STATUS, estimate = .pred_class)

### Using TUNE for logistic regression ###

# Define the logistic regression model with penalty and mixture hyperparameters
log_reg <- logistic_reg(mixture = tune(), penalty = tune(), engine = "glmnet")

# Define the grid search for the hyperparameters
grid <- grid_regular(mixture(), penalty(), levels = c(mixture = 4, penalty = 3))

# Define the workflow for the model
log_reg_wf <- workflow() %>%
  add_model(log_reg) %>%
  add_formula(STATUS ~ .)

# Define the resampling method for the grid search
folds <- vfold_cv(train, v = 5)

# Tune the hyperparameters using the grid search
log_reg_tuned <- tune_grid(
  log_reg_wf,
  resamples = folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

select_best(log_reg_tuned, metric = "roc_auc")

# Fit the model using the optimal hyperparameters
log_reg_final <- logistic_reg(penalty = 0.0000000001, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(STATUS~., data = train)

# Evaluate the model performance on the testing set
pred_class <- predict(log_reg_final,
                      new_data = test,
                      type = "class")
results <- test %>%
  select(STATUS) %>%
  bind_cols(pred_class, pred_proba)

# Create confusion matrix
conf_mat(results, truth = STATUS,
         estimate = .pred_class)

#Precision
precision(results, truth = STATUS,
          estimate = .pred_class)

#RECALL
recall(results, truth = STATUS,
       estimate = .pred_class)



################################################# RANDON FOREST ##################################################
library(randomForest)
require(caTools)

summary(df_novo)
sapply(df_novo, class)

#sample = sample.split(df_novo$STATUS, SplitRatio = .75)
#train = subset(df_novo, sample == TRUE)
#test  = subset(df_novo, sample == FALSE)

split <- initial_split(df_novo, prop = 0.70, strata = STATUS)
train <- split %>% 
  training()
test <- split %>% 
  testing()

dim(train)
dim(test)

#Train the model
rf <- randomForest(
  STATUS ~ .,
  data=train,
  ntree = 100, 
  nodesize = 10
)

print(rf)

pred1=predict(rf,type = "prob")
pred1

install.packages('ROCR')
library(ROCR)


confusionMatrix(table(test[,10], pred))
#Acuracia 0.66

############## IMPROVING THE PARAMETERS OF THE RANDOM FOREST ALGORITHM ###################

train
bestmtry <- tuneRF(train,train$STATUS,stepFactor = 1.2, improve = 0.01, trace=T, plot= T)

model <- randomForest(STATUS~.,data= train)

model
importance(model)
varImpPlot(model)

pred_test <- predict(model, newdata = test, type= "class")

pred_test

confusionMatrix(table(pred_test,test$STATUS))
#Acuracia de 0.9677

##################################################################################################################
