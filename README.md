# Using Random forest and Gradient boosting trees to improve wave forecast at a specific location

## Callens Aurélien, Denis Morichon, Stéphane Abadie, Matthias Delpey, Benoit Liquet

## 29/06/2020

----------------

### Introduction

This suplementary material demonstrates how to use the data assimilation presented in the article: “Using Random forest and Gradient boosting trees to improve wave forecast at a specific location”. Unlike the article, the data assimilation presented in this post is performed on forecast wave data used in operational applications.

The data assimilation consists in three steps :

1. Compute the deviations (Observed values at the buoy - Modeled values from wave model)

2. Train a machine learning algorithm to model the deviations with explanatory variables (here the modeled values from the model)

3. Add the deviations predicted by the machine learning algorithm to the forecasts made by the numerical model

Available data:

* The wave forecast data for our study site. This dataset called “wave_forecast_ibi” is available on Copernicus website (Link here. It regroups the predictions of the wave model for the three parameters of interests: the significant wave height (Hs), the peak period (Tp) of the wave and the peak direction (θp). This dataset regroups hourly predictions during 3 years (2017-2020).

* The observed data that will be used to compute the deviations of the model that we want to predict with machine learning. The hourly observed data for the same period as the forecast are available for our study site. They can obtain free of charge by contacting the Centre for Studies and Expertise on Risks, Environment, Mobility, and Urban and Country Planning (CEREMA).

In the article we also used data about meteorological conditions. However, due to the privacy of the data provided by MeteoFrance we can’t make them available on github.


### Data assimilation with R
The first step if to load the libraries and import a custom function which compute the desired metrics to compare the different machine learning algorithms.

```r
# Load the libraries : 

library(tidyverse)
library(data.table)
library(magrittr)
library(caret)
library(ranger)
library(xgboost)
library(ranger)
library(keras)

# Import a homemade function to calculate the desired metrics 
source("metric_function.R")
```

### Importing and spliting the dataset
Next, we import the dataset Biarritz_data.csv. This file contains the predictions of the wave model and the observed values at the buoy. We separate the data in two sets, one for training and one for testing.

```r
# Importation 
biarritz_data <- read.csv("Biarritz_data.csv")


# Create training and testing dataset 85% vs 15%
set.seed(10)
samp <- sample(1:nrow(biarritz_data), nrow(biarritz_data)*0.15)

training <- biarritz_data[-samp,]
test <- biarritz_data[samp,]
```

### Fitting the models (default hyperparameter values) and computing the metrics

#### Wave model
We can already calculate the metrics for the wave model :

```r
num <- sapply(c("Hs", "Tp", "Dir"), 
              function(x){
                metrics_wave(obs = test[ , paste0(x,"_b")],
                             pred = test[ , paste0(x,"_m")],
                             digits = 3,
                             var = x)
              })

row.names(num) <- c("Biais", "Rmse", "SI", "Cor")

num3 <- sapply(c("Hs", "Tp", "Dir"), 
               function(x){
                 metrics_wave(obs = test[which(test$Hs_b > 3) , paste0(x,"_b")],
                              pred = test[which(test$Hs_b > 3) , paste0(x,"_m")],
                              digits = 3,
                             var = x)
               })

row.names(num3) <- c("Biais", "Rmse", "SI", "Cor")
```

#### Train algorithms with default hyperparameter values

##### Random forest

The first step is to fit the random forest with the training data :

```r
rf_hs <- ranger(Hs_diff ~  Hs_m + Tp_m + Dir_m ,
                data = training, importance = 'impurity')

rf_tp <- ranger(Tp_diff ~  Hs_m + Tp_m + Dir_m,
                data = training, importance = 'impurity')

rf_dir <- ranger(Dir_diff ~  Hs_m + Tp_m + Dir_m,
                 data = training, importance = 'impurity')
```

Then we can calculate the metrics for our two case (all data and Hs > 3m) :

```r
test %<>% mutate(Hs_c = round(Hs_m + predict(rf_hs, test)$predictions, digits = 2),
                      Tp_c = round(Tp_m + predict(rf_tp, test)$predictions, digits = 2),
                      Dir_c = round((Dir_m + predict(rf_dir, test)$predictions) %% 360, digits = 2))

rf <- sapply( c("Hs", "Tp", "Dir"), 
                 function(x){
                   metrics_wave(obs = test[ , paste0(x,"_b")],
                                pred = test[ , paste0(x,"_c")],
                                digits = 3,
                             var = x)})

row.names(rf) <- c("Biais", "Rmse", "SI", "Cor")



rf3 <- sapply(c("Hs", "Tp", "Dir"), 
              function(x){
                metrics_wave(obs = test[which(test$Hs_b > 3) , paste0(x,"_b")],
                             pred = test[which(test$Hs_b > 3) , paste0(x,"_c")],
                             digits = 3,
                             var = x)})

row.names(rf3) <- c("Biais", "Rmse", "SI", "Cor")
```



##### Gradient boosting trees

We follow the same workflow for gradient boosting trees. First we fit the models to the data :

```r
x_train = data.matrix(training[, c(5:7)])
y_train = data.matrix(training[, c(8:10)])

x_test = data.matrix(test[, c(5:7)])
y_test = data.matrix(test[, c(8:10)])

test_predictions <- data.frame(rep(NA, nrow(test)))

for(j in 1:3){
  
  xgb <- xgboost(data = x_train, 
                 label = y_train[,j], 
                 nround = 500, 
                 eval_metric = "rmse",
                 print_every_n = 100
  )
  test_predictions[,j] <- predict(xgb, x_test)  
}
```

Then we compute the metrics :

```
test %<>% mutate(Hs_c_xg = round(Hs_m + test_predictions[,1], digits = 2),
                      Tp_c_xg = round(Tp_m + test_predictions[,2], digits = 2),
                      Dir_c_xg = round((Dir_m + test_predictions[,3])%%360, digits = 2))


xg <- sapply(c("Hs", "Tp", "Dir"), 
             function(x){
               metrics_wave(obs = test[ , paste0(x,"_b")],
                            pred = test[ , paste0(x,"_c_xg")],
                            digits = 3,
                             var = x)})

row.names(xg) <- c("Biais", "Rmse", "SI", "Cor")



xg3 <- sapply(c("Hs", "Tp", "Dir"), 
              function(x){
                metrics_wave(obs = test[which(test$Hs_b > 3) , paste0(x,"_b")],
                             pred = test[which(test$Hs_b > 3) , paste0(x,"_c_xg")],
                             digits = 3,
                             var = x)})

row.names(xg3) <- c("Biais", "Rmse", "SI", "Cor")
```


##### Neural networks

We fit the neural networks:

```r
#Function to build NN model
build_model <- function(){
  model <- keras_model_sequential() %>%
    layer_dense(units = (2*dim(x_training)[2]+1), activation = "sigmoid",
                input_shape = dim(x_training)[2]) %>%
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(lr = 0.001)
  )
  
  model
}



#Center and scale input variables 
x_training <- training[, c(5:7)]
x_test <- test[, c(5:7)]

pp = preProcess(x_training, method = c("center", "scale")) 
x_training <- as.matrix(predict(pp, x_training))
x_test <- as.matrix(predict(pp, x_test)) 



#Center and scale output variables
y_training <- training[, c(8:10)]
y_test <-  test[, c(8:10)]

pp1 = preProcess(y_training, method = c("center", "scale")) 
y_training<- as.matrix(predict(pp1, y_training))
y_test <- as.matrix(predict(pp1, y_test)) 


test_predictions <- data.frame(rep(NA, nrow(x_test)))
dvar = c("Hs_diff", "Tp_diff", "Dir_diff")

for(j in 1:3){
  
  model <- build_model()
  epochs <- 30
  # Fit the model and store training stats
  
  model %>% fit(
    x_training,
    y_training[,j],
    epochs = epochs,
    validation_split = 0.2, 
    verbose = 0 )
  
  test_predictions[,j] <- model %>% predict(as.matrix(x_test))
  k_clear_session()
  
}
```

We then compute the metrics :

```r
names(test_predictions) <- c("Hs_diff", "Tp_diff","Dir_diff")

# Function to transform back the data : 
unPreProc <- function(x, pp){
  nc <- ncol(x)
  y <- data.frame(rep(NA,nrow(x)))
  if(names(pp$method)[1] == "range"){
    for(i in 1:nc){
      y[,i] <-  data.frame((x[,i] * (max(pp$ranges[,i]) - min(pp$ranges[,i])) + min(pp$ranges[,i])))
    }
  }else{
    for(i in 1:nc){
      y[,i] <-  data.frame((x[,i] * pp$std[i])+ pp$mean[i])
    }
    
  }
  
  
  names(y) <- names(test_predictions)[1:ncol(x)]
  return(y)
}

test %<>% mutate(Hs_c_ann = round(Hs_m + unPreProc(test_predictions, pp1)[,1], digits = 2),
                      Tp_c_ann = round(Tp_m + unPreProc(test_predictions, pp1)[,2], digits = 2),
                      Dir_c_ann = round((Dir_m + unPreProc(test_predictions, pp1)[,3]) %% 360, digits = 2))


ann <- sapply(c("Hs", "Tp", "Dir"), 
              function(x){
                metrics_wave(obs = test[ , paste0(x,"_b")],
                             pred = test[ , paste0(x,"_c_ann")],
                             digits = 3,
                             var = x)})

row.names(ann) <- c("Biais", "Rmse", "SI", "Cor")



ann3 <- sapply(c("Hs", "Tp", "Dir"), 
               function(x){
                 metrics_wave(obs = test[which(test$Hs_b > 3) , paste0(x,"_b")],
                              pred = test[which(test$Hs_b > 3) , paste0(x,"_c_ann")],
                              digits = 3,
                             var = x)})

row.names(ann3) <- c("Biais", "Rmse", "SI", "Cor")
```
#### Results for prelimanry assimilation (no hyperparameter tuning)

Metrics computed on all data: 

```r
# Numerical model
##           Hs    Tp    Dir
## Biais -0.195 0.736 -0.296
## Rmse   0.391 1.883 14.176
## SI     0.164 0.157  0.047
## Cor    0.955 0.770  0.321

# Ann assimilation
##           Hs    Tp    Dir
## Biais -0.006 0.021  1.051
## Rmse   0.314 1.642 12.797
## SI     0.152 0.148  0.042
## Cor    0.960 0.793  0.356

# Random forest assimilation 
##          Hs    Tp    Dir
## Biais 0.002 0.002  1.079
## Rmse  0.292 1.494 11.676
## SI    0.141 0.135  0.039
## Cor   0.965 0.832  0.433

# Gradient boosting trees assimilation
##          Hs    Tp    Dir
## Biais 0.003 0.009  0.612
## Rmse  0.307 1.572 12.338
## SI    0.149 0.142  0.041
## Cor   0.961 0.814  0.300



```
Metrics computed with data where Hs > 3m:

```r
# Numerical model
##           Hs    Tp   Dir
## Biais -0.533 0.705 2.378
## Rmse   0.755 1.384 8.045
## SI     0.130 0.091 0.026
## Cor    0.832 0.820 0.528

# Ann assimilation
##           Hs     Tp    Dir
## Biais -0.187 -0.044 -0.368
## Rmse   0.546  1.167  6.179
## SI     0.124  0.089  0.021
## Cor    0.856  0.821  0.526

#Random forest assimilation
##           Hs     Tp  Dir
## Biais -0.177 -0.041 0.54
## Rmse   0.497  1.086 5.89
## SI     0.113  0.083 0.02
## Cor    0.876  0.847 0.58

# Gradient boosting trees assimilation
##           Hs     Tp   Dir
## Biais -0.159 -0.051 0.223
## Rmse   0.532  1.150 5.884
## SI     0.123  0.088 0.020
## Cor    0.861  0.828 0.584


```

### Fitting the model with optimal hyperparameters values 

#### Bayesian optimization for hyperaparameters

We want to find the hyperparameter values that yield the best results on unseen data by using Bayesian optimization coupled with cross validation. 


##### For neural networks

```r
library(rBayesianOptimization)

# Define cross validation
cv_folds <- rBayesianOptimization::KFold(training[, "Hs_diff"], nfolds = 4,
                                         stratified = F, seed = 0)



OPT_Res <- list()
# Loop for the three parameters to improve
for(i in 1:3){
  # Function to optimize= out-of-sample performance of ANN: 
  Ann_cv_bayes <- function(n_units, bacth_size, acti_fun, learning_rate, epo) {
    b_size <- c(32, 64, 128)[bacth_size]
    acti_func <- c('relu', 'sigmoid','tanh')[acti_fun]
    epoch <- c(10, 30, 50, 100, 150)[epo]
    test_predictions <- c()
    for(j in 1:length(cv_folds)){
      sampcv <- unlist(cv_folds[-j])
      
      x_training <- training[sampcv, c(5:7)]
      x_test <- training[cv_folds[[j]], c(5:7)]
      
      pp = preProcess(x_training,method = c("center", "scale")) 
      x_training <- as.matrix(predict(pp, x_training))
      x_test <- as.matrix(predict(pp, x_test)) 
      
      
      y_training <- training[sampcv, c(8:10)]
      y_test <-  training[cv_folds[[j]], c(8:10)]
      pp1 = preProcess(y_training, method = c("center", "scale")) 
      y_training <- as.matrix(predict(pp1, y_training))
      y_test <- as.matrix(predict(pp1, y_test)) 
      
      
      
      model <- keras_model_sequential() %>%
        layer_dense(units = n_units, activation = acti_func,
                    input_shape = dim(x_training)[2]) %>%
        layer_dense(units = 1, activation = "linear")
      
      model %>% compile(
        loss = "mse",
        metrics = list("RootMeanSquaredError"),
        optimizer = optimizer_adam(lr = learning_rate)
      )
    
      
      
      model %>% fit(
        x_training,
        y_training[,i],
        batch_size = b_size,
        epochs = epoch,
        validation_split = 0.2,
        verbose = 0 
      )
      
      test_predictions[j] <- evaluate(model, x = x_test, y = unlist(y_test[,i]),
                                      verbose = 0)$root_mean_squared_error
      k_clear_session()
      
    }
    
    
    list(Score = -mean(test_predictions),
         Pred = NA)
  }
  
  # Bayesian optimization function with the range associated with each hyperparamter
  OPT_Res[[i]] <-BayesianOptimization(Ann_cv_bayes,
                                      bounds = list(n_units = c(1L, 20L),
                                                    bacth_size = c(1L, 3L),
                                                    acti_fun = c(1L,3L),
                                                    learning_rate = c(0.0001, 0.1),
                                                    epo = c(1L, 5L)),
                                      init_grid_dt = NULL, init_points = 5, n_iter = 25,
                                      acq = "ucb", verbose = T)
}
#Store the results of the optimization for the three parameters 
res_opt_ann <- sapply(OPT_Res, function(x){return(x[1])})
```


##### For gradient boosting

```r
OPT_Res <- list()
# Loop for the three parameters to improve
for(i in 1:3){
 
  dtrain <- xgb.DMatrix(as.matrix(training[,  c(5:7)]),
                        label = training[, dvar[i]])
   # Function to optimize= out-of-sample performance of gradient boosting trees:  
  xgb_cv_bayes <- function(min_child_weight, subsample, eta,
                           lambda, colsample, maxdepth, nround) {
    
    cv <- xgb.cv(params = list(booster = "gbtree", 
                               eta = eta,
                               max_depth = maxdepth,
                               min_child_weight = min_child_weight,
                               subsample = subsample,
                               colsample_bytree = colsample,
                               objective = "reg:linear",
                               eval_metric = "rmse"),
                 data = dtrain, nround = nround, prediction = TRUE,
                 showsd = TRUE, folds = cv_folds, verbose = 0)
    
    
    list(Score = -cv$evaluation_log$test_rmse_mean[which.min(cv$evaluation_log$test_rmse_mean)],
         Pred = cv$pred)
  }
   #Bayesian optimization function with the range associated with each hyperparamter
  OPT_Res[[i]] <- BayesianOptimization(xgb_cv_bayes,
                                       bounds = list(eta = c(0.0001, 0.3),
                                                     maxdepth = c(1L,20L),
                                                     min_child_weight = c(1L, 15L),
                                                     subsample = c(0.5, 1),
                                                     colsample = c(0.5, 1),
                                                     nround = c(10L, 1000L)),
                                       init_grid_dt = NULL, init_points = 5, n_iter = 25,
                                       acq = "ucb", verbose = TRUE)
  
  
}
# Storing the results 
res_opt_xgb <- sapply(OPT_Res, function(x){return(x[1])})
```

##### For random forest

```r
# Loop for the three parameters to improve
for(i in 1:3){
  # Function to optimize= out-of-sample performance of random forest: 
  Rf_cv_bayes <- function(n_tree, mtry) {
    test_predictions <- c()
    for(j in 1:length(cv_folds)){
      
      sampcv <- unlist(cv_folds[-j])
      
      x_training <- training[sampcv,  c(5:7)] 
      x_test <- training[cv_folds[[j]], c(5:7)]
      
      
      y_training <- as.matrix(training[sampcv, c(8:10)])
      y_test <-  as.matrix(training[cv_folds[[j]], c(8:10)])
      
      
      model <- ranger(as.formula(paste(dvar[i],"~  Hs_m + Tp_m + Dir_m")),
                      data = training[sampcv,], 
                      num.trees = n_tree,
                      mtry = mtry,
                      max.depth = 0,
                      min.node.size = 1)
      
      
      test_predictions[j] <- caret::RMSE(predict(model, training[-sampcv,])$predictions,
                                         training[-sampcv, dvar[i]])
      
    }
    
    
    list(Score = -mean(test_predictions),
         Pred = NA)
  }
  
  OPT_Res[[i]] <- BayesianOptimization(Rf_cv_bayes,
                                       bounds = list(n_tree = c(100L, 1000L),
                                                     mtry = c(1L, 3L)),
                                       init_grid_dt = NULL, init_points = 5, n_iter = 25,
                                       acq = "ucb", verbose = TRUE)
  
  
}

res_opt_rf <- sapply(OPT_Res, function(x){return(x[1])})
```

#### Fitting the model with the optimized hyperparameter values

Now that the bayesian optimization has been performed for each wave parameter, we can fit the models with the optimal hyperparameter values.

##### For neural networks

```r
# Fit the model with function and stored hyperpara. values: 
Ann_opt <- function(list_opt){
  test_prediction <- matrix(NA, ncol = 3, nrow = nrow(test))
  x_training <- training[, c(5:7)]
  x_test <- test[, c(5:7)]
  
  pp = preProcess(x_training, method = c("center", "scale")) 
  x_training <- as.matrix(predict(pp, x_training))
  x_test <- as.matrix(predict(pp, x_test)) 
  
  
  y_training <- as.matrix(training[, c(8:10)])
  y_test <-  as.matrix(test[, c(8:10)])
  pp1 <<- preProcess(y_training, method = c("center", "scale")) 
  y_training <- as.matrix(predict(pp1, y_training))
  y_test <- as.matrix(predict(pp1, y_test)) 
  
  for(i in 1:3){
    b_size <- c(16, 32, 64, 128)[list_opt[[i]][2]]
    acti_fun <- c('relu', 'sigmoid','tanh')[list_opt[[i]][3]]
    epoch <-  c(10, 30, 50, 100, 150)[list_opt[[i]][5]]
    model <- keras_model_sequential() %>%
      layer_dense(units = list_opt[[i]][1], activation = acti_fun,
                  input_shape = dim(x_training)[2]) %>%
      layer_dense(units = 1, activation = "linear")
    
    model %>% compile(
      loss = "mse",
      metrics = list("RootMeanSquaredError"),
      optimizer = optimizer_adam(lr = list_opt[[i]][4])
    )
    
    model %>% fit(
      x_training,
      y_training[,i],
      batch_size = b_size,
      epochs = epoch,
      validation_split = 0.2,
      verbose = 1
    )
    
    test_prediction[,i] <-  predict(model, as.matrix(x_test))
    k_clear_session()
  }
  return(test_prediction)
}

# Predictions of the optimized models: 

test_prediction <- Ann_opt(res_opt_ann)


# Metric computation: 

test %<>% mutate(Hs_c_annt = round(Hs_m + unPreProc(test_prediction, pp1)[,1], digits = 2),
                      Tp_c_annt = round(Tp_m + unPreProc(test_prediction, pp1)[,2], digits = 2),
                      Dir_c_annt = round((Dir_m + unPreProc(test_prediction, pp1)[,3])%%360, digits = 2))

annt <- sapply(c("Hs", "Tp", "Dir"), 
               function(x){
                 metrics_wave(obs = test[ , paste0(x,"_b")],
                              pred = test[ , paste0(x,"_c_annt")],
                              digits = 3, 
                             var = x)})

row.names(annt) <- c("Biais", "Rmse", "SI", "Cor")



annt3 <- sapply(c("Hs", "Tp", "Dir"), 
                function(x){
                  metrics_wave(obs = test[which(test$Hs_b > 3) , paste0(x,"_b")],
                               pred = test[which(test$Hs_b > 3) , paste0(x,"_c_annt")],
                               digits = 3, 
                             var = x)})

row.names(annt3) <- c("Biais", "Rmse", "SI", "Cor")
```


###### For gradient boosting trees

```r
# Fit the models with a single function: 
Xgboost_opt <- function(list_opt){
  
  for(i in 1:3){
    dtrain <- xgb.DMatrix(as.matrix(training[, c(5:7)]),
                          label = training[, dvar[i]])
    
    
    model <- xgboost(data = dtrain,
                     nround = list_opt[[i]][6], 
                     eta = list_opt[[i]][1],
                     max_depth = list_opt[[i]][2],
                     min_child_weight = list_opt[[i]][3],
                     subsample = list_opt[[i]][4],
                     colsample_bytree = list_opt[[i]][5],
                     eval_metric = "rmse",
                     print_every_n = 100
    )
    
    assign(paste0("xgb_final_", str_remove(dvar[i], "_diff")), model,
           envir = .GlobalEnv)
  }
}

Xgboost_opt(res_opt_xgb)

# Evaluate performance by computing metrics: 
test %<>% mutate(Hs_c_xgt = round(Hs_m + predict(xgb_final_Hs, data.matrix(test[, c(5:7)])), digits = 2),
                      Tp_c_xgt = round(Tp_m + predict(xgb_final_Tp, data.matrix(test[, c(5:7)])), digits = 2),
                      Dir_c_xgt = round((Dir_m + predict(xgb_final_Dir, data.matrix(test[, c(5:7)])))%%360, digits = 2))


xgt <- sapply(c("Hs", "Tp", "Dir"), 
              function(x){
                metrics_wave(obs = test[ , paste0(x,"_b")],
                             pred = test[ , paste0(x,"_c_xgt")],
                             digits = 3, 
                             var = x)})

row.names(xgt) <- c("Biais", "Rmse", "SI", "Cor")



xgt3 <- sapply(c("Hs", "Tp", "Dir"), 
               function(x){
                 metrics_wave(obs = test[which(test$Hs_b > 3) , paste0(x,"_b")],
                              pred = test[which(test$Hs_b > 3) , paste0(x,"_c_xgt")],
                              digits = 3, 
                             var = x)})

row.names(xgt3) <- c("Biais", "Rmse", "SI", "Cor")
```

###### For random forest

```r
# fit the models with single function: 
ranger_opt <- function(list_opt){
  for(i in 1:3){
    model <- ranger(as.formula(paste(dvar[i],"~  Hs_m + Tp_m + Dir_m")),
                    data = training, 
                    num.trees = res_opt_rf[[i]][1],
                    importance =  'permutation',
                    mtry = list_opt[[i]][2],
                    max.depth = 0,
                    min.node.size = 1)
    
    assign(paste0("rf_final_", str_remove(dvar[i], "_diff")), model,
           envir = .GlobalEnv)
  }
}

ranger_opt(res_opt_rf)

# Evaluate performance by computing metrics: 

test %<>% mutate(Hs_c_rft = round(Hs_m + predict(rf_final_Hs, test)$predictions, digits = 2),
                      Tp_c_rft = round(Tp_m + predict(rf_final_Tp, test)$predictions, digits = 2),
                      Dir_c_rft = round((Dir_m + predict(rf_final_Dir, test)$predictions)%%360, digits = 2))


rft <- sapply(c("Hs", "Tp", "Dir"), 
              function(x){
                metrics_wave(obs = test[ , paste0(x,"_b")],
                             pred = test[ , paste0(x,"_c_rft")],
                             digits = 3, 
                             var = x)})

row.names(rft) <- c("Biais", "Rmse", "SI", "Cor")



rft3 <- sapply(c("Hs", "Tp", "Dir"), 
               function(x){
                 metrics_wave(obs = test[which(test$Hs_b > 3) , paste0(x,"_b")],
                              pred = test[which(test$Hs_b > 3) , paste0(x,"_c_rft")],
                              digits = 3, 
                             var = x)})

row.names(rft3) <- c("Biais", "Rmse", "SI", "Cor")
```

#### Final results optimized

The final results are presented below:

Metrics computed on all data: 

``` r
## For all data:
## Numerical model
##           Hs    Tp    Dir
## Biais -0.195 0.736 -0.296
## Rmse   0.391 1.883 14.176
## SI     0.164 0.157  0.047
## Cor    0.955 0.770  0.321
## Ann assimilation
##           Hs     Tp    Dir
## Biais -0.009 -0.087  0.702
## Rmse   0.311  1.606 12.726
## SI     0.151  0.145  0.042
## Cor    0.960  0.804  0.362
## Rf assimilation
##          Hs    Tp    Dir
## Biais 0.002 0.003  1.019
## Rmse  0.292 1.493 11.649
## SI    0.142 0.135  0.039
## Cor   0.965 0.833  0.398
## Gbt assimilation
##          Hs    Tp    Dir
## Biais 0.003 0.010  0.860
## Rmse  0.312 1.570 12.247
## SI    0.151 0.142  0.041
## Cor   0.960 0.816  0.333
```


Metrics computed on data where Hs > 3 meters: 

```r
## For data where Hs > 3 meters:
## Numerical model
##           Hs    Tp   Dir
## Biais -0.533 0.705 2.378
## Rmse   0.755 1.384 8.045
## SI     0.130 0.091 0.026
## Cor    0.832 0.820 0.528
## Ann assimilation
##           Hs     Tp    Dir
## Biais -0.181 -0.090 -0.305
## Rmse   0.537  1.136  6.215
## SI     0.123  0.087  0.021
## Cor    0.858  0.832  0.511
## Rf assimilation
##           Hs     Tp   Dir
## Biais -0.180 -0.038 0.574
## Rmse   0.500  1.085 5.903
## SI     0.113  0.083 0.020
## Cor    0.875  0.847 0.578
## Gbt assimilation
##           Hs     Tp   Dir
## Biais -0.164 -0.037 0.264
## Rmse   0.533  1.139 5.948
## SI     0.123  0.087 0.020
## Cor    0.859  0.832 0.570
```

It is worth noting that the succes of the assimilation method depends on the quantity of data used. The more data we have, the more efficient the assimilation will be. More details can be found in the article. 



