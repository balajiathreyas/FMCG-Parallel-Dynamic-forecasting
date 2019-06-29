# Libraries needed
library(tidyverse)
library(randomForest)
library(caret)
library(xgboost)
library(data.table)
library(lubridate)
library(forecast)
library(dummies)
# Functions

#Define MAPE

mape <- function(real, pred){
  return(100 * mean(abs((real - pred)/real))) # MAPE - Mean Absolute Percentage Error
}


# Define sMAPE

sm <- function(act, pred){ 
  sm <- 200 * abs(act - pred) / (abs(act) + abs(pred))# normal formula
  sm <- ifelse(is.na(act), NA,sm)                     # omit if act is NA
  sm <- ifelse(is.na(pred) & !is.na(act), 200,sm)     # max error if pred is NA and act is available
  sm <- ifelse(pred==0 & act==0, 0,sm)                # perfect (arbitrary 0/0=0)
  return (sm) 
}  
# read input files

sales <- fread("C:\\Projects\\PoC\\FMCG- Retail Stores sales(External agents)\\sales data-set.csv")

stores <-fread("C:\\Projects\\PoC\\FMCG- Retail Stores sales(External agents)\\stores data-set.csv")

features <- fread("C:\\Projects\\PoC\\FMCG- Retail Stores sales(External agents)\\Features data set.csv")

# merge files

df1 <- merge(sales,stores,by='Store',all.x = TRUE)

all_data <- merge(df1,features,by=c('Store','Date'),all.x = TRUE)

#============= Structure features  ================###

all_data$IsHoliday.y <- NULL

all_data$Saledate <- as.Date(all_data$Date,format = "%d/%m/%Y")

all_data$week <- week(all_data$Saledate)

all_data$month<- month(all_data$Saledate)

all_data$year <- year(all_data$Saledate)

all_data$year_week <- as.numeric(as.character(paste0(all_data$year,all_data$week)))

all_data$Weekly_Sales <- as.numeric(all_data$Weekly_Sales)
class(all_data$Weekly_Sales)
 

##=========================================================================##
# filter only valid weekly sales data > = 0 and take log1P of it
# Dummy variables for columns and create holiday week indicators

sales_data <- all_data %>% filter(Weekly_Sales >= 0) %>% mutate(IsHoliday=as.numeric(IsHoliday.x))  

sales_data$log_weekly_sales <- log1p(sales_data$Weekly_Sales)



# check for correlation in data

num_x <- sales_data %>% select('Temperature','Fuel_Price','CPI','Unemployment','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','log_weekly_sales')

corr_mat<-cor(num_x)

corrplot::corrplot(corr_mat,method=c("circle"),type = c('lower'))


sales_data<- fastDummies::dummy_cols(sales_data, select_columns =c('year','Store','Dept','Type'))

# Initialize festive weeks and set the flag to 1 for respective weeks
sales_data$Dec3 <- 0
sales_data$Dec4 <- 0
sales_data$Jan1 <- 0
sales_data$Oct3 <- 0
sales_data$Oct4 <- 0

sales_data$Dec3[sales_data$week == 51] <- 1

sales_data$Dec4[sales_data$week == 52] <- 1

sales_data$Jan1[sales_data$week == 01] <- 1

sales_data$Oct3[sales_data$week == 43] <- 1

sales_data$Oct4[sales_data$week == 44] <- 1

# Initialize NA values to 0

sales_data[is.na(sales_data)] <- 0


# filter weeks 36 -43 for test data and the remaining as train data

x_test <- sales_data %>% select("week","MarkDown1","MarkDown4","MarkDown5","CPI","Fuel_Price","IsHoliday","Dec3","Dec4","Jan1","year_week") %>% filter(between (year_week,201240,201243))

y_test <- x_test$log_weekly_sales


x_train <- sales_data %>% select("week","MarkDown1","MarkDown4","MarkDown5","CPI","Fuel_Price","IsHoliday","Dec3","Dec4","Jan1","year_week") %>% filter(year_week < 201240)

y_train  <-  x_train$log_weekly_sales


sapply(x_train,class)



#-------Lasso modelling --------------------###

set.seed(200)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

lasso_mod <- train(x=x_train, y=y_train, method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune


min(lasso_mod$results$RMSE)

lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')


LassoPred <- predict(lasso_mod, x_test)
predictions_lasso <- expm1(LassoPred) #need to reverse the log to the real values
head(predictions_lasso)

######-------- Trend forecasting ---------------------------######

#Decompose Trend component in train data

data_ts <- ts(y_train, freq = 365.25/7)

plot(data_ts)
stl_sales = stl(data_ts,"periodic")

#trend_stl_sales <-decomp_sales$trend
seasonal_stl_sales  <- stl_sales$time.series[,1]
trend_stl_sales     <- stl_sales$time.series[,2]
random_stl_sales    <- stl_sales$time.series[,3]

plot(trend_stl_sales)

#fit trend with arima
trend_fit <- auto.arima(trend_stl_sales)

# Forecast for the required weeks
trend_for <- as.vector(forecast(trend_fit, 23664)$mean)
plot(trend_for)

########----------------------------------------------------#######
#####-----------------------------------------------------------------------####
##########-----------   Model creation & tuning ----------------------------####
####----------------- Tuning and fitting a xgboost model ------------------#####
#####-----------------------------------------------------------------------####

feature_names <- names(x_train)

dtrain <- xgb.DMatrix(data=as.matrix(x_train),label=y_train, missing=NA)

dtest <- xgb.DMatrix(data=as.matrix(x_test), missing=NA) 

####################
# Cross-validation
####################

# Set up cross-validation scheme (3-fold)
foldsCV <- createFolds(y_train, k=5, list=TRUE, returnTrain=FALSE)


# Further grid tuning is needed. 

param <- list(booster = "gblinear"
              , objective = "reg:linear"
              , subsample = 0.7
              , max_depth = 5
              , colsample_bytree = 0.7
              , eta = 0.037
              , eval_metric = 'rmse'
              , base_score = 0.012 #average
              , min_child_weight = 100)

# Perform xgboost cross-validation

xgb_cv <- xgb.cv(data=dtrain,
                 params=param,
                 nrounds=100,
                 prediction=TRUE,
                 maximize=FALSE,
                 folds=foldsCV,
                 early_stopping_rounds = 30,
                 print_every_n = 5
)


# Check best results and get best nrounds
print(xgb_cv$evaluation_log[which.min(xgb_cv$evaluation_log$test_rmse_mean)])
nrounds <- xgb_cv$best_iteration

################
# Final model
################

xgb <- xgb.train(params = param
                 , data = dtrain
                 # , watchlist = list(train = dtrain)
                 , nrounds = nrounds
                 , verbose = 1
                 , print_every_n = 10
                 #, feval = amm_mae
)

###############
# Results
###############

# Feature Importance
importance_matrix <- xgb.importance(feature_names,model=xgb)

xgb.plot.importance(importance_matrix[1:30,])

# Predict
y_preds <- predict(xgb,dtest) 


#### Validation of prediction ##########

new_preds <- (y_preds+LassoPred)/2
  
df <-  (sm(expm1(y_test),expm1(new_preds)))

smape <- mean(df, na.rm=TRUE)


#------------------ARIMA -------------------------#

arima.model <-auto.arima(y_train,xreg = c(x_train$Fuel_Price,x_train$CPI,x_train$Unemployment))

summary(arima.model)





