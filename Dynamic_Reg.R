# Libraries needed
library(tidyverse)
library(randomForest)
library(caret)
library(xgboost)
library(data.table)
library(lubridate)
library(forecast)
library(dummies)
library(parallel)
library(jsonlite)
library(ggplot2)

library(listviewer)
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

sales_data <- all_data %>% filter(Weekly_Sales >= 0 & Store ==1 & Dept ==3) %>% mutate(IsHoliday=as.numeric(IsHoliday.x))  

sales_data$log_weekly_sales <- log1p(sales_data$Weekly_Sales)

# EDA graphs on Sales_data

store_graph <- sales_data%>% filter(Store == 20)
ggplot(store_graph,aes(x=CPI,y=Weekly_Sales)) + geom_point(aes(color=store_graph$IsHoliday)) + geom_smooth()


ggplot(sales_data,aes(x=CPI,y=Weekly_Sales)) + geom_point(aes(color=sales_data$Type)) + geom_smooth()

ggplot(sales_data,aes(x=Temperature,y=Weekly_Sales)) + geom_point(aes(color=sales_data$Type)) + geom_smooth()

A_stores <- sales_data %>% filter(Type=='A')
ggplot(A_stores,aes(x=CPI,y=Weekly_Sales)) + geom_point(aes(color=A_stores$IsHoliday)) + geom_smooth()


#num_x <- sales_data[sapply(sales_data,is.numeric)]


sales_data<- fastDummies::dummy_cols(sales_data, select_columns =c('Dept','Type'))

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


# check for correlation in data
num_x <- sales_data %>% select('Temperature','Fuel_Price','CPI','Unemployment','week','month','Weekly_Sales')

#num_x <- sales_data[ , purrr::map_lgl(sales_data,is.numeric)]

corr_mat<-cor(num_x)

corrplot::corrplot(corr_mat,method=c("circle"),type = c('full'))

# filter weeks 36 -43 for test data and the remaining as train data

x_test <- sales_data %>% select(-"IsHoliday.x",-"Date",-"Saledate",-"Weekly_Sales") %>% filter(between (year_week,201240,201243))

y_test <- x_test$log_weekly_sales


x_train <- sales_data %>% select(-"IsHoliday.x",-"Date",-"Saledate",-"Weekly_Sales") %>% filter(year_week < 201240)

#y_train  <-  x_train$log_weekly_sales

#x_train$log_weekly_sales <- NULL

# check regression
xreg <- cbind(markdown1 = x_train[, "MarkDown1"],
              markdown4 = x_train[, "MarkDown4"],
              markdown5 = x_train[, "MarkDown5"],
              fuel_price = x_train[,"Fuel_Price"],
              cpi        = x_train[,"CPI"],
              Temp       = x_train[,"Temperature"],
              week       = x_train[,"week"],
              month      = x_train[,"month"],
              Holiday    = x_train[,"IsHoliday"],
              Dec3       = x_train[,"Dec3"],
              Dec4       = x_train[,"Dec4"],
              Jan1       = x_train[,"Jan1"],
              Oct3       = x_train[,'Oct3'],
              Oct4       = x_train[,"Oct4"])

xreg_test <- cbind(markdown1 = x_test[, "MarkDown1"],
                   markdown4 = x_test[, "MarkDown4"],
                   markdown5 = x_test[, "MarkDown5"],
                   fuel_price = x_test[,"Fuel_Price"],
                   cpi        = x_test[,"CPI"],
                   Temp       = x_test[,"Temperature"],
                   week       = x_test[,"week"],
                   month      = x_test[,"month"],
                   Holiday    = x_test[,"IsHoliday"],
                   Dec3       = x_test[,"Dec3"],
                   Dec4       = x_test[,"Dec4"],
                   Jan1       = x_test[,"Jan1"],
                   Oct3       = x_test[,'Oct3'],
                   Oct4       = x_test[,"Oct4"])


checkresiduals(fit)
summary(fit)

###--- ------create model at store,dept and type level --------------###
#Column names are STORE.ITEM
tr_split <-  split(x_train,list(x_train$Store,x_train$Dept))

# create prophet model to apply 
makepreds <- function(df)
{
  fit <- auto.arima(x_train[, "log_weekly_sales"], xreg = xreg)
  # print(x_train$Store)
  # print(x_train$Dept)
  fcast <- forecast(fit,xreg=xreg_test)
}


#Apply the makepreds function to each data frame in tr_split


mypreds <- sapply(tr_split,makepreds)

jsonedit(mypreds, mode = "view")

jsonedit(test,mode="view")

str(mypreds)

test <- map(tr_split,~makepreds(.x))
 

map(mypreds, length)

map_dbl(mypreds, ~length(.x))

#mypreds <- mclapply(tr_split,makepreds,mc.cores = 1)

#n <- length(mypreds[[1]])



 

ggtsdisplay(residuals(fit), main="ARIMA residuals")

ggtsdisplay(arima.errors(fit),main="ARIMA errors")


##parallel processing
library(doParallel)
library(foreach)

# Use the detectCores() function to find the number of cores in system
no_cores <- detectCores()

registerDoParallel(cores=4)

gc()

#List output
foreach(tr_split, .combine = list, .multicombine=TRUE)  %dopar%  makepreds


#### Validation of prediction ##########

#new_preds <- (y_preds+LassoPred)/2

df <-  (sm(expm1(y_test),expm1(mypreds[[4]])))

smape <- mean(df, na.rm=TRUE)
