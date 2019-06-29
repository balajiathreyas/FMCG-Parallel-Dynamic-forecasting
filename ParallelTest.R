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

#* write the combined input file

write.csv(all_data,"C:\\Projects\\PoC\\FMCG- Retail Stores sales(External agents)\\all_data.csv")

##=========================================================================##
# filter only valid weekly sales data > = 0 and take log1P of it
# Dummy variables for columns and create holiday week indicators

sales_data <- all_data %>% filter(Weekly_Sales >= 0 & Store <=5 & Dept <=20) %>% mutate(IsHoliday=as.numeric(IsHoliday.x))  

sales_data$log_weekly_sales <- log1p(sales_data$Weekly_Sales)

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


# Dropping variables

n_var <- c("IsHoliday.x","Date","Saledate","Weekly_Sales")

final_data <-sales_data[,!(names(sales_data) %in% n_var)]

# filter weeks 36 -43 for test data and the remaining as train data

x_test <- sales_data %>% select(-"IsHoliday.x",-"Date",-"Saledate",-"Weekly_Sales") %>% filter(between (year_week,201240,201243))

y_test <- x_test$log_weekly_sales

x_train <- sales_data %>% select(-"IsHoliday.x",-"Date",-"Saledate",-"Weekly_Sales") %>% filter(year_week < 201240)


tr_split <-  split(x_train,list(x_train$Store,x_train$Dept))

#y_train  <-  x_train$log_weekly_sales

#x_train$log_weekly_sales <- NULL


###--- ------create model at store,dept and type level --------------###
#Column names are STORE.ITEM

# create prophet model to apply 

makepreds <- function(df)
{
  library(forecast) 
  
    # check regression
    xreg <- cbind(markdown1 = tr_split[[i]]["MarkDown1"],
                markdown4  = tr_split[[i]]["MarkDown4"],
                markdown5  = tr_split[[i]]["MarkDown5"],
                fuel_price = tr_split[[i]]["Fuel_Price"],
                cpi        = tr_split[[i]]["CPI"],
                Temp       = tr_split[[i]]["Temperature"],
                week       = tr_split[[i]]["week"],
                month      = tr_split[[i]]["month"],
                Holiday    = tr_split[[i]]["IsHoliday"]
                )
  
  st <- tr_split[[i]]$Store[i]
  dep <- tr_split[[i]]$Dept[i]
  y <- tr_split[[i]]$log_weekly_sales
  
  fit <- auto.arima(y,xreg=xreg,approximation=FALSE,trace=FALSE,stepwise=TRUE)
  
   
  
  # save the each model to disk
  currDate <- Sys.Date()
  
  FileName <- paste("C:\\Projects\\PoC\\FMCG- Retail Stores sales(External agents)\\models\\final_model","-",st,"-",dep,"-",currDate,".rds")
  
  saveRDS(fit, FileName)
  
  #Reset values
  y  <- NULL
  st <- NULL
  dep<- NULL
  
  #Return output to call
  
  return(fit)

}



#Apply the makepreds function to each data frame in tr_split


#mypreds <- sapply(tr_split,makepreds)

jsonedit(mypreds, mode = "view")

jsonedit(test,mode="view")

str(mypreds)



#ggtsdisplay(residuals(fit), main="ARIMA residuals")

#ggtsdisplay(arima.errors(fit),main="ARIMA errors")

##------------------------parallel processing parallel------------------######
library(parallel)
gc()

# Use the detectCores() function to find the number of cores in system
no_cores <- detectCores()
# Setup cluster
clust <- makeCluster(no_cores)

test<- parSapply(clust, tr_split, makepreds)
 

stopCluster(clust)

##------------------------parallel processing foreach-------------------######
library(foreach)
library(doParallel)

# Use the detectCores() function to find the number of cores in system
no_cores <- detectCores()
registerDoParallel(makeCluster(no_cores))

gc()


##------------------------ Parallel logic -----------------------------------------###
models <-list()

modelsP<-list()

start=Sys.time()

models <- foreach(i =1:length(tr_split)) %dopar% {
  
   makepreds(tr_split)
   
  
}

end=Sys.time()

 
#### Validation of prediction ##########

# load the model
super_model <- readRDS("C:\\Projects\\PoC\\FMCG- Retail Stores sales(External agents)\\models\\final_model - 1 - 1 - 2018-12-01 .rds")
print(super_model)


###---- Check for a store ------------------###

store_test <- x_test %>% filter(Store==1 & Dept==1) %>% select(MarkDown1,MarkDown4,MarkDown5,Temperature,CPI,Fuel_Price,week,month,IsHoliday,log_weekly_sales)

y_test <- store_test$log_weekly_sales

store_test$log_weekly_sales <- NULL
 
xreg_test <- cbind(markdown1 = store_test["MarkDown1"],
              markdown4  = store_test["MarkDown4"],
              markdown5  = store_test["MarkDown5"],
              fuel_price = store_test["Fuel_Price"],
              cpi        = store_test["CPI"],
              Temp       = store_test["Temperature"],
              week       = store_test["week"],
              month      = store_test["month"],
              Holiday    = store_test["IsHoliday"]
)
# make a predictions on "new data" using the final model
final_predictions <- forecast(super_model,xreg=xreg_test,h=4)

#new_preds <- (y_preds+LassoPred)/2

df <-  (sm(expm1(y_test),expm1(final_predictions$mean)))

smape <- mean(df, na.rm=TRUE)
