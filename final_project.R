###### Semi-supervised Learning with Convolutional Neural Network for Image Classification
###### Appliaction to CT Images
###### Author: Haiyue Song, Han Ji
# tensorflow
library(reticulate)
use_condaenv("r-reticulate",required = T)
reticulate::py_module_available("tensorflow")

# library
library(keras)
library(tensorflow)
library(oro.dicom)
library(tidyverse)
library(imager)
library(kableExtra)
library(pROC)

# read dicom
dat.path = "/2650 Final Project/DICOM" # path to DICOM images
dcmSphere <- readDICOM(dat.path, verbose=TRUE)
dcmImage <- create3D(dcmSphere)

# read the label
datainfo <- read.csv("overview.csv")

############### function defination
reduceresolution <- function(imageArray,scaleFactor){
  #' Reduce the resolution of image in array
  #' @param imageArray, the array / matrix saving the image
  #' @param scaleFactor, the resize scale factor
  #' @return resized image in array
  require(imager)
  # Convert the image to a 'cimg' object
  image_cimg <- as.cimg(imageArray)
  
  # Decrease the resolution
  resized_image <- imresize(image_cimg, scaleFactor)
  
  return(as.array(resized_image))
}
evaluation <- function(pred_prob,y.test,threshold=0.5,pred_type="prob"){
  #' get AUC, sensitivity, specificity, accuracy, and precision values
  #' @param pred_prob, predicted probability as input
  #' @param y.test, the labels of test dataset
  #' @param threshold, a numeric number of threshold to classify the probability to classes
  #' @return AUC, sensitivity, specificity, accuracy, and precision values
  
  require(pROC)
  # create roc curve
  roc_object <- roc(y.test, as.vector(pred_prob),levels=levels(y.test),direction="<")
  # auc
  auc <- roc_object$auc
  
  df <- data.frame(pred=as.numeric(pred_prob>threshold),label=as.numeric(y.test)-1)
  TP <- dim(df[(df$pred==1&df$label==1),])[1]
  TN <- dim(df[(df$pred==0&df$label==0),])[1]
  FP <- dim(df[(df$pred==1&df$label==0),])[1]
  FN <- dim(df[(df$pred==0&df$label==1),])[1]
  
  return(c(AUC=auc,sensitivity=TP/(TP+FN),specificity=TN/(TN+FP),accuracy=(TP+TN)/(TP+TN+FP+FN),precision=TP/(TP+FP)))
}

################# Read files and pre-processing
# read all the DICOM images in the directory
dicom_files <- list.files(dat.path, pattern = ".dcm", full.names = TRUE)

# create an empty array to store the pixel data of the DICOM images
x <- array(0, dim = c(length(dicom_files), 256, 256, 1))
train_x <- array(0, dim = c(0.8*length(dicom_files), 256, 256, 1))
test_x <- array(0, dim = c(0.2*length(dicom_files), 256, 256, 1))
set.seed(1)
test_ind <- c(sample(1:50,10,replace = FALSE), sample(51:100, 10))
# label dropping (40 remaining)
unlabeled_ind <- c(sample(setdiff(1:100,test_ind)[1:40],20,replace = FALSE), 
                   sample(setdiff(1:100,test_ind)[41:80], 20,replace = FALSE))
labeled_x <- array(0, dim = c(40, 256, 256, 1))
unlabeled_x <- array(0, dim = c(40, 256, 256, 1))
# label dropping (20 remaining)
unlabeled_ind_2 <- c(sample(setdiff(1:100,test_ind)[1:40],30,replace = FALSE), 
                     sample(setdiff(1:100,test_ind)[41:80], 30,replace = FALSE))
labeled_x_2 <- array(0, dim = c(20, 256, 256, 1))
unlabeled_x_2 <- array(0, dim = c(60, 256, 256, 1))
# loop through the DICOM files and save the pixel data to the array
j <- 0 
k <- 0
t <- 0
z <- 0
for (i in seq_along(dicom_files)) {
  newImage <- reduceresolution(dcmImage[,,i],1/2)
  if(i %in% test_ind){
    j <- j+1
    test_x[j,,,1]<-newImage
  }else{
    k <- k+1
    train_x[k,,,1]<-newImage
    if (i %in% unlabeled_ind){
      t <- t+1
      unlabeled_x[t,,,1] <- newImage
    }else{
      z <- z+1
      labeled_x[z,,,1] <-newImage
    }
  }
  x[i,,,1] <- newImage
}
test_x_ssl <- test_x

# normalization for supervised learning
scalefactor <- max(train_x)
train_x <- train_x/scalefactor
test_x <- test_x/scalefactor
# normalization for ssl
labeled_x <- labeled_x /scalefactor
unlabeled_x <- unlabeled_x/scalefactor
# split label
y <- ifelse(datainfo$Contrast=="True",1,0)
train_y <- y[-test_ind]
test_y <- y[test_ind]
labeled_y <- y[-union(test_ind,unlabeled_ind)]

################### supervised learning with full labeled images
model_full <- keras_model_sequential() %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu", input_shape = dim(train_x)[-1]) %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")
model_full %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)
history <- model_full %>% fit(
  train_x, train_y,
  epochs = 40,
  batch_size = 8
)
model_full %>% evaluate(test_x, test_y)
full_prob <- model_full%>% predict(test_x)
pred <- round(full_prob)
roc_full_test<- roc(as.factor(test_y), as.vector(full_prob))

# evaluation indicators
rbind("Training Data"=evaluation(model_full%>% predict(train_x,verbose=F),as.factor(train_y)),"Test Data"=evaluation(model_full%>% predict(test_x,verbose=F),as.factor(test_y))) %>%
  round(4)%>%
  kbl(caption="Evaluation Indicators of the full supervised learning model",booktabs = T, escape = F, align = "c") %>%
  kable_styling(full_width = F, latex_options = c("HOLD_position"))

# roc and auc
roc_full_train <- roc(train_y, as.vector(model_full%>% predict(train_x)),levels=levels(as.factor(test_y)),direction="<")

roc_curve_full_train <- data.frame(FPR = 1-roc_full_train$specificities, TPR = roc_full_train$sensitivities)
roc_curve_full_test <- data.frame(FPR = 1-roc_full_test$specificities, TPR = roc_full_test$sensitivities)
roc_data_full <- rbind(roc_curve_full_train, roc_curve_full_test)
roc_data_full$model <- c(rep("Supervised Learning (Training)",nrow(roc_curve_full_train)),rep("Supervised Learning (Test)",nrow(roc_curve_full_test)))

# plot ROC curves 
ggplot(roc_data_full, aes(x = FPR, y = TPR, color = model)) +
  geom_path() +
  geom_abline(intercept = 0, slope = 1, color = "grey", linetype = "dashed") +
  labs(x = "False Positive Rate", y = "True Positive Rate", 
       title = "ROC Curves of Supervised Learning",
       subtitle = paste0("Training AUC = ", round(roc_full_train$auc, 4),", Test AUC = ",round(roc_full_test$auc, 4))) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5,size = 8))+
  theme_minimal()

##################### Supervised learning on 40 labeled images
input_img <- layer_input(shape = dim(train_x)[-1])
x <- input_img %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_labeled <- keras_model(input_img, x)

# Compile the model
model_labeled %>% compile(optimizer = "adam", loss = "binary_crossentropy", metrics = c("accuracy"))

# Train the model on the labeled data
model %>% fit(labeled_x, labeled_y, epochs = 50, batch_size = 8)

# model_labeled %>% evaluate(test_x, test_y)
labeled_prob <- model_labeled %>% predict(test_x)
# evaluation
rbind("Training Data"=evaluation(model_labeled%>% predict(labeled_x,verbose=F),as.factor(labeled_y)),"Test Data"=evaluation(model_labeled%>% predict(test_x,verbose=F),as.factor(test_y))) %>%
  round(4)%>%
  kbl(caption="Evaluation Indicators of supervised learning model with 40 labeled images",booktabs = T, escape = F, align = "c") %>%
  kable_styling(full_width = F, latex_options = c("HOLD_position"))
# roc and auc
roc_40_train <- roc(as.factor(labeled_y), as.vector(model_labeled%>% predict(labeled_x)),levels=levels(as.factor(labeled_y)),direction="<")
roc_40_test <- roc(as.factor(test_y), as.vector(model_labeled%>% predict(test_x)),levels=levels(as.factor(test_y)),direction="<")
roc_curve_40_train <- data.frame(FPR = 1-roc_40_train$specificities, TPR = roc_40_train$sensitivities)
roc_curve_40_test <- data.frame(FPR = 1-roc_40_test$specificities, TPR = roc_40_test$sensitivities)
roc_data_40 <- rbind(roc_curve_40_train, roc_curve_40_test)
roc_data_40$model <- c(rep("Supervised Learning on 40 images (Training)",nrow(roc_curve_40_train)),rep("Supervised Learning on 40 images (Test)",nrow(roc_curve_40_test)))

# plot ROC curves for both models on one canvas using ggplot2
ggplot(roc_data_40, aes(x = FPR, y = TPR, color = model)) +
  geom_path() +
  geom_abline(intercept = 0, slope = 1, color = "grey", linetype = "dashed") +
  labs(x = "False Positive Rate", y = "True Positive Rate", 
       title = "ROC Curves of Supervised Learning on 40 images",
       subtitle = paste0("Training AUC = ", round(roc_40_train$auc, 4),", Test AUC = ",round(roc_40_test$auc, 4))) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5,size = 8))+
  theme_minimal()

#################### Train the SSL model
# Set a confidence threshold
threshold <- 0.95
record40 <- list()
model40 <- model_labeled
# Repeat the pseudo-labeling process for a number of iterations
for (i in 1:10) {
  
  # Generate pseudo-labels for the unlabeled data
  pseudo_labels <- model40 %>% predict(unlabeled_x)
  
  # Select only the most confident predictions
  max_probs <- apply(pseudo_labels, 1, max)
  confident_indices <- which(max_probs > threshold)
  confident_indices <- c(confident_indices, which((1 - max_probs) > threshold))
  
  if (length(confident_indices) > 0) { # check if there are any confident predictions
    confident_images <- unlabeled_x[confident_indices, , , ]
    confident_labels <- pseudo_labels[confident_indices, ]
    
    # Combine the labeled data and confident pseudo-labeled data
    if (length(dim(confident_images)) == 3) { # if the image has 3 dimensions, add an extra dimension
      confident_images <- array_reshape(confident_images, c(dim(confident_images), 1))
    }
    combined_images <- abind::abind(labeled_x, confident_images, along = 1)
    combined_labels <- c(labeled_y, confident_labels)
    
    # Retrain the model on the combined data
    model40 %>% fit(combined_images, combined_labels, epochs = 50, batch_size = 8, verbose = F)
    # Evaluate the model on the test set
    record40[[i]] <- model40 %>% evaluate(test_x, test_y,verbose = F)
  }
}

# Evaluate the model on the test set
model40 %>% evaluate(test_x, test_y)
ssl_prob <- model40%>% predict(test_x)

# Evaluate the first ssl model with 40 labeled images on the test set
# evaluation indicators
rbind("Labeled Training Data"=evaluation(model40%>% predict(labeled_x,verbose=F),as.factor(labeled_y)),"Test Data"=evaluation(model40%>% predict(test_x,verbose=F),as.factor(test_y))) %>%
  round(4)%>%
  kbl(caption="Evaluation Indicators of semi-supervised learning model with 40 labeled images",booktabs = T, escape = F, align = "c") %>%
  kable_styling(full_width = F, latex_options = c("HOLD_position"))
# roc and auc
roc_ssl40_train <- roc(as.factor(labeled_y), as.vector(model40%>% predict(labeled_x)),levels=levels(as.factor(labeled_y)),direction="<")
roc_ssl40_test <- roc(as.factor(test_y), as.vector(model40%>% predict(test_x)),levels=levels(as.factor(test_y)),direction="<")
roc_curve_ssl40_train <- data.frame(FPR = 1-roc_ssl40_train$specificities, TPR = roc_ssl40_train$sensitivities)
roc_curve_ssl40_test <- data.frame(FPR = 1-roc_ssl40_test$specificities, TPR = roc_ssl40_test$sensitivities)
roc_data_ssl40 <- rbind(roc_curve_ssl40_train, roc_curve_ssl40_test)
roc_data_ssl40$model <- c(rep("40 labeled traing images",nrow(roc_curve_ssl40_train)),rep("Test",nrow(roc_curve_ssl40_test)))

# plot ROC curves for both models on one canvas using ggplot2
ggplot(roc_data_ssl40, aes(x = FPR, y = TPR, color = model)) +
  geom_path() +
  geom_abline(intercept = 0, slope = 1, color = "grey", linetype = "dashed") +
  labs(x = "False Positive Rate", y = "True Positive Rate", 
       title = "ROC Curves of Semi-Supervised Learning with 40 labeled images",
       subtitle = paste0("Labeled Training AUC = ", round(roc_ssl40_train$auc, 4),", Test AUC = ",round(roc_ssl40_test$auc, 4))) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5,size = 8))+
  theme_minimal()

############## Comparison
cbind(cbind(full_prob, ssl_prob, labeled_prob) > 0.5, test_y)
cbind(full_prob, ssl_prob, labeled_prob)