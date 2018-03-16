# Run this R script with the following arguments:
# Rscript neuralnet trainfile num_folds learning_rate num_epochs 
 
# Clear the workspace
rm(list = ls())

# Download package foreign to read arff data format
# use method read.arff()
library("foreign")

# Set the working directory
setwd("./")

# debug
#setwd("/Users/nikoescanilla/Google Drive/CS760/HW/HW3")

# debug
#file = "sonar.arff"
#folds = 10
#rate = 0.1
#epoch = 100

# Get command line arguments
args = commandArgs(trailingOnly = TRUE)
file = args[1]
folds = args[2]
folds = as.numeric(folds)
rate = args[3]
rate = as.numeric(rate)
epoch = args[4]
epoch = as.numeric(epoch)

# Load the data
file = read.arff(file)

# Part A
# Set seed for reproducibility 
set.seed(1)
source("functionsNN.R")

# Part B
# For submission
results = partB(dataSet = file, numFolds = folds, learningRate = rate, epochs = epoch)
cat(results, "\n")

# For pdf
# COMMENT OUT WHEN YOU SUBMIT
# 1. Plot accuracy of the neural network constructed for 25, 50, 75 and 100 epochs.
# (With learning rate = 0.1 and number of folds = 10)
#epochs = c(25, 50, 75, 100)
#nn_10Folds_0.1RateResults = c()
#for(i in 1:length(epochs))
#{
#  file = "sonar.arff"
#  file = read.arff(file)
#  set.seed(1)
#  source("functionsNN.R")
#  
  # Compute accuracy
#  result = partB(dataSet = file, numFolds = 10, learningRate = 0.1, epochs = as.numeric(epochs[i]))
  
  # Add to nn_10Folds_0.1RateResults
#  nn_10Folds_0.1RateResults = c(nn_10Folds_0.1RateResults, result)
#  
#  rm(list=setdiff(ls(), union("nn_10Folds_0.1RateResults","epochs")))
#}

# Plot the accuracy results
#plot(nn_10Folds_0.1RateResults, type = "o",
#     ylim = c(0.70, 0.80), axes = FALSE, ann = FALSE)
#grid()
#
# Make x axis using training set sizes 25, 50, and 100
#axis(1, at = 1:4, lab = c(25, 50, 75, 100))

# Make y axis with horizontal labels that display ticks at certain 
# range
#vector = c(0.01*70:80)
#vector = vector[c(TRUE,FALSE)]
#axis(2, las = 1, at = vector)

# Create box around plot
#box()

#title(main = "Accuracy of Neural Network Constructed for 25, 50, 75 and 100 Epochs",
#      col.main = "red", font.main = 4)
#title(xlab = "Number of Epochs")
#title(ylab = "Test-Set Accuracy")

# 2. Plot accuracy of the neural network constructed with number of folds as 5, 10, 15, 20 and 25. 
# (With learning rate = 0.1 and number of epochs = 50)
#folds = c(5, 10, 15, 20, 25)
#nn_50Epochs_0.1RateResults = c()
#for(i in 1:length(folds))
#{
#  file = "sonar.arff"
#  file = read.arff(file)
#  set.seed(1)
#  source("functionsNN.R")
  
  # Compute accuracy
#  result = partB(dataSet = file, numFolds = as.numeric(folds[i]), learningRate = 0.1, epochs = 50)
  
  # Add to nn_10Folds_0.1RateResults
#  nn_50Epochs_0.1RateResults = c(nn_50Epochs_0.1RateResults, result)
  
#  rm(list=setdiff(ls(), union("nn_50Epochs_0.1RateResults","folds")))
#}

# Plot the accuracy results
#plot(nn_50Epochs_0.1RateResults, type = "o",
#     ylim = c(0.70, 0.80), axes = FALSE, ann = FALSE)
#grid()

# Make x axis using training set sizes 25, 50, and 100
#axis(1, at = 1:5, lab = c(5, 10, 15, 20, 25))

# Make y axis with horizontal labels that display ticks at certain 
# range
#vector = c(0.01*70:80)
#vector = vector[c(TRUE,FALSE)]
#axis(2, las = 1, at = vector)

# Create box around plot
#box()

#title(main = "Accuracy of Neural Network Constructed with \n5, 10, 15, 20, and 25 Cross Validation",
#      col.main = "red", font.main = 4)
#title(xlab = "Number of Folds")
#title(ylab = "Test-Set Accuracy")

# 3.Plot ROC curve for the neural network constructed with the following parameters: 
# (With learning rate = 0.1, number of epochs = 50, number of folds = 10)
#library(ROCR)

#file = "sonar.arff"
#folds = 10
#rate = 0.1
#epochs = 50
#file = read.arff(file)
#set.seed(1)
#source("functionsNN.R")

#pred = partB3(dataSet = file, numFolds = 10, learningRate = 0.1, epochs = 50)
#out = performance(prediction(pred$score, pred$actualLabel), measure = "tpr", x.measure = "fpr")
#plot(out)
#abline(a = 0, b = 1, lwd = 1, lty = 2)
#grid()

#title(main = "ROC Curve for a Neural Network Constructed with \nLearning Rate 0.1, 50 Epochs, and 10 Fold Cross Validation",
#      col.main = "red", font.main = 4)
