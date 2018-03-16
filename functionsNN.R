
# Sigmoid function 
sigmoidFunc <- function(x)
{
  1/(1 + exp(-x))
}

# Create a function to change class labels into binary (0,1) encoding
binaryData <- function(data){
  # Record class label
  # Note: class label is typically placed in last column of dataset
  classLabelIndex = ncol(data)
  
  # Get a vector of the class labels for each instance
  classLabelForEachInstance = data[,classLabelIndex]
  
  # Convert to binary
  levels(classLabelForEachInstance) = 1:0
  
  # Change dataset to have binary encoding
  data[,classLabelIndex] = classLabelForEachInstance
  
  return(data)
}

# Create number of folds with stratified sampling
createFolds <- function(data, numFolds)
{
  # Determine the number of class label values
  classIndex = ncol(data)
  classLabels = data[,classIndex]
  levels(classLabels) = 1:0
  data[,classIndex] = classLabels
  
  # Create a list of folds
  folds = vector("list", numFolds)
  
  # Apply stratified sampling:
  # Separate class labels
  data0 = subset(data, data[,classIndex] == 0, drop = FALSE)
  data1 = subset(data, data[,classIndex] == 1, drop = FALSE)
  
  # Record number of rows in each dataset
  instanceIndexData0 = as.numeric(rownames(data0))
  instanceIndexData1 = as.numeric(rownames(data1))
  
  # For each dataset do:
    # Pick a random set of instance indices from instanceIndexData0
#set.seed(1)
    sequenceOfInstanceData0 = sample(instanceIndexData0, length(instanceIndexData0), replace = FALSE)
    
    # Place random list of numbers in ten folds in sequential order
    # Set counter for and fold index number (j)
    j = 1
    while(length(sequenceOfInstanceData0) > 0)
    {
      # If the index number for folds exceeds the number of folds,
      # then set back to first fold
      if(j > length(folds))
      {
        j = 1;
        j = as.numeric(j);
      }
      
      # Add the ith element in sequenceOfInstanceData0 to the jth fold
      folds[[j]] = c(folds[[j]], sequenceOfInstanceData0[1])
      
      # Increment jth index number
      j = as.numeric(j);
      j = j + 1;
      
      # Remove the ith element in sequenceOfInstanceData0
      sequenceOfInstanceData0 = sequenceOfInstanceData0[-1]
    }
    
    # Pick a random set of instance indices from instanceIndexData1
    sequenceOfInstanceData1 = sample(instanceIndexData1, length(instanceIndexData1), replace = FALSE)
    
    # Place random list of numbers in ten folds in sequential order
    # Note: work our way backwards to ensure that each fold has roughly the same size
    j = numFolds
    while(length(sequenceOfInstanceData1) > 0)
    {
      # If the index number for folds exceeds the number of folds,
      # then set back to first fold
      if(j < 1)
      {
        j = numFolds;
        j = as.numeric(j);
      }
      
      # Add the 1st element in sequenceOfInstanceData1 to the jth fold
      folds[[j]] = c(folds[[j]], sequenceOfInstanceData1[1])
      
      # Increment jth index number
      j = as.numeric(j);
      j = j - 1;
      
      # Remove the 1st element in sequenceOfInstanceData1
      sequenceOfInstanceData1 = sequenceOfInstanceData1[-1]
    }
    
  return(folds)
}

# Build and train the neural network with 1 layer
trainNN <- function(
                    # x = indices 1:numFeatures
                    x, 
                    # y = class label index
                    y, 
                    # traindata = training data
                    traindata, 
                    # set hidden layers and neurons
                    hidden = numFeatures, 
                    # max iteration step (i.e. number of epochs)
                    epoch,
                    # learning rate 
                    learningRate)
{
  # Makes it reproducible
#set.seed(randomSeed)
  
  # Total number of istances
  numInstances = nrow(traindata)
  
  # Randomly organize data
  traindata = traindata[sample(nrow(traindata)),]
  
  # Get the training data 
  trainingData = unname(data.matrix(traindata[,x]))
  
  # Get the label
  classLabelForEachInstance = traindata[,y]
  levels(classLabelForEachInstance) = 1:0
  traindata[,y] = classLabelForEachInstance
  
  # Get the number of input features
  numFeatures = ncol(trainingData)
  
  # Number of categories for classifying (should be two: "mine" and "rock")
  classLabelCategories = length(unique(classLabelForEachInstance)) - 1
  
  # Set hidden layer size
  hiddenLayerSize = hidden
  
  # Create initial weights and bias 
  # Weights will be a 60 x 60 matrix with initial value range from (0,1)
  # Note: each weights column represents the jth hidden layer's weight for each input feature i
  weights = matrix(rnorm((numFeatures*hiddenLayerSize), mean = 0, sd = 1), nrow = numFeatures, ncol = hiddenLayerSize)

  # Bias is a 1 x 60 matrix
  bias = matrix(rnorm(hiddenLayerSize, mean = 0, sd = 1), nrow = 1, ncol = hiddenLayerSize)
  
  # Weights and bias for hidden layer to output layer
  # Weights will be a 60 x 1 since we have only one output node
  weights2 = matrix(rnorm((hiddenLayerSize*classLabelCategories), mean = 0, sd = 1), 
                    nrow = hiddenLayerSize, ncol = classLabelCategories);
  bias2 = matrix(rnorm(classLabelCategories, mean = 0, sd = 1), nrow = 1, ncol = classLabelCategories)
  
  # Create a variable to return the updated weights2
  weights2Updated = 0
  
  # Train the network
  # Stopping criteria here should be the number of epochs
  i = 0
  while(i < epoch) 
  {
    # Increment iteration index
    i = i + 1;
    
    # Loop through each instance
    for(j in 1:nrow(trainingData))
    {
      # Record the ith instance's feature vector
      X = trainingData[j,]
      
      # Record the ith instance's class label
      Y = as.numeric(as.character(classLabelForEachInstance[j]))
      
      # Forward: Compute output o_j
      # 1. Sum the product of the inputs with their corresponding set of weights and bias
      # Note: Returns a 1 x 60 matrix
      hiddenLayerResults = (X %*% weights) + bias
      
      # 2. apply sigmoid func to get score at each hidden layer node
      # In the notes, this is o_j for the hidden units
      hiddenLayer = sigmoidFunc(hiddenLayerResults)
      
      # 3. Sum the product of the hidden layer results with the second set of weights and bias
      # Note: output sum = product of hidden layer result and weights between the hidden and output layer.
      outputSum = (hiddenLayer %*% weights2) + bias2
     
      # 4. apply sigmoid func to get score at output node
      # In the notes, this is the o_j for the output unit (i.e. this is one number for binary classification)
      score = sigmoidFunc(outputSum)
      
      # Backpropagation:
      # 1. calculate error of output units
      del_out = score * (1 - score) * (Y - score);
      
      # 2. Determine updates for weights going to output units
      learningRate = as.numeric(learningRate)
      del_out = as.numeric(del_out)
      deltaWeights2 = learningRate * del_out * hiddenLayer;
      
      # Update bias2 here
      # bias2 value (o_i) is 1
      deltaBias2 = learningRate * as.numeric(del_out) * 1
        
      # Now update weights2 with deltaWeights
      weights2Updated = weights2 + as.numeric(deltaWeights2)
      
      # Update bias2 with deltaBias2
      bias2 = bias2 + deltaBias2
 
      # 3. Calculate error for hidden units
      # Note: In equation, we sum out for all k, but k in our case is 1 because
      # of binary classification
      del_hidden = t(hiddenLayer * (1 - hiddenLayer) * as.numeric(del_out)) * weights2
      
      # 4. determine updates for weights to hidden units using hidden_unit errors
      # i.e. o_i (the last argument) in this equation represents the output of the 
      # input units => just the feature value itself.
      deltaWeights1 = learningRate * del_hidden * X
      
      # Now update weights
      # Apply each row of deltaWeights1 to each column of weights
      for(j in 1:ncol(weights))
      {
        weights[,j] = weights[,j] + deltaWeights1[j]
      }
      
      # update bias for each del_hidden
      # Change back to a 1 x n, n = number of features
      bias = t(t(bias) + deltaWeights1)
      
      # update weights2 to weights2Updated
      weights2 = weights2Updated
    }
  }
  
  # Return the model to use in predicting
  model = list(weights = weights, weights2 = weights2Updated, bias = bias, bias2 = bias2);
  
  return(model)
}


# Function to predict test instances
# Set threshold to 0.5, to determine what the output class label is.
predictAccuracy <- function(model, data, foldNum, threshold = 0.5)
{
  x = 1:(ncol(data) - 1)
  testSet = unname(data.matrix(data[,x]))
  
  # Record class label index
  classIndex = ncol(data)
  
  # Extract information from model
  # 1. first set of weights
  weights = as.matrix(model$weights)
  
  # 2. second set of weights
  weights2 = as.matrix(model$weights2)
  
  # 3. first set of bias weights
  bias = as.matrix(model$bias)
  
  # 4. second set of bias weights
  bias2 = as.matrix(model$bias2)
  
  # Set count for number of times the predicted label matches the actual class label
  countRight = 0
  
  # For each instance, determine it's class label
  for(i in 1:nrow(testSet))
  {
    # Record the ith instance's feature vector
    X = testSet[i,]
    
    # Get the actual label
    actualLabel = data[i, classIndex]
    actualLabel = as.numeric(as.character(actualLabel))
    
    # Forward: Compute output o_j
    # 1. Sum the product of the inputs with their corresponding set of weights and bias
    # Note: Returns a 1 x 60 matrix
    hiddenLayerResults = (X %*% weights) + bias
    
    # 2. apply sigmoid func to get score at each hidden layer node
    # In the notes, this is o_j for the hidden units
    hiddenLayer = sigmoidFunc(hiddenLayerResults)
    
    # 3. Sum the product of the hidden layer results with the second set of weights and bias
    # Note: output sum = product of hidden layer result and weights between the hidden and output layer.
    outputSum = (hiddenLayer %*% weights2) + bias2
    
    # 4. apply sigmoid func to get score at output node
    # In the notes, this is the o_j for the output unit (i.e. this is one number for binary classification)
    score = sigmoidFunc(outputSum)
    
    # 5. determine predicted label based on threshold
    predictedLabel = c()
    if(score < threshold)
    {
      predictedLabel = 0
    }
    else
    {
      predictedLabel = 1
    }
    
    # 6. update countRight if predictedLabel == actualLabel
    if(predictedLabel == actualLabel)
    {
      countRight = countRight + 1
    }
    
    # print results
    cat(foldNum, predictedLabel, actualLabel, score, "\n")
  }

  accuracy = countRight/nrow(testSet)
  
  return(accuracy)
}

# Function to predict test instances
# Set threshold to 0.5, to determine what the output class label is.
predictLabel <- function(model, data, foldNum, threshold = 0.5)
{
  x = 1:(ncol(data) - 1)
  testSet = unname(data.matrix(data[,x]))
  
  # Record class label index
  classIndex = ncol(data)
  
  # Extract information from model
  # 1. first set of weights
  weights = as.matrix(model$weights)
  
  # 2. second set of weights
  weights2 = as.matrix(model$weights2)
  
  # 3. first set of bias weights
  bias = as.matrix(model$bias)
  
  # 4. second set of bias weights
  bias2 = as.matrix(model$bias2)
  
  # Set count for number of times the predicted label matches the actual class label
  countRight = 0
  
  # Create an empty data frame to store results
  results = data.frame(foldNum = double(), predictedLabel = double(), actualLabel = double(), score = double())
  
  # For each instance, determine it's class label
  for(i in 1:nrow(testSet))
  {
    # Record the ith instance's feature vector
    X = testSet[i,]
    
    # Get the actual label
    actualLabel = data[i, classIndex]
    actualLabel = as.numeric(as.character(actualLabel))
    
    # Forward: Compute output o_j
    # 1. Sum the product of the inputs with their corresponding set of weights and bias
    # Note: Returns a 1 x 60 matrix
    hiddenLayerResults = (X %*% weights) + bias
    
    # 2. apply sigmoid func to get score at each hidden layer node
    # In the notes, this is o_j for the hidden units
    hiddenLayer = sigmoidFunc(hiddenLayerResults)
    
    # 3. Sum the product of the hidden layer results with the second set of weights and bias
    # Note: output sum = product of hidden layer result and weights between the hidden and output layer.
    outputSum = (hiddenLayer %*% weights2) + bias2
    
    # 4. apply sigmoid func to get score at output node
    # In the notes, this is the o_j for the output unit (i.e. this is one number for binary classification)
    score = sigmoidFunc(outputSum)
    
    # 5. determine predicted label based on threshold
    predictedLabel = c()
    if(score < threshold)
    {
      predictedLabel = 0
    }
    else
    {
      predictedLabel = 1
    }
    
    # 6. update countRight if predictedLabel == actualLabel
    if(predictedLabel == actualLabel)
    {
      countRight = countRight + 1
    }
    
    # Record results in dataframe
    results[i,] = c(foldNum, predictedLabel, actualLabel, score)
  }
  
  return(results)
}

# Run part B
partB <- function(dataSet, numFolds, learningRate, epochs)
{
  # Make class label values binary
  dataSet = binaryData(dataSet)
  
  # Create folds based on numFolds
  folds = createFolds(dataSet, numFolds)
  
  # Create a list of accuracies for the ten folds
  accuracies = c()
  
  # For each fold
  for(i in 1:length(folds))
  {
    # Train a model
    m = trainNN(x = 1:(ncol(dataSet) - 1), y = ncol(dataSet), traindata = dataSet[-folds[[i]],], 
                    hidden = ncol(dataSet) - 1, epoch = epochs, learningRate = learningRate)

    # Predict test set using model (i.e. returns accuracy)
    accuracy = predictAccuracy(model = m, data = dataSet[folds[[i]],], foldNum = i)

    # Place accuracy into 'accuracies' vector
    accuracies = c(accuracies, accuracy)
  }
  
  # Returns the accuracy for given neural network
  return(mean(accuracies))
}

# Return a dataframe that consists of the results
partB3 <- function(dataSet, numFolds, learningRate, epochs)
{
  # Make class label values binary
  dataSet = binaryData(dataSet)
  
  # Create folds based on numFolds
  folds = createFolds(dataSet, numFolds)
  
  # Create a data frame of predictions for the ten folds
  predictions = data.frame(foldNum = double(), predictedLabel = double(), actualLabel = double(), score = double())
  
  # For each fold
  for(i in 1:length(folds))
  {
    # Train a model
    m = trainNN(x = 1:(ncol(dataSet) - 1), y = ncol(dataSet), traindata = dataSet[-folds[[i]],], 
                hidden = ncol(dataSet) - 1, epoch = epochs, learningRate = learningRate)
    
    # Predict test set using model (i.e. returns accuracy)
    prediction = predictLabel(model = m, data = dataSet[folds[[i]],], foldNum = i)
    
    # Place accuracy into 'accuracies' vector
    predictions = rbind(predictions, prediction)
  }
  
  # Returns the accuracy for given neural network
  return(predictions)
}







