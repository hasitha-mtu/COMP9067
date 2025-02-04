   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gradient_descent(bias, lambda1, alpha, X, Y, max_iter):


    # array will track MSE for each iteration of GD
    errorValues = []
    length = len(Y)

    
    for num in range(0, max_iter):   

        # calculate difference between predicted values and actual y values
        predictionError = ((bias + lambda1*X) - Y)

        # calculate the gradient for the bias and lambda
        gradBias =    (1.0/length)*  (np.sum((predictionError)))
        gradLambda = (1.0/length)*  (np.sum((predictionError*X)))

        # gradient descent rule to calculate new values for parameters
        bias = bias - (alpha * gradBias)
        lambda1 = lambda1 - (alpha * gradLambda)
    
        
        # calculate MSE for current values of lambda and bias
        MSE = (np.sum((predictionError)**2))/(2*length)
        print (MSE)
        errorValues.append(MSE)
        
    
  
    return bias,lambda1, errorValues




def linearRegression(X, Y):
    
    # set initial parameters for model
    bias = 0
    lambda1 = 0
    
    alpha = 0.1 # learning rate
    max_iter=500

    # call gredient decent to calculate intercept(=bias) and slope(lambda1)
    bias, lambda1, errorValues = gradient_descent(bias, lambda1, alpha, X, Y, max_iter)
    print ('Final bias and  lambda1 values are = ', bias, lambda1, " respecively." )
    
    
    # Plot the data and the liner regression line
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    plt.scatter(X, Y)
    yPredictions = (lambda1*X)+bias
    plt.plot(X,yPredictions,'k-')
    
    # Plot the error values for current apla value
    plt.subplot(2,1,2)
    plt.plot(errorValues)
    
   
    
def main():
    
    # Read data into a dataframe
    df = pd.read_excel('data.xlsx')
    df = df.dropna() 

    # Store feature and target data in separate arrays
    Y = df['Y'].values
    X = df['X'].values
    
    # Perform standarization on the feature data
    X = (X - np.mean(X))/np.std(X)
    
    linearRegression(X, Y)
    

    

main()
