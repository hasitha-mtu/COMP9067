
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# This function will take in all the feature data X
# as well as the current coefficient and bias values
# It should multiply all the feature value by their associated 
# coefficient and add the bias. It should then return the predicted 
# y values
def hypothesis(X, coefficients, bias):
    print(f'hypothesis|X shape: {X.shape}')
    print(f'hypothesis|coefficients shape: {coefficients.shape}')
    # TODO: Calculate and return predicted results
    y = (X @ coefficients.reshape(-1, 1)) + bias
    print(f'hypothesis|y shape: {y.shape}')
    return y




def calculateRSquared(bias, coefficients,X, Y):
    
    predictedY = hypothesis(X, coefficients, bias)
    
    avgY = np.average(Y)
    totalSumSq = np.sum((avgY - Y)**2)
    
    sumSqRes = np.sum((predictedY - Y)**2)
    
    r2 = 1.0-(sumSqRes/totalSumSq)
    
    return r2
    


# Complete the gradient calculations in the gradient_descent function. The function calculates
# the gradient for the bias and the updated bias value. You should also update each of the
# coefficients using the same gradient descent update rule. Calculate the gradient and use this
# to update the coefficient. You will need one for loop for iterating with gradient descent and
# in this basic implementation you can use a second (inner) for loop for updating the value of
# each of the weights (coefficients). Once complete you should now be able to run your MLR
# algorithm
# def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
#     m, n = X.shape  # m: number of samples, n: number of features
#     X = np.concatenate((np.ones((m, 1)), X), axis=1)  # Add bias term to X
#     b = np.zeros(n + 1)  # Initialize coefficients and bias to 0
#
#     for _ in range(n_iterations):
#         y_hat = X @ b  # Predictions
#         error = y_hat - y  # Error
#
#         db = (2 / m) * np.sum(error)  # Gradient of bias
#         dw = (2 / m) * (X.T @ error)  # Gradient of coefficients
#
#         b[0] = b[0] - learning_rate * db  # Update bias
#         b[1:] = b[1:] - learning_rate * dw[1:]  # Update coefficients
#
#     return b  # Return the learned coefficients and bias

def gradient_descent(bias, coefficients, alpha, X, Y, max_iter):
    m, n = X.shape
    length = len(X)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    
    for num in range(0, max_iter):
        y_pred = hypothesis(X, coefficients, bias)
        error = y_pred - Y
        errorValues.append(error)
        db = (2 / m) * np.sum(error)
        dw = (2 / m) * (X.T @ error)
        print(f'gradient_descent|dw shape: {dw.shape}')
        print(f'gradient_descent|coefficients shape: {coefficients.shape}')
        print(f'gradient_descent|alpha: {alpha}')

        bias = bias - alpha * db
        coefficients = coefficients - alpha * dw
        # TODO:
        # Calculate predicted y values for current coefficient and bias values 
        # calculate and update bias using gradient descent rule
        # Update each coefficient value in turn using gradient descent rule

    
    # calculate R squared value for current coefficient and bias values
    rSquared = calculateRSquared(bias, coefficients,X, Y)
    print ("Final R2 value is ", rSquared)

    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients



# Complete this function for part 2 of the exercise. 

def calculateTestR2(bias, coefficients, testFile):
    pass

def multipleLinearRegression(X, Y):

    # set the number of coefficients (weights) equal to the number of features
    #complete this line of code
    coefficients = np.random.rand(X.shape[1])
    print(f'coefficients shape: {coefficients.shape}')
    
    bias = 0.0
    
    alpha = 0.1 # learning rate
    
    max_iter=100

    # call gredient decent, and get intercept(=bias) and coefficents
    bias, coefficients = gradient_descent(bias, coefficients, alpha, X, Y, max_iter)
    
    return bias, coefficients
    
    
    
def main():
    
    df = pd.read_csv("Boston.csv")
    df = df.dropna()
    print (df.shape)
    
    data = df.values

     
    # Seperate teh features from the target feature    
    Y = data[:, -1]
    print(f'Y shape: {Y.shape}')
    X = data[:, :-1]
    print(f'X shape: {X.shape}')
    
    # Standardize each of the features in the dataset. 
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        X[:, num] = feature
     
    # run regression function and return bias and coefficients (weights)
    bias, coefficients = multipleLinearRegression(X, Y)
    
    # Enable code if you have a test set  (as in part 2)
    testFile = "testData.csv"
    calculateTestR2(bias, coefficients, testFile)
    

    

if __name__ == "__main__":
    main()
