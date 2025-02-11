
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd


# Sigmoid function
def logistic(x):
    return 1.0 / (1.0 + np.exp(-1.0 *x))


def hypothesisLogistic(X, coefficients, bias):
    print(f'hypothesisLogistic|X shape: {X.shape}')
    print(f'hypothesisLogistic|coefficients shape: {coefficients.shape}')
    # TODO: 3. This function should use matrix operations to push X through
    # the logistic regression unti and return the results
    coefficients = np.transpose(coefficients)
    print(f'hypothesisLogistic|coefficients new shape: {coefficients.shape}')
    hypothesis = np.matmul(coefficients, X) + bias
    print(f'hypothesisLogistic|hypothesis shape: {hypothesis.shape}')
    return coefficients@X + bias


def calculateCrossEntropyCost(predictedY, Y):

    return (- 1 / Y.shape[1]) * np.sum(Y * np.log(predictedY) + (1 - Y) * (np.log(1 - predictedY)))  # compute cost



def gradient_descent_log(bias, coefficients, alpha, X, Y, max_iter, X_val, y_val):
    print(f'gradient_descent_log|X shape: {X.shape}')
    print(f'gradient_descent_log|Y shape: {Y.shape}')
    m = X.shape[1]

    # array is used to store change in cost function for each iteration of GD
    trainingLoss= []
    validationLoss= []
    trainingAccuracies = []
    validationAccuracies = []


    for num in range(0, max_iter):

        # Calculate predicted y values for current coefficient and bias values 
        predictedY = hypothesisLogistic(X, coefficients, bias)
        print(f'gradient_descent_log|predictedY shape: {predictedY.shape}')
        E = Y - predictedY
        print(f'gradient_descent_log|E shape: {E.shape}')

        # TODO: 4. Calculate gradients for coefficients and bias
        d_lamda = (E@X)/m
        print(f'gradient_descent_log|d_lamda shape: {d_lamda.shape}')

        # TODO: 5. Execute gradient descent update rule 
        coefficients = coefficients - alpha*d_lamda

        if num %10 == 0:
            print ("Iteration Number ", num)

            # Cross Entropy Error  and accuracy for training data
            trainCost = calculateCrossEntropyCost(predictedY, Y)
            trainingLoss.append(trainCost)
            trainAccuracy = calculateAccuracy(predictedY, Y)
            trainingAccuracies.append(trainAccuracy)


            # Cross Entropy Error  and accuracy for validation data
            predictedYVal = hypothesisLogistic(X_val, coefficients, bias)
            valCost = calculateCrossEntropyCost(predictedYVal, y_val)
            validationLoss.append(valCost)
            valAccuracy = calculateAccuracy(predictedYVal, y_val)
            validationAccuracies.append(valAccuracy)


    plt.plot(validationLoss, label="Val Loss")
    plt.plot(validationAccuracies, label="Val Acc")
    plt.plot(trainingLoss, label="Train Loss")
    plt.plot(trainingAccuracies, label="Train Acc")
    plt.legend()

    plt.show()

    return bias, coefficients




def calculateAccuracy(predictedYValues, y_test):

    # Logistic regression is a probabilistic classifier.
    # If the probability is less than 0.5 set class to 0
    # If probability is greater than 0.5 set class to 1 
    predictedYValues[predictedYValues <= 0.5] = 0
    predictedYValues[predictedYValues > 0.5] = 1

    return np.sum(predictedYValues==y_test, axis=1 ) /y_test.shape[1]



def logisticRegression(X_train, y_train, X_validation, y_validation):


    # TODO 2: Create a column vector of coefficients for the model 
    coefficients = np.random.random((X_train.shape[1], 1))
    print(f'logisticRegression|coefficients shape: {coefficients.shape}')
    print(f'logisticRegression|coefficients type: {type(coefficients)}')

    bias = 0.0

    alpha = 0.005 # learning rate

    max_iter =500



    # call gredient decent, and get intercept(bias) and coefficents
    bias, coefficients = gradient_descent_log(bias, coefficients, alpha, X_train, y_train, max_iter, X_validation, y_validation)

    predictedY = hypothesisLogistic(X_train, coefficients, bias)
    print ("Final Accuracy: " ,calculateAccuracy(predictedY, y_train))

    predictedYVal = hypothesisLogistic(X_validation, coefficients, bias)
    print ("Final Validatin Accuracy: " ,calculateAccuracy(predictedYVal, y_validation))




def main():

    df = pd.read_csv('data/train.csv', sep=',' ,header=None)
    print(f'df shape: {df.shape}')
    trainData = df.values

    train_set_x_orig = trainData[:, 0:-1]
    train_set_y = trainData[:, -1]

    train_set_y = train_set_y.astype(np.int32)
    print (np.unique(train_set_y))

    df =pd.read_csv('data/validation.csv', sep=',' ,header=None)
    valData = df.values

    val_set_x_orig = valData[:, 0:-1]
    val_set_y = valData[:, -1]
    val_set_y = val_set_y.astype(np.int32)

    # Standarize the data
    scaler = preprocessing.StandardScaler()
    train_set_x_orig = scaler.fit_transform(train_set_x_orig)
    val_set_x_orig = scaler.fit_transform(val_set_x_orig)

    # Reshape the y data to that it becomes a real row vector
    train_set_y = train_set_y.reshape((1 ,len(train_set_y)))
    val_set_y = val_set_y.reshape((1 ,len(val_set_y)))

    print(f'train_set_x_orig shape: {train_set_x_orig.shape}')
    print(f'train_set_x_orig type: {type(train_set_x_orig)}')
    print(f'train_set_y shape: {train_set_y.shape}')
    print(f'val_set_x_orig shape: {val_set_x_orig.shape}')
    print(f'val_set_x_orig type: {type(val_set_x_orig)}')
    print(f'val_set_y shape: {val_set_y.shape}')

    # TODO: 1 Reshape the training data and test data so 
    # that the features becomes the rows of the matrix
    # Reshape train_set_x_orig
    train_set_x_orig = np.reshape(train_set_x_orig, (train_set_x_orig.shape[1], train_set_x_orig.shape[0]))
    print(f'train_set_x_orig new shape: {train_set_x_orig.shape}')
    # Reshape val_set_x_orig
    val_set_x_orig = np.reshape(val_set_x_orig, (val_set_x_orig.shape[1], val_set_x_orig.shape[0]))
    print(f'val_set_x_orig new shape: {val_set_x_orig.shape}')


    logisticRegression(train_set_x_orig, train_set_y, val_set_x_orig, val_set_y)


if __name__ == "__main__":
    main()
