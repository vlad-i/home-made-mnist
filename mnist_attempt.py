
import numpy
from keras.datasets import mnist
import matplotlib.pyplot as plt

def predict(input, weights):
    output = input.T.dot(weights)
    # normalize the output
    output /= (28*28)
    return output


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # well uh I guess we're training the network now, oh geez

    training_set_number = x_train.shape[0]

    binary_maker = lambda t: 1 if t > 15 else 0
    binarizer = numpy.vectorize(binary_maker)
    x_train = binarizer(x_train)

    x_train = x_train.reshape(training_set_number, 28*28)
    x_train = numpy.c_[numpy.ones((x_train.shape[0])), x_train]


    #same stuff for the test dataset
    test_set_number = x_test.shape[0]
    x_test = binarizer(x_test)
    x_test = x_test.reshape(test_set_number, 28 * 28)
    x_test = numpy.c_[numpy.ones((x_test.shape[0])), x_test]


    # to be a fully connected layer, we need 10 * 28 * 28 weights
    W = numpy.random.uniform(size=(x_train.shape[1], 10))

    lossHistory = []

    num_of_epochs = 12
    alpha = 0.1



    for epoch in numpy.arange(num_of_epochs):
        num_of_examples = x_train.shape[0]
        gradient = numpy.zeros(x_train.shape[1]*10).reshape(10, x_train.shape[1])
        loss = 0


        for i in range(0, num_of_examples):

            preds = predict(x_train[i], W)

            # we have an error vector for each number class, so the size is 10
            error = numpy.zeros(10)

            # we set the ground truth on the error vector, just like in gradient descent
            error[y_train[i]] = 1

            error = error - preds

            loss += numpy.sum(error ** 2)

            for j in range(0, 10):
                gradient[j] += x_train[i].T * error[j] / x_train[i].shape


        print("loss = ", str(loss))

        lossHistory.append(loss)
        gradient = gradient #/ (num_of_examples / 10)


        W += alpha * gradient.T

        #let's try a test, then
        successfulPredictions = 0

        for k in range(0, test_set_number):
            preds = predict(x_test[k],W)
            if numpy.argmax(preds) == y_test[k]:
                successfulPredictions += 1
        print("For the ", epoch, " epoch we predicted ", str(successfulPredictions/test_set_number), " of the cases successfulPredictions = ", successfulPredictions)




# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(numpy.arange(0, num_of_epochs), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()




