import keras
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def plot_sample(classes, X, y, index):
    plt.figure(figsize=(2, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


def main():
    # Load dataset
    # (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    X_train.shape

    #print(X_test.shape)
    #print(X_train.shape)
    #print(y_train[:5])
    y_train = y_train.reshape(-1, )
    #print(y_train[:5])

    y_test = y_test.reshape(-1, )

    classes = ["airplanes", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    plot_sample(classes, X_train, y_train, 95)

    plot_sample(classes, X_train, y_train, 110)

    #plt.show()

    # Normalize the data
    X_train = X_train / 255
    X_test = X_test / 255

    # ANN
    ann = models.Sequential([layers.Flatten(input_shape=(32, 32, 3)), layers.Dense(3000, activation='relu'),
                             layers.Dense(1000, activation='relu'), layers.Dense(10, activation='softmax')])

    ann.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_train = ann.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Number of parameters
    ann.summary()

    # Classification Report
    y_pred = ann.predict(X_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]

    print("Classification Report: \n", classification_report(y_test, y_pred_classes))

    accuracy = model_train.history['accuracy']
    val_accuracy = model_train.history['val_accuracy']
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('ANN: Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('ANN: Training and validation loss')
    plt.legend()
    plt.figure()

    # CNN
    cnn = models.Sequential([layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
                             layers.MaxPooling2D(2, 2),
                             layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
                             layers.MaxPooling2D(2, 2), layers.Flatten(), layers.Dense(64, activation='relu'),
                             layers.Dense(10, activation='softmax')])

    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_train = cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Number of parameters CNN
    cnn.summary()

    # Classification Report
    y_pred = cnn.predict(X_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]
    print("Classification Report: \n", classification_report(y_test, y_pred_classes))

    accuracy = model_train.history['accuracy']
    val_accuracy = model_train.history['val_accuracy']
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('CNN: Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('CNN: Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
