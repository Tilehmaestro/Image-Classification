import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd


def load_data():
    # Load Fashion MNIST dataset
    train_images = np.fromfile('data/train-images-idx3-ubyte', dtype=np.uint8)[16:].reshape(-1, 28, 28, 1).astype(
        np.float32) / 255.0
    train_labels = np.fromfile('data/train-labels-idx1-ubyte', dtype=np.uint8)[8:].astype(np.int64)
    test_images = np.fromfile('data/t10k-images-idx3-ubyte', dtype=np.uint8)[16:].reshape(-1, 28, 28, 1).astype(
        np.float32) / 255.0
    test_labels = np.fromfile('data/t10k-labels-idx1-ubyte', dtype=np.uint8)[8:].astype(np.int64)

    return (train_images, train_labels), (test_images, test_labels)


def load_csv_data():
    # Load Fashion MNIST dataset from CSV files
    train_df = pd.read_csv('data/fashion-mnist_train.csv')
    test_df = pd.read_csv('data/fashion-mnist_test.csv')

    train_images = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    train_labels = train_df.iloc[:, 0].values.astype(np.int64)
    test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    test_labels = test_df.iloc[:, 0].values.astype(np.int64)

    return (train_images, train_labels), (test_images, test_labels)


def preprocess_data():
    # You can choose either binary format or CSV format for loading the data
    # Uncomment one of the following lines based on the format you want to use
    # (train_images, train_labels), (test_images, test_labels) = load_data()
    (train_images, train_labels), (test_images, test_labels) = load_csv_data()

    # Convert the labels into one-hot vectors
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return (train_images, train_labels), (test_images, test_labels)


def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    return history


if __name__ == "__main__":
    # Preprocess the data
    (train_images, train_labels), (test_images, test_labels) = preprocess_data()

    # Create the model
    model = create_model()

    # Compile the model
    model = compile_model(model)

    # Train the model
    history = train_model(model, train_images, train_labels, test_images, test_labels)

    # Save the trained model
    model.save('fashion_mnist_model.h5')
