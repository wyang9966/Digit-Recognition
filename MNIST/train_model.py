import tensorflow as tf
from tensorflow.keras import layers, models

def train_and_save_model():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Build the CNN model
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

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    # Save the model
    model.save('digit_recognition_model.h5')
    # model.save(r'C:\Users\wenta\PycharmProjects\Digit-Recognition\digit_recognition_model.h5')
    print("Model trained and saved as 'digit_recognition_model.h5'")

if __name__ == "__main__":
    train_and_save_model()
