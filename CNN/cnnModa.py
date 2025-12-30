# Import libraries
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix


# Load Fashion MNIST dataset
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

print("Training data shape:", train_X.shape, train_Y.shape)
print("Testing data shape:", test_X.shape, test_Y.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print("Total number of output classes:", nClasses)
print("Output classes:", classes)

# Display the first image in training and testing data
plt.figure(figsize=(5, 5))

plt.subplot(121)
plt.imshow(train_X[0, :, :], cmap='gray')
plt.title(f"Ground Truth: {train_Y[0]}")

plt.subplot(122)
plt.imshow(test_X[0, :, :], cmap='gray')
plt.title(f"Ground Truth: {test_Y[0]}")

plt.show()

# Reshape the dataset to have a single channel
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
print(train_X.shape, test_X.shape)

# Convert to float32 and normalize to [0,1]
train_X = train_X.astype('float32') / 255.
test_X = test_X.astype('float32') / 255.

# Convert class vectors to binary class matrices (one-hot encoding)
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print("Original label:", train_Y[0])
print("After conversion to one-hot:", train_Y_one_hot[0])

# Split training data into training and validation sets
train_X, valid_X, train_label, valid_label = train_test_split(
    train_X, train_Y_one_hot, test_size=0.2, random_state=13
)

print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

# Model parameters

# batch 64 -> 128
# epochs 5 -> 10

batch_size = 128
epochs = 20
num_classes = 10

# Build CNN model
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',input_shape=(28, 28, 1), padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# 4ta capa Conv2D
fashion_model.add(Conv2D(256, (3, 3), activation='linear', padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

fashion_model.add(Flatten())
# Capa Densa intermedia
fashion_model.add(Dense(256, activation='relu'))
fashion_model.add(LeakyReLU(alpha=0.1))
# Dropout para capa de salida
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))


# Compile the model
fashion_model.compile(
    loss='categorical_crossentropy',
    # optimizer
    # adam -> SGD con learning rate 0.1 y momentum 0.9
    optimizer=SGD(learning_rate=0.1, momentum=0.9),
    metrics=['accuracy']
)

# Display model summary again
fashion_model.summary()

# Train the model
fashion_train = fashion_model.fit(
    train_X, train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)

# Evaluate the model on test data
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print("Test loss:", test_eval[0])
print("Test accuracy:", test_eval[1])

# Extract training history
accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs_range = range(len(accuracy))

# Plot training and validation accuracy
plt.figure()
plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

# Plot training and validation loss
plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predict on test data
predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

# Metricas de evaluacion
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names, zero_division=0))

cm = confusion_matrix(test_Y, predicted_classes)
print("Confusion matrix:\n", cm)

per_class_acc = cm.diagonal() / cm.sum(axis=1)
per_class_acc = np.nan_to_num(per_class_acc)

for i, acc in enumerate(per_class_acc):
    print(f"{target_names[i]} accuracy: {acc:.4f}")

# Check how many predictions were correct
correct = np.where(predicted_classes == test_Y)[0]
print("Found %d correct labels." % len(correct))

# Plot some correctly classified examples
plt.figure(figsize=(10, 10))
for i, idx in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[idx].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(f"Predicted: {predicted_classes[idx]}, Actual: {test_Y[idx]}")
plt.tight_layout()
plt.show()

# Plot some incorrectly classified examples
incorrect = np.where(predicted_classes != test_Y)[0]
print("Found %d incorrect labels." % len(incorrect))

plt.figure(figsize=(10, 10))
for i, idx in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[idx].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(f"Predicted: {predicted_classes[idx]}, Actual: {test_Y[idx]}")
plt.tight_layout()
plt.show()