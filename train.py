import os
import matplotlib.image
import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Gets the names of all the classes.
class_names = os.listdir("test")
num_classes = len(class_names)
# Create a data pipeline to load in the dataset.
ds_train: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory="train",
    labels="inferred",
    image_size=(224, 224),
    label_mode="int",
    color_mode="rgb",
    class_names=class_names
)

ds_validation: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory="validation",
    labels="inferred",
    image_size=(224, 224),
    label_mode="int",
    color_mode="rgb",
    class_names=class_names
)

ds_test: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory="test",
    labels="inferred",
    image_size=(224, 224),
    label_mode="int",
    color_mode="rgb",
    class_names=class_names
)

# Normalize the data
ds_train = ds_train.map(lambda x, y: (x / 255, y))
ds_validation = ds_validation.map(lambda x, y: (x / 255, y))
ds_test = ds_test.map(lambda x, y: (x / 255, y))

plt.figure(figsize=(10, 7))
i = 1
# Display a random image from each training set.
for folder in os.listdir("train"):
    plt.subplot(3, 5, i)
    i += 1
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img_p = os.path.abspath("train/" + folder + "/" + os.listdir("train/" + folder)[np.random.randint(0, 1000)])
    plt.imshow(matplotlib.image.imread(img_p))
    plt.xlabel(folder)

plt.savefig('data.png')
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


history = model.fit(ds_train, epochs=2, validation_data=ds_validation)

# Display accuracy and loss graphs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('acc.png')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig('loss.png')
plt.show()

# Predict the labels for the test dataset.
test_pred = []
test_true = []

for image_batch, label_batch in ds_test:
    # Store the correct labels
    test_true.append(label_batch)
    # Predict the class for the current batch of files.
    preds = model.predict(image_batch, verbose=0)
    # Store the predicted labels
    test_pred.append(np.argmax(preds, axis=1))

# Convert the values into usable arrays.
correct = tf.concat([item for item in test_true], axis=0)
predicted = tf.concat([item for item in test_pred], axis=0)

# Calculate the confusion matrix using the arrays.
confusion_mx = np.zeros((num_classes, num_classes))
for i in range(len(correct)):
    confusion_mx[correct[i]][predicted[i]] += 1
print(confusion_mx)

# Display the confusion matrix as a table.
plt.figure()
plt.table(cellText=confusion_mx, loc=(0, 0), cellLoc='center')
plt.axis('off')
plt.title("Confusion matrix for the Test data: ")
plt.savefig('confusion.pdf')
plt.show()
Accuracy = (np.trace(confusion_mx) / 3000) * 100
print("Accuracy: ", Accuracy, "%")
