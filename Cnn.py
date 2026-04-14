#%%
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#%% Load in the data
dataset = tf.keras.utils.image_dataset_from_directory(
    "Datasets/", 
labels= 'inferred', 
image_size=(256, 256),
batch_size=32,
shuffle=True,
seed=42
)


# %%
print(dataset.class_names)
class_names = dataset.class_names

normalize = tf.keras.layers.Rescaling(1./255)
dataset = dataset.map(lambda x, y: (normalize(x), y))

# %%
# splitting dataset into training and test 
total = len(dataset)  # number of batches
train_size = int(0.7 * total)
val_size   = int(0.2 * total)
test_size  = total - train_size - val_size

train_ds = dataset.take(train_size)
val_ds   = dataset.skip(train_size).take(val_size)
test_ds  = dataset.skip(train_size + val_size)

# %%
# establishing model architecture
model = Sequential([    

    # For these elements we could change the kernal size? or the type of function?
    # addtionally maybe the size of the pooling? 
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),


    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Perceptron part 

    Flatten(),
    Dense(256, activation='relu'),
    # we would want to change this to multi classification for the differnt types of contamination
    Dense(1, activation='sigmoid')
])

#%%

# reading that adam is better for this problem than SGD 
# trying that out 

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      loss= 'binary_crossentropy', 
      metrics=['accuracy'])    

#TRAINING 
#%%
history = model.fit(train_ds, validation_data=val_ds, epochs=10)



# %%
loss, acc = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")







# for the fun of looking we are making a part that will show and image and predict
# %% Grab a single image from test set and predict
import matplotlib.pyplot as plt

# Pull one batch from test_ds, then take the first image
for images, labels in test_ds.take(1):
    img = images[0]        # shape: (256, 256, 3)
    true_label = labels[0].numpy()

# Model expects a batch dimension → add it with expand_dims
img_batch = tf.expand_dims(img, axis=0)  # shape: (1, 256, 256, 3)

# Predict
pred_prob = model.predict(img_batch)[0][0]  # scalar probability
pred_label = 1 if pred_prob >= 0.5 else 0

# Map to class names

print(f"Predicted: {class_names[pred_label]} ({pred_prob:.2%} confidence)")

# Plot it
plt.imshow(img.numpy())
plt.title(f"True: {class_names[true_label]} | Pred: {class_names[pred_label]} ({pred_prob:.2%})")
plt.axis('off')
plt.show()
# %%
# seeing pretty much always predicing clean class 
# checking if its class imbalance but very sure it is not
# %%
# Print first batch of labels to see what 0 and 1 map to
import os
for cls in os.listdir("Datasets/"):
    count = len(os.listdir(f"Datasets/{cls}"))
    print(f"{cls}: {count} images")
# %%
# See what probabilities the model is actually outputting
for images, labels in test_ds.take(1):
    preds = model.predict(images)
    print(preds[:10])  # are these all clustered near 0 or 1?
# %%
