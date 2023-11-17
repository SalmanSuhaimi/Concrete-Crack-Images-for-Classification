#%%
#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications, models
import numpy as np
import matplotlib.pyplot as plt
import os, datetime
from sklearn.model_selection import train_test_split
import shutil
# %%
# Import dataset
dataset_concrete = r"C:\Users\salma\Desktop\10) SHRDC - AI AND ML\8) CAPSTONE\DAY 1\Capstone 1\data"

# List all classes
classes = os.listdir(dataset_concrete)

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Iterate through each class
for label, class_name in enumerate(classes):
    class_path = os.path.join(dataset_concrete, class_name)
    
    # Iterate through each file in the class
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        
        # Append the file path and label to the lists
        file_paths.append(file_path)
        labels.append(label)

# Split the data into training, validation, and test sets
train_files, test_files, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)
#%%
# Define your destination directories
train_dir = r"C:\Users\salma\Desktop\10) SHRDC - AI AND ML\8) CAPSTONE\DAY 1\Capstone 1\split_dataset\train"
validation_dir = r"C:\Users\salma\Desktop\10) SHRDC - AI AND ML\8) CAPSTONE\DAY 1\Capstone 1\split_dataset\validation"

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Copy files to the training directory
for src_path, label in zip(train_files, train_labels):
    class_name = classes[label]
    dst_path = os.path.join(train_dir, class_name, os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

# Copy files to the validation directory
for src_path, label in zip(val_files, val_labels):
    class_name = classes[label]
    dst_path = os.path.join(validation_dir, class_name, os.path.basename(src_path))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

# %%
#2. Data loading 
PATH = os.path.join(os.getcwd(), 'split_dataset')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
# %%
#3. Inspect some data examples
class_names = train_dataset.class_names   #class_names attributes

plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        plt.grid('off')
# %%
#4.Further split validation and test dataset
val_batches = tf.data.experimental.cardinality(validation_dataset )
test_dataset= validation_dataset.take(val_batches//5)
validation_dataset  = validation_dataset.skip(val_batches//5)

# %%
#5. Covert the tensorflow datasets into PrefetchDataset
AUTOTUNE  = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size = AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size = AUTOTUNE)

# %%
#6. Create a sequential model for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#7. Repeatedly apply image augmentation on a single image 
for image, _ in train_dataset.take(1):
    first_image = image[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,0))
        plt.imshow(augmented_image[0]/255)
        plt.axis('off')
        plt.grid('off')

# %%
#8. Define a layer for data normalization
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# %%
"""
Transfer learning model pipelin

data augmentation > preprocess input > transfer learning model
"""

#(A) Load the pretrained model using keras.applications
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape =IMG_SHAPE, include_top  =False,weights='imagenet')
base_model.summary()
keras.utils.plot_model(base_model)

# %%
#(B) Freeze the entire feature extractor
base_model.trainable=False
base_model.summary()

# %%
#(C) Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#(D) Create the output layer
output_layer = layers.Dense(len(class_names), activation = 'softmax')
#(E) Build the entire pipeline using Funtional API
#a. Input
inputs =keras.Input(shape=IMG_SHAPE)
#b. Data augmentation
x = data_augmentation(inputs)
#c. Data normalization
x = preprocess_input(x)
#d. Transfer learning feature extractor
x = base_model(x,training=False)
#e. Classification layers
x =  global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)
#g. Build the model
model = keras.Model(inputs = inputs, outputs = outputs)
model.summary()

# %%
#10. Compile the model
optimizer = optimizers.Adam(learning_rate = 0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss = loss ,metrics = ['accuracy'])

# %%
# Create a TensorBoard callback object
PATH = os.getcwd()
logpath = os.path.join(PATH,"tensorboard_log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb= callbacks.TensorBoard(logpath)
# %%
#Evaluate the model before training 
model.evaluate(test_dataset)
# %%
#11. Model training
early_stopping = callbacks.EarlyStopping(patience=2)
EPOCHS = 10
history = model.fit(train_dataset, 
                    validation_data = validation_dataset,
                    epochs = EPOCHS, 
                    callbacks = [tb,early_stopping])
# %%
#Evaluate the model with test data
model.evaluate(test_dataset)
# %%
#12. Fine tune the model 
fine_tune_at = 100
#Freeze all the layers before the 'fine_tune_at' layer
base_model.trainable =True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
base_model.summary()
# %%
#14. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# %%
#15. Model fine tune training
fine_tune_epoch = 10 
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(train_dataset,
                         validation_data = validation_dataset,
                         epochs = total_epoch,
                         initial_epoch=history.epoch[-1],
                         callbacks =[tb,early_stopping])
# %%
#Evaluate the model with test data
model.evaluate(test_dataset)
# %%
#16. Deployment
#(A) Retrieve a batch of images from test data and perform prediction
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions =model.predict_on_batch(image_batch)
# %%
#(B) Display results in matplotlib
prediction_indexes = np.argmax(predictions, axis=1)
# %%
# Create a label map for the classes
label_map = {i: names for i, names in enumerate(class_names)}
prediction_label = [label_map[i] for i in prediction_indexes]
label_class_list = [label_map[i] for i in label_batch]

plt.figure(figsize=(15, 15))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(f"Label:{label_class_list[i]}, Prediction:{prediction_label[i]}")
    plt.axis('off')
    plt.grid('off')
# %%
model.save(os.path.join('models','classify_v1.h5'))
# %%
