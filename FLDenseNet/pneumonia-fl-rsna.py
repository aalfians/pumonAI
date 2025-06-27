import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import flwr as fl
import tensorflow as tf
import os
import pydicom as dicom
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from tensorflow import keras
from keras.utils import Sequence
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, f1_score
from keras.applications.densenet import DenseNet121, preprocess_input

IMG_SIZE = 256
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VERDICT = ['NORMAL', 'PNEUMONIA']
EPOCHS = 9
BATCH_SIZE = 16
ROOT_PATH = "../rsna-pneumonia-detection-challenge/stage_2_train_images/"

DATA_SPLIT = 'DIFF'

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = np.array(self.x[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = np.array(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])
        return batch_x, batch_y

x_data = []
y_data = []

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []
    
labels = pd.read_csv('../rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
labels2 = pd.read_csv('../rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')

combine = pd.merge(labels, labels2, how='inner', on=['patientId'])
combine = combine.drop_duplicates("patientId")
combine = combine[combine['class'] != 'No Lung Opacity / Not Normal']

for patient in combine['patientId']: 
    dcm_path = ROOT_PATH + patient + '.dcm'
    dcm = dicom.dcmread(dcm_path).pixel_array
    class_num = combine['Target'].loc[combine['patientId'] == patient].values[0]

    resized_arr = cv2.resize(dcm, (IMG_SIZE, IMG_SIZE)) # Reshaping images to preferred size
    resized_arr = np.stack((resized_arr,)*3, axis=-1)
    resized_arr = preprocess_input(resized_arr.astype(np.float32))
    x_data.append(resized_arr)
    y_data.append(class_num)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify=y_data, test_size=0.20)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, stratify=y_test, test_size=0.50)

x_test = np.array(x_test)
y_test = np.array(y_test)

def ClsModel(n_classes=1):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    x = layers.AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', name='dense_post_pool')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='sigmoid', name='predictions')(x)
    model = keras.Model(inputs=base_model.input, outputs = outputs)
    
    return model

model = ClsModel()
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])

train_gen = DataGenerator(x_train, y_train, BATCH_SIZE)
val_gen = DataGenerator(x_val, y_val, BATCH_SIZE)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.history = dict()
        self.history['loss'] = []
        self.history['accucary'] = []
        self.history['val_loss'] = []
        self.history['val_accuracy'] = []
        self.history['test_f1_score'] = []

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(train_gen, epochs=1, batch_size=BATCH_SIZE, validation_data = val_gen, steps_per_epoch=np.ceil(len(y_train)/BATCH_SIZE))
        
        self.history['loss'].append(r.history['loss'][0])
        self.history['accucary'].append(r.history['accuracy'][0])
        self.history['val_loss'].append(r.history['val_loss'][0])
        self.history['val_accuracy'].append(r.history['val_accuracy'][0])

        print("Fit history: ", r.history)
        return model.get_weights(), len(x_train), {"accuracy": float(r.history['accuracy'][0]), "loss": float(r.history['loss'][0]), "val_accuracy": float(r.history['val_accuracy'][0]), "val_loss": float(r.history['val_loss'][0])}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, _ = model.evaluate(x_test, y_test)

        predictions = model.predict(x_test, steps = np.ceil(len(y_test)/BATCH_SIZE))
        predictions = np.where(predictions > 0.5, 1, 0)
        predictions = predictions.flatten()

        epoch_f1_score = f1_score(y_test, predictions, average='weighted')
        self.history['test_f1_score'].append(epoch_f1_score)

        print("Test F1 Score: ", epoch_f1_score)
        return loss, len(x_test), {"f1_score": float(epoch_f1_score)}
    
client = FlowerClient()

history = fl.client.start_numpy_client(
    server_address="10.30.200.41:5002",
    client=client,
    grpc_max_message_length=1024*1024*1024,
    )

epochs = [i for i in range(1,EPOCHS+1)]
fig , ax = plt.subplots(1,3)
train_acc = client.history['accucary']
train_loss = client.history['loss']
val_acc = client.history['val_accuracy']
val_loss = client.history['val_loss']
test_f1_score = client.history['test_f1_score']
fig.set_size_inches(30,5)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")

ax[2].plot(epochs , test_f1_score , 'g-o' , label = 'F1 Score')
ax[2].set_title('Predictions F1 Score')
ax[2].legend()
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("F1 Score")

plt.show()

predictions = model.predict(x_test)
predictions = np.where(predictions > 0.5, 1, 0)
predictions = predictions.flatten()

# predictions = predictions.reshape(1,-1)[0]
predictions[:15]

df = pd.DataFrame(client.history)
df.to_csv('./every-split-data-metrics/' + DATA_SPLIT + '-batch-1.csv')
df

print(classification_report(y_test, predictions, target_names = ['Normal (Class 0)','Pneumonia (Class 1)']))

cm = confusion_matrix(y_test, predictions)
cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])

plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = VERDICT, yticklabels = VERDICT)