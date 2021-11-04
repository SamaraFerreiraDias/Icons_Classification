#===============================================================================
#Packages that I used 
#===============================================================================
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import sklearn.metrics as metrics
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline  
import seaborn as sns

from sklearn.model_selection import train_test_split

#!pip install keras==2.2.4                                       #Instaling 
#!pip install tensorflow                                    #the packges to work  
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from tensorflow.keras.optimizers import Nadam, Adam                  #I changed
from tensorflow.keras.utils import to_categorical                    #I changed
from tensorflow.keras.applications.vgg16 import VGG16                #I changed
from tensorflow.keras.applications.vgg19 import VGG19   
from tensorflow.keras.applications.densenet import DenseNet121
from keras.layers import Dropout, Flatten, Input, Dense      
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import random
import tensorflow as tf
import cv2 as cv
import os 
import glob

#===============================================================================
#Using data base of my Google Drive's links :)
#===============================================================================
from google.colab import drive
drive.mount("/content/drive")
#===============================================================================
#Importing data base (Icons)
#===============================================================================
airplane_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/airplane/', '*'))
ball_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/ball/', '*'))
bike_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/biking/', '*'))
bird_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/bird/', '*'))
kiss_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/kiss/', '*'))
arrow_directions_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/arrow_directions/', '*'))
blade_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/blade/', '*'))
boat_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/boat/', '*'))
emoticon_face_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/emotion_face/', '*'))
flower_dir = glob.glob(os.path.join('/content/drive/MyDrive/ColabNotebooks/Icons-50/flower/', '*'))

X_path = (airplane_dir + ball_dir + bike_dir + bird_dir + kiss_dir +
          arrow_directions_dir +  blade_dir + boat_dir + emoticon_face_dir + flower_dir)

X = []

for f in X_path:
    X.append(np.array(cv.resize(cv.imread(f), (224,224),
                                interpolation = cv.INTER_AREA))) 
    
X = np.array(X)
X = X / 255

#===============================================================================
#Defining labels (Icons)
#===============================================================================
l_airplane = np.zeros(len(airplane_dir))
l_airplane_string = ['airplane' for i in range(len(airplane_dir))]

l_ball = np.ones(len(ball_dir))
l_ball_string = ['ball' for i in range(len(ball_dir))]

l_bike = 2*np.ones(len(bike_dir))
l_bike_string = ['bike' for i in range(len(bike_dir))]

l_bird = 3*np.ones(len(bird_dir))
l_bird_string = ['bird' for i in range(len(bird_dir))]

l_kiss = 4*np.ones(len(kiss_dir))
l_kiss_string = ['kiss' for i in range(len(kiss_dir))]

l_arrow_directions = 5*np.ones(len(arrow_directions_dir))
l_arrow_directions_string = ['arrow_directions' for i in range(len(arrow_directions_dir))]

l_blade = 6*np.ones(len(blade_dir))
l_blade_string = ['blade' for i in range(len(blade_dir))]

l_boat = 7*np.ones(len(boat_dir))
l_boat_string = ['boat' for i in range(len(boat_dir))]

l_emoticon_face = 8*np.ones(len(emoticon_face_dir))
l_emoticon_face_string = ['emoticon_face' for i in range(len(emoticon_face_dir))]

l_flower = 9*np.ones(len(flower_dir))
l_flower_string = ['flower' for i in range(len(flower_dir))]

#===============================================================================
#Concatenating...
y_string = np.concatenate((l_airplane_string, l_ball_string, l_bike_string,
                           l_bird_string, l_kiss_string, l_arrow_directions_string,
                           l_blade_string, l_boat_string, l_emoticon_face_string,
                           l_flower_string))

y = np.concatenate((l_airplane, l_ball, l_bike, l_bird, l_kiss, l_arrow_directions,
                    l_blade, l_boat, l_emoticon_face, l_flower))
y = to_categorical(y, 10)

#=========================================================================
#Creating plots
#=========================================================================
fig,ax=plt.subplots(2,3)
fig.set_size_inches(15,15)
for i in range(2):
    for j in range (3):
        r = random.randint(0,len(y_string))
        ax[i,j].imshow(X[r][:,:,::-1])
        ax[i,j].set_title('Icons: ' + y_string[r])
        
plt.tight_layout()

#=========================================================================
#Quanty of Icons
#=========================================================================

listStyles_Of_Icons = []

for i in range(len(y_string)):
    found = False
    for j in range(len(listStyles_Of_Icons)):
        if y_string[i] == listStyles_Of_Icons[j]:
            found = True
            break
    if found == False:
        listStyles_Of_Icons.append(y_string[i])

Quantity_Icons = []

for i in range(len(listStyles_Of_Icons)):
    Quantity_Icons.append(0)
    for j in range(len(y_string)):
        if y_string[j] == listStyles_Of_Icons[i]:
            Quantity_Icons[i] += 1

Icons_data_Set = {'Type_of_Icons': listStyles_Of_Icons,
                 'Quantity': Quantity_Icons
                 }

df = pd.DataFrame(Icons_data_Set, columns=['Type_of_Icons', 'Quantity'])

print(df)

g = sns.barplot(x="Type_of_Icons", y="Quantity", data=df)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

#=============================================================================
#Bases, Train and Test
#Maybe it will be necessary add the parameter for Stratified Sampling so the class proportion will be preserved in the splitting. (stratify=y)
#Study some methods to balance this dataset, we have a disbalance one. (One possibility is choose 80 image per class and delete the others)
#==============================================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=42,stratify=y)

#===============================================================================
#Data Augmentation (manipuling figures zoom, shifting)
#===============================================================================
datagen = ImageDataGenerator(
        zoom_range = 0.1, # Aleatory zoom
        rotation_range= 15, 
        width_shift_range=0.1,  # horizontal shift
        height_shift_range=0.1,  # vertical shift
        horizontal_flip=True,  
        vertical_flip=True) 

datagen.fit(X_train)
#===============================================================================
#Problems of overfiting
#===============================================================================
datagen = ImageDataGenerator(
        zoom_range = 0.1, # Aleatory zoom
        rotation_range= 15, 
        width_shift_range=0.1,  # horizontal shift
        height_shift_range=0.1,  # vertical shift
        horizontal_flip=True,  
        vertical_flip=True)
datagen.fit(X_train)

#===============================================================================
#Creating CNN 
#===============================================================================
inp = Input((224,224,3))

conv1 = Conv2D(64, (5,5), padding='valid', activation= 'relu')(inp)
conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
conv1 = BatchNormalization()(conv1)

conv2 = Conv2D(96, (4,4), padding='valid', activation= 'relu')(conv1)
conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
conv2 = BatchNormalization()(conv2)

conv3 = Conv2D(128, (3,3), padding='valid', activation= 'relu')(conv2)
conv3 = MaxPooling2D(pool_size=(2,2))(conv3)
conv3 = BatchNormalization()(conv3)

conv4 = Conv2D(256, (3,3), padding='valid', activation= 'relu')(conv3)
conv4 = MaxPooling2D(pool_size=(2,2))(conv4)
conv4 = BatchNormalization()(conv4)

flat = Flatten()(conv4)

dense1 = Dense(512, activation= 'relu')(flat)
dense1 = Dropout(0.5)(dense1)

dense2 = Dense(64, activation= 'relu')(dense1)
dense2 = Dropout(0.1)(dense2)

out = Dense(10, activation = 'softmax')(dense2)

model = Model(inp, out)
model.summary()

# Other things to the neural net
#model.load_weights('my-CNN.32-0.03-0.99-1.33-0.72.hdf5')
#filepath = 'my-CNN.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'

# These 2 line below are used when we need to return for some saved point (It won't be used in this project)
#lr_red = keras.callbacks.ReduceLROnPlateau(monitor='acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001)
#chkpoint = keras.callbacks.ModelCheckpoint( monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy',
                 metrics=["accuracy"])

# The callback parameter is used only we need to continue a previous cnns weight
#history = model.fit(X_train, y_train, batch_size = 32, epochs = 50, initial_epoch = 0, 
#                    validation_data = (X_val, y_val), callbacks=[lr_red, chkpoint])

history = model.fit(X_train, y_train, batch_size = 32, epochs = 50, initial_epoch = 0, validation_data = (X_val, y_val))

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


pred = model.predict(X_val)
pred = np.argmax(pred, axis=1)
pred = pd.DataFrame(pred).replace({0: 'Airplane', 1: 'Ball',2: 'Bike', 3: 'Bird', 4: 'Kiss', 5: 'Arrow_directions', 6: 'Blade', 7: 'Boat',8: 'Emoticon_face', 9: 'Flower'})

y_val_string = np.argmax(y_val, axis=1)
y_val_string = pd.DataFrame(y_val_string).replace({0: 'Airplane', 1: 'Ball',2: 'Bike', 3: 'Bird', 4: 'Kiss', 5: 'Arrow_directions', 6: 'Blade', 7: 'Boat',8: 'Emoticon_face', 9: 'Flower'})

confusion_matrix = metrics.confusion_matrix(y_true=pred, y_pred=y_val_string, labels=['Airplane','Ball','Bike','Bird','Kiss','Arrow_directions','Blade','Boat','Emoticon_face','Flower'])

plot_confusion_matrix(confusion_matrix, 
                      normalize    = False,
                      target_names = ['Airplane','Ball','Bike','Bird','Kiss','Arrow_directions','Blade','Boat','Emoticon_face','Flower'],
                      title        = "Confusion Matrix")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()  

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#===============================================================================
#Transfer Learning - VGG 19 
#===============================================================================

datagen.fit(X_train)

vgg = VGG19(input_shape=(224,224,3), include_top = False, weights= 'imagenet')

x = vgg.output
x = Flatten()(x)
x = Dense(3078,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(10,activation='softmax')(x)

tf_model=Model(inputs=vgg.input,outputs=out)

#Block trains of CC
for layer in tf_model.layers[:20]:
    layer.trainable=False

tf_model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy',
                 metrics=["accuracy"])

history = tf_model.fit(X_train, y_train, batch_size = 32, epochs = 5,
                       initial_epoch = 0, validation_data = (X_val, y_val))

_, val_acc = tf_model.evaluate(X_val, y_val, verbose=1)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


pred = tf_model.predict(X_val)
pred = np.argmax(pred, axis=1)
pred = pd.DataFrame(pred).replace({0: 'Airplane', 1: 'Ball',2: 'Bike', 3: 'Bird', 4: 'Kiss', 5: 'Arrow_directions', 6: 'Blade', 7: 'Boat',8: 'Emoticon_face', 9: 'Flower'})

y_val_string = np.argmax(y_val, axis=1)
y_val_string = pd.DataFrame(y_val_string).replace({0: 'Airplane', 1: 'Ball',2: 'Bike', 3: 'Bird', 4: 'Kiss', 5: 'Arrow_directions', 6: 'Blade', 7: 'Boat',8: 'Emoticon_face', 9: 'Flower'})

confusion_matrix = metrics.confusion_matrix(y_true=pred, y_pred=y_val_string, labels=['Airplane','Ball','Bike','Bird','Kiss','Arrow_directions','Blade','Boat','Emoticon_face','Flower'])

plot_confusion_matrix(confusion_matrix, 
                      normalize    = False,
                      target_names = ['Airplane','Ball','Bike','Bird','Kiss','Arrow_directions','Blade','Boat','Emoticon_face','Flower'],
                      title        = "Confusion Matrix")
                      
tf_model.summary()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

pred = tf_model.predict(X_val)
pred = np.argmax(pred, axis = 1)
pred = pd.DataFrame(pred).replace({0:'airplane',1:'ball',2:'bike',3:'bird',4:'kiss',5:'arrow_directions',6:'blade',7:'boat',8:'emoticon_face',9:'flower'})

y_val_string = np.argmax(y_val, axis = 1)
y_val_string = pd.DataFrame(y_val_string).replace({0:'airplane',1:'ball',2:'bike',3:'bird',4:'kiss',5:'arrow_directions',6:'blade',7:'boat',8:'emoticon_face',9:'flower'})

mis_class = []
for i in range(len(y_val_string)):
    if(not y_val_string[0][i] == pred[0][i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break
        
count = 0
fig,ax = plt.subplots(3,2)
fig.set_size_inches(15,15)
for i in range (3):
    for j in range (2):
        ax[i,j].imshow(X_val[mis_class[count]][:,:,::-1])
        ax[i,j].set_title("Predicted Icon : "+str(pred[0][mis_class[count]])+"\n"+"Actual Icon : " + str(y_val_string[0][mis_class[count]]))
        plt.tight_layout()
        count+=1
#===============================================================================
#Transfer Learning - DenseNet 
#===============================================================================

datagen.fit(X_train)

densenet = DenseNet121(input_shape=(224,224,3), include_top = False, weights= 'imagenet')

x = densenet.output
x = Flatten()(x)
x = Dense(3078,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(10,activation='softmax')(x)

tf_model=Model(inputs=densenet.input,outputs=out)

#Block trains of CC
for layer in densenet.layers:
    layer.trainable = False

tf_model.compile(optimizer = Nadam(0.0001) , loss = 'categorical_crossentropy',
                 metrics=["accuracy"])

history = tf_model.fit(X_train, y_train, batch_size = 32, epochs = 10,
                       initial_epoch = 0, validation_data = (X_val, y_val))

_, val_acc = tf_model.evaluate(X_val, y_val, verbose=1)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


pred = tf_model.predict(X_val)
pred = np.argmax(pred, axis=1)
pred = pd.DataFrame(pred).replace({0: 'Airplane', 1: 'Ball',2: 'Bike', 3: 'Bird', 4: 'Kiss', 5: 'Arrow_directions', 6: 'Blade', 7: 'Boat',8: 'Emoticon_face', 9: 'Flower'})

y_val_string = np.argmax(y_val, axis=1)
y_val_string = pd.DataFrame(y_val_string).replace({0: 'Airplane', 1: 'Ball',2: 'Bike', 3: 'Bird', 4: 'Kiss', 5: 'Arrow_directions', 6: 'Blade', 7: 'Boat',8: 'Emoticon_face', 9: 'Flower'})

confusion_matrix = metrics.confusion_matrix(y_true=pred, y_pred=y_val_string, labels=['Airplane','Ball','Bike','Bird','Kiss','Arrow_directions','Blade','Boat','Emoticon_face','Flower'])

plot_confusion_matrix(confusion_matrix, 
                      normalize    = False,
                      target_names = ['Airplane','Ball','Bike','Bird','Kiss','Arrow_directions','Blade','Boat','Emoticon_face','Flower'],
                      title        = "Confusion Matrix")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

mis_class = []
for i in range(len(y_val_string)):
    if(not y_val_string[0][i] == pred[0][i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break
count = 0
fig,ax = plt.subplots(3,2)
fig.set_size_inches(15,15)
for i in range (3):
    for j in range (2):
        ax[i,j].imshow(X_val[mis_class[count]][:,:,::-1])
        ax[i,j].set_title("Predicted Icon : "+str(pred[0][mis_class[count]])+"\n"+"Actual Icon : " + str(y_val_string[0][mis_class[count]]))
        plt.tight_layout()
        count+=1


