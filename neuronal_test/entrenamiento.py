import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from d2l import  tensorflow as d2l

tf.keras.backend.clear_session()

def cargarDataset():
    dataForTraining  = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, shear_range=0.3, zoom_range=0.3, horizontal_flip=True)
    
    dataForTrainingTest = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    classCategorical = dataForTraining.flow_from_directory(
        './dataset/entrenamiento', target_size=(150,150), batch_size=32, class_mode='categorical')
    
    classCategoricalTest = dataForTrainingTest.flow_from_directory(
        './dataset/validacion', target_size=(150,150), batch_size=32, class_mode='categorical')
    
    print('Clases Entrenamiento:\n',classCategorical.class_indices)
    
    return classCategorical,classCategoricalTest

def guardarEntrenamiento(net):
    path_datos = './data/'
    if not os.path.exists(path_datos):
        os.mkdir(path_datos)
    net.save('./data/aprendido.h5')
    net.save_weights('./data/pesos.h5')

def ArquitecturaEntrenamiento(entrena, prueba):
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Conv2D(32,(3,3),padding='same',input_shape=(150,150,3),activation='relu')) #primera capa de convolución
    net.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) #1er filtro MaxPool

    net.add(tf.keras.layers.Conv2D(64,(2,2),padding='same'))#2da capa convolucion
    net.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))#2do filtro max pool
    #Conexion de capas completas
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(256, activation='relu'))
    net.add(tf.keras.layers.Dropout(0.5))
    net.add(tf.keras.layers.Dense(9,activation='softmax'))
    net.compile(optimizer='adam', loss='categorical_crossentropy',metrics='accuracy')
    net.fit(entrena, epochs=100, batch_size=32, validation_data=prueba)
    return net

entrenamiento, test = cargarDataset()
net= ArquitecturaEntrenamiento(entrenamiento,test)
guardarEntrenamiento(net)









