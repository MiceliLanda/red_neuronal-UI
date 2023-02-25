import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter import filedialog
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

nn = load_model('./data/aprendido.h5')
nn.load_weights('./data/pesos.h5')

def Predecir(imagen):
    im = tf.keras.preprocessing.image.load_img(imagen,target_size=(150,150))
    im = tf.keras.preprocessing.image.img_to_array(im)
    im = np.expand_dims(im,axis=0)
    arr = nn.predict(im)
    res = arr[0]
    img = np.argmax(res)
    
    if img == 0:
        return("FLOR DE BIGOTILLO")
    elif img == 1:
        return("FLOR  DE CAMPANA")
    elif img == 2:
        return("FLOR GERANIO")
    elif img == 3:
        return("FLOR DE GIRASOL")
    elif img == 4:
        return("FLOR  MARGARITA")
    elif img == 5:
        return("FLOR MAÑANITA")
    elif img == 6:
        return("ROSA COLOR ROSA")
    elif img == 7:
        return("ROSA CHINA")
    elif img == 8:
        return("FLOR VIOLETA")
    else: return(' No existe')

def inicio():
    window = Tk()
    window.title('Clasificador de imágenes')
    window.geometry("400x200+300+300")
    window.columnconfigure(0, weight=1)      
    window.rowconfigure(0,weight=1)
    
    labelFile = Label(window,text="Abrir")
    labelFile.grid(column=0,row=0)
    
    def getImage():
        files = filedialog.askopenfile(initialdir='./',filetypes=(('jpeg files','*.jpg'),("all files","*.*")))
        name = files.name.split("/")

        labelNameFile.configure(text=name[-1])
        
    def Calculo():
        if(labelNameFile.cget("text")==''):
            lblRes.configure(text="No se ha cargado una imagen")
        else:
            prediccion = Predecir(labelNameFile.cget("text"))
            lblRes.configure(text='La imagen pertenece a : '+ prediccion)
        
    btnUpload = Button(window, text='Abrir archivo',bg='black',fg='white',command=getImage)
    btnUpload.grid(column=0,row=0)
    
    labelNameFile = Label(window, text="")
    labelNameFile.grid(column=0, row=1,sticky="nsew")
    
    btnCheck = Button(window,text="Predecir",bg='green',fg='white',command=Calculo)
    btnCheck.grid(column=0,row=2,sticky="nsew")
    
    lblRes = Label(window,text="", font=("Arial", 12,"bold"))
    lblRes.grid(column=0,row=1,sticky="nsew")
    
    window.mainloop()

inicio()