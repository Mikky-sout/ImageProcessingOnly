from tkinter import *
import tkinter as tk
from tkinter import filedialog
import os
import PIL.Image
import matplotlib.pyplot as plt
from PIL import Image,ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib
import cv2



def showimage():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(),title="Select Image File",filetypes=(("JPG File","*.jpg"),("PNG file","*.png")))
    global arr_img
    re = ''
    result.configure(text=re)
    arr_img = cv2.imread(fln)
    image = PIL.Image.open(fln)

    image = image.resize((600,600))
    img = ImageTk.PhotoImage(image)

    lbl.configure(image=img)
    lbl.image = img



def predict():
    face = arr_img
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (150, 150))
    face = img_to_array(face)

    face = preprocess_input(face)
    face.shape = ((1, 150, 150, 3))
    face = np.array(face, dtype="float32")
    # print(face.shape)
    preds = model.predict(face)
    pred_max = np.argmax(preds, axis=1)
    re = ''
    if pred_max[0] == 0:
        re = 'Wearing Mask'
        pre = round(float(preds[0][0]) * 100,2)
        re += ' (' + str(pre) + ' %)'
        result.config(fg='#0f0')
    else:
        re = "Don't wearing Mask"
        pre = round(float(preds[0][1])*100,2)
        re += ' (' + str(pre) + ' %)'
        result.config(fg='#f00')

    result.configure(text=re)





model = load_model("Project.model")
root = Tk()
# faces = np.array(dtype="float32")
frm = Frame(root)
frm.pack(side=BOTTOM , padx = 15 ,pady=15)

lbl = Label(root,borderwidth=10)
lbl.pack()

# menu = Frame(root)
# menu.pack(side=RIGHT , padx= 20,pady=20)
#
result = Label(root,highlightcolor="#000")
result.pack()

btn = Button(frm , text="Browse Image",command=showimage)
btn.pack()
btn.config(font=('Helvetica bold',24))

btn1 = Button(frm , text="Predict",command=predict)
btn1.pack()
btn1.config(font=('Helvetica bold',24))
result.config(font=('Helvetica bold',40))


root.title("Mask Detector")
root.geometry("1000x800")
root.minsize(1000, 900)
root.maxsize(1000, 900)
root.mainloop()