from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import cv2
from PIL import Image
from PIL import ImageTk
import imutils

import numpy as np
import os

from sklearn.model_selection import train_test_split
from  tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,Activation

import mysql.connector
mydb = mysql.connector.connect(host="localhost",user="root",password="",database="faceattendance")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'

def testVal(inStr,acttyp):
    if acttyp == '1': 
        if not inStr.isdigit():
            return False
    return True

def age_limit(var3):
    if len(var3.get()) > 0:
        var3.set(var3.get()[:3])

def phone_limit(var5):
    if len(var5.get()) > 0:
        var5.set(var5.get()[:10])

def check(email):
    if(re.search(regex, email)):
        print("Valid Email") 
    else:
        print("Invalid Email")
        messagebox.showerror("showerror", "Invalid Email")

def submit_data():
    global path1
    path="C:\\PROJECT\\pro\\facedata"
    nm=var1.get()        
    usn=var2.get()
    ag=var3.get()
    gr=comboa1.get()
    adrs=e4.get("1.0","end")
    mbl=var5.get()
    eml=var6.get()
    usn_entry = usn
    usn=usn_entry[3:]
    try:
        path1 = os.path.join(path,str(usn))
        print(path1)
        os.mkdir(path1)        
    except(Exception):
        if (os.path.exists(path)):
            messagebox.showinfo("showinfo", "USN Number Already Exists Or USN Number is Empty")
            
    if not var1.get():
        messagebox.showinfo("showinfo", "Please Enter Name")
    elif not var2.get():
        messagebox.showinfo("showinfo", "Please Enter Age")
    elif not e4.get("1.0","end"):
        messagebox.showinfo("showinfo", "Please Enter Place")
    elif not var5.get():
        messagebox.showinfo("showinfo", "Please Enter Phone Number")
    elif (len(mbl)<10):
        messagebox.showinfo("showinfo", "Please Enter Valid Phone Number")
    elif not var6.get():
        messagebox.showinfo("showinfo", "Please Enter Email-Id")
    elif not(re.search(regex, eml)):
         messagebox.showinfo("showinfo", "Please Valid Enter Email-Id")        
    else:        
        mycursor = mydb.cursor()
        sql = "INSERT INTO studentdetails (sname, usn, age, gender, address, mobile, email) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (nm, usn, ag, gr, adrs, mbl, eml)
        mycursor.execute(sql, val)
        mydb.commit()
        messagebox.showinfo("showinfo", "Data Updates Successfully")
        


def txt_box_reset():

    acv=""
    var1.set(acv)
    var2.set(acv)
    var3.set(acv)
    var4.set(acv)
    var5.set(acv)
    var6.set(acv) 

def cam_on():
    global camsts
    ij, frame = cap.read()
    if(camsts==1):
        cap.release()
    else:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        resized_image = cv2.resize(cv2image,(500,400))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(resized_image, (x,y), (x+w,y+h), (255,0,0), 2)
        img = Image.fromarray(resized_image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(5, cam_on)


camsts=0
def cam_off():
    global camsts
    camsts=camsts+1
    

    
def take_photo():
    global path1
    #path1="C:\\PROJECT\\pro\\facedata\\17CS001"
    ij, frame = cap.read()   
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    resized_image1 = cv2.resize(cv2image,(500,400))
    resized_image = cv2.resize(frame,(500,400))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(resized_image1, (x,y), (x+w,y+h), (255,255,255), 2)
        for ic in range(1,101):
            fimg=resized_image[y:y+h,x:x+w]
            fimg=cv2.resize(fimg,(100,100))
            cv2.imwrite(path1 +'/'+ str(ic) + ".jpg", fimg)
            




def exit_page():
    root.destroy()


def Train_CNN():
     global folder_selected
     data = []
     labels = []
     classes = 2
     cur_path = os.getcwd()
     for i in range(1,classes+1):
         ia= "{0:02d}".format(i)
         path = os.path.join(cur_path,'facedata','17CS0' + str(ia))
         print(path)
         images = os.listdir(path)
         for a in images:
             try:
                 image = Image.open(path + '\\'+ a).convert("RGB")
                 image = image.resize((100,100))
                 image = np.array(image)
                 data.append(image)
                 labels.append(i)                
             except:
                 print("Error loading image")

     data = np.array(data)
     labels = np.array(labels)
     print(data.shape, labels.shape)
    
     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
     print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
     y_train = to_categorical(y_train-1, classes)
     y_test = to_categorical(y_test-1, classes)

     model = Sequential()
     model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
     model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
     model.add(MaxPool2D(pool_size=(2, 2)))
     model.add(Dropout(rate=0.25))
     model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
     model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
     model.add(MaxPool2D(pool_size=(2, 2)))
     model.add(Dropout(rate=0.25))
     model.add(Flatten())
     model.add(Dense(256, activation='relu'))
     model.add(Dropout(rate=0.5))
     model.add(Dense(classes, activation='softmax'))

     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

     epochs = 20
     history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
     loss, accuracy = model.evaluate(X_test, y_test, verbose=1)    
     accuracy=accuracy*100    
     model.save("traincnn.h5")
     print("done")

    
        
root = Tk() 
root.title('SMART CCTV')
root.geometry('1920x1080')
root.configure(background='lightgray')

var1=StringVar()
var2=StringVar()
var3=StringVar()
var5=StringVar()
var6=StringVar()




c1 = Canvas(root,bg='red',width=1525,height=80)
c1.place(x=5,y=5)
l1=Label(root,text='SMART CCTV',foreground="white",background='red',font =('Verdana',25,'bold'))
l1.place(x=630,y=25)

c2 = Canvas(root,bg='red',width=760,height=710)
c2.place(x=5,y=90)


l2=Label(c2,text="STUDENT DETAILS",foreground="white",background='red',font =('Verdana',18)).place(x=250,y=12)
l3=Label(c2, text='Name',foreground="white",background='red', font=('Verdana',12)).place(x=152,y=60)
l4=Label(c2, text='USN', foreground="white",background='red',font=('Verdana',12)).place(x=152,y=145)
l5=Label(c2, text='Age', foreground="white",background='red',font=('Verdana',12)).place(x=320,y=145)
l6=Label(c2, text='Gender', foreground="white",background='red',font=('Verdana',12)).place(x=485,y=145)
l7=Label(c2, text='Address',foreground="white",background='red', font=('Verdana',12)).place(x=152,y=225)
l8=Label(c2, text='Mobile',foreground="white",background='red', font=('Verdana',12)).place(x=152,y=365)
l9=Label(c2, text='Email-id',foreground="white",background='red', font=('Verdana',12)).place(x=152,y=445)

e1 = Entry(root,textvariable=var1,font =('Verdana',12),foreground="red",justify=LEFT)
e1.place(height=40,width=450,x=160, y=190)

e2 = Entry(root, textvariable=var2,font =('Verdana',12),foreground="red",justify=LEFT)
e2.place(height=40,width=120,x=160, y=270)

e3 = Entry(root, textvariable=var3,font =('Verdana',12),foreground="red",justify=LEFT,validate="key")
e3['validatecommand'] = (e3.register(testVal),'%P','%d')
var3.trace("w", lambda *args: age_limit(var3))
e3.place(height=40,width=120,x=325, y=270)

n1 = StringVar()
comboa1 = ttk.Combobox(root,font =('Verdana', 12),foreground = 'red',state='readonly',textvariable = n1)
comboa1['values'] = ('Male','Female') 
comboa1.current(0) 
comboa1.place(height=40,width=120,x=490, y=270)


e4 = Text(root,font =('Verdana',12),foreground="red")
e4.place(height=100,width=450,x=160, y=350)

e5 = Entry(root, textvariable=var5,font =('Verdana',12),foreground="red",justify=LEFT)
e5['validatecommand'] = (e5.register(testVal),'%P','%d')
var5.trace("w", lambda *args: phone_limit(var5))
e5.place(height=40,width=450,x=160, y=490)

e6 = Entry(root, textvariable=var6,font =('Verdana',12),foreground="red",justify=LEFT)
e6.place(height=40,width=450,x=160, y=570)

b1=Button(root,borderwidth=1,relief="flat",text ="SUBMIT",font="verdana 12 bold",bg="white",fg="red",command = submit_data)
b1.place(height=50,width=200,x=160,y=630)

b2=Button(root,borderwidth=1,relief="flat",text ="RESET",font="verdana 12 bold",bg="white",fg="red",command = txt_box_reset)
b2.place(height=50,width=200,x=410,y=630)


c3 = Canvas(root,bg='red',width=762,height=710)
c3.place(x=768,y=90)

l10=Label(c3,text="STUDENT PHOTO",foreground="white",background='red',font =('Verdana',18)).place(x=250,y=12)

c4 = Canvas(root,bg='white',width=500,height=400) 
c4.place(x=900,y=190) 

lmain = Label(root)
lmain.place(x=900,y=190)

b3=Button(root,borderwidth=1,relief="flat",text ="CAMERA ON",font="verdana 12 bold",bg="white",fg="red",command = cam_on)
b3.place(height=50,width=130,x=860,y=630)

b4=Button(root,borderwidth=1,relief="flat",text ="TAKE PHOTO",font="verdana 12 bold",bg="white",fg="red",command = take_photo)
b4.place(height=50,width=130,x=1005,y=630)

b5=Button(root,borderwidth=1,relief="flat",text ="CAMERA OFF",font="verdana 12 bold",bg="white",fg="red",command = cam_off)
b5.place(height=50,width=130,x=1155,y=630)

b6=Button(root,borderwidth=1,relief="flat",text ="TRAIN DATA",font="verdana 12 bold",bg="white",fg="red",command = Train_CNN)
b6.place(height=50,width=130,x=1305,y=630)

mainloop()

