from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox 

import cv2
from PIL import Image
from PIL import ImageTk
import numpy 
import imutils
import numpy as np
import time
import os
import csv
import datetime as dt
from tkcalendar import Calendar, DateEntry
from keras.models import load_model

import mysql.connector
mysqldb = mysql.connector.connect(host="localhost", user="root", password="", database="faceattendance")

##face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
##cap = cv2.VideoCapture(0)
##
##
##def cam_on():
##    cnt=0
##    cur_path = os.getcwd()
##    path=cur_path + "\\test\\"
##    ij, frame = cap.read()            
##    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
##    resized_image1 = cv2.resize(cv2image,(750,510))
##    resized_image = cv2.resize(frame,(750,510))
##    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
##    faces = face_detector.detectMultiScale(gray, 1.3, 5)
##    for (x,y,w,h) in faces:
##        cv2.rectangle(resized_image1, (x,y), (x+w,y+h), (255,255,255), 2)
##        imgr=resized_image[y:y+h,x:x+w]
##        cnt+=1
##        cv2.putText(resized_image1,'FACE COUNT:' + str(cnt) , (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_4)
##        imgr=cv2.resize(imgr, (100, 100))
##        model = load_model('traincnn.h5')
##        cropped_img = np.expand_dims(np.array(imgr), axis=0)
##        prediction = model.predict(cropped_img)
##        maxindex = int(np.argmax(prediction))
##        print(maxindex)
##        
##
##    img = Image.fromarray(resized_image1)
##    imgtk = ImageTk.PhotoImage(image=img)
##    lmain.imgtk = imgtk
##    lmain.configure(image=imgtk)
##    lmain.after(100, cam_on)
##
##def cam_off():
##     cap.release()
##     root.destroy()
##     exec(open('robo.py').read())



def Students_details():
    root.destroy()
    import details

def get_date():

    def cal_done():
        top.withdraw()
        root.quit()
    root = Tk()
    root.withdraw() 
    top = Toplevel(root)
    cal = Calendar(top,font="Arial 14", selectmode='day',cursor="hand1")
    cal.pack(fill="both", expand=True)
    ttk.Button(top, text="ok", command=cal_done).pack()
    selected_date = None
    root.mainloop()
    return cal.selection_get()


count=0
def increment():
    global count
    count = count+1

cap = cv2.VideoCapture(0)#type ip addres of ipcam app

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
camsts=0
def cam_on():    
    global camsts
    ij, frame = cap.read()
    if(camsts==1):
        cap.release()
    else:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
 
        resized_image = cv2.resize(cv2image,(750,510))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(resized_image, (x,y), (x+w,y+h), (255,0,0), 2)
        img = Image.fromarray(resized_image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(5, cam_on)




    

def Take_Photo():
    global camsts
    cnt=0
    cur_path = os.getcwd()
    path=cur_path + "\\test\\"   
    ij, frame = cap.read()
    if(camsts==1):
        cap.release()
    else:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        resized_image1 = cv2.resize(cv2image,(750,510))
        resized_image = cv2.resize(frame,(750,510))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(resized_image1, (x,y), (x+w,y+h), (0,0,255), 2)
            increment()
            cnt+=1
            cv2.putText(resized_image1,'IMAGE COUNT:' + str(count+1) + '  FACE COUNT:' + str(cnt),(30, 30),cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1, cv2.LINE_4)
            fimg=cv2.resize(resized_image[y:y+h,x:x+w],(100,100))
            cv2.imwrite(path + str(count) + ".jpg",fimg)
            if count >= 2:
                cap.release()
                camsts=camsts+1
     

  
   
def View_Attendance():
    pth='C:\\PROJECT\\pro\\test\\'
    for ct in range(1,3):
        pth1=pth + str(ct) +'.jpg'
        print(pth1)
        model = load_model('traincnn.h5')
        image = Image.open(pth1).convert("RGB")
        image = image.resize((100,100))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        pred = model.predict([image])[0]
        classes = np.argmax(pred, axis=-1)
        print(classes+1)
        rid=classes+1
        present='P'
        absent='A'
        mycursor = mysqldb.cursor(buffered=True)
        mycursor.execute("SELECT * FROM studentdetails WHERE id = '%d'" % (rid))
        myresult = mycursor.fetchall()
        print(myresult)
        dtr = dt.datetime.now().strftime('%d-%m-%Y')
        dtr = str(dtr)
        for ele in myresult:
            mycursor.execute("INSERT INTO student_attendance(Sl_no, date, Name, USN, Status) VALUES (%s, %s, %s, %s, %s)",(ele[0],dtr,ele[1],ele[2],present))
            listBox.insert("", "end", values=(ele[0],dtr,ele[1],ele[2],present))
            mysqldb.commit()  
      




def export():

    mycursor = mysqldb.cursor(buffered=True)
    mycursor.execute("SELECT * FROM student_attendance")
    records = mycursor.fetchall()
    print(records)
    with open("new_file.csv","w") as file:
        for row in records:
            csv.writer(file).writerow(row)
    mycursor.close()

    
def close_page():
    root.destroy()

root = Tk() 
root.title('SMART CCTV SYSTEM')
root.geometry('1920x1080')
root.configure(background='lightgray')


c1 = Canvas(root,bg='red',width=1530,height=790)
c1.place(x=2,y=2)
l1=Label(root,text='SMART CCTV SYSTEM',foreground="white",background='red',font =('Verdana',25))
l1.place(x=540,y=35)


c2 = Canvas(root,bg='white',width=1045,height=675)
c2.place(x=2,y=115)
b1=Button(root,borderwidth=1, relief="flat",text="TAKE ATTENDENCE", font="verdana 15", bg="red", fg="white",command=Take_Photo)
b1.place(height=150,width=245,x=6,y=121)
b2=Button(root,borderwidth=1, relief="flat",text="RECORDED", font="verdana 15", bg="red", fg="white",command=cam_on)
b2.place(height=150,width=245,x=6,y=273)
b3=Button(root,borderwidth=1, relief="flat",text="STUDENT DETAILS", font="verdana 15", bg="red", fg="white",command=Students_details)
b3.place(height=150,width=245,x=6,y=426)
b4=Button(root,borderwidth=1, relief="flat",text="VIEW ATTENDENCE", font="verdana 15", bg="red", fg="white",command=View_Attendance)
b4.place(height=150,width=245,x=6,y=578)


c3 = Canvas(root,bg='white',width=1270,height=620)
c3.place(x=258 ,y=115)
dts = dt.datetime.now().strftime('%d-%m-%Y')
tms = dt.datetime.now().strftime('%H:%M %p')
label = Label(root, text="DATE :", foreground="red",background='white',font =('Verdana',20)).place(height=30,width=100,x=275,y=145)
label = Label(root, text=dts, foreground="red",background='white',font =('Verdana',20)).place(height=30,width=160,x=390,y=145)
label = Label(root, text="TIME :", foreground="red",background='white',font =('Verdana',20)).place(height=30,width=100,x=790,y=145)
label = Label(root, text=tms, foreground="red",background='white',font =('Verdana',20)).place(height=30,width=160,x=880,y=145)
c4 = Canvas(root,bg='red',width=750,height=510) 
c4.place(x=275,y=200) 
lmain = Label(root,bg='red')
lmain.place(x=280,y=205)


c5 = Canvas(root,bg='white',width=480,height=650)
c5.place(x=1050,y=115)
labela = Label(root,text ="REPORT ",foreground="red",background='white',font =('Verdana', 15))
labela.place(x=1200,y=135)

cols = ('Sl_no', 'Date', 'Name', 'USN','Status')
listBox = ttk.Treeview(root, columns=cols, show='headings')
hsb = ttk.Scrollbar(root, orient="horizontal", command=listBox.xview)
hsb.place(height=20,width=400,x=1080,y=200)


for col in cols:
    listBox.heading(col, text=col)    
listBox.place(height=400,width=400,x=1080,y=230)
listBox.configure(xscrollcommand=hsb.set)

b5 = Button(root,borderwidth=1,relief="flat",text="EXPORT",font="verdana 12 bold",bg="red",fg="white", width=15, command=export)
b5.place(height=70,width=180,x=1082,y=650)

b6 = Button(root,borderwidth=1,relief="flat",text="CLOSE",font="verdana 12 bold",bg="red",fg="white", width=15, command=close_page)
b6.place(height=70,width=180,x=1300,y=650)

c6 = Canvas(root,bg='red',width=1530,height=65)
c6.place(x=2,y=730)

mainloop()
