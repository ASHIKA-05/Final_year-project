from tkinter import *
from tkinter import messagebox 
import cv2
from PIL import Image
from PIL import ImageTk


def login_data():
    usnm=e1.get()
    pwd=e2.get()
    if(usnm=='admin' and pwd=='admin'):
        messagebox.showinfo("Login", "Login Successful")
        root.destroy()
        #exec(open('main.py').read())
        import main
    else:
        messagebox.showerror("Login", "Wrong Password")


def data_reset():
     e1.delete(0,'end')
     e2.delete(0,'end')


root = Tk() 
root.title('SMART CCTV SYSTEM')
root.geometry('1920x1080')
root.configure(background='lightgray')

c2 = Canvas(root,bg='white',width=673,height=790)
c2.place(x=2,y=2)

lmain = Label(root)
lmain.place(x=2,y=60)

img = cv2.imread('1.jpg')
resized_image1 = cv2.resize(img,(673,690))
img = Image.fromarray(resized_image1)
imgtk = ImageTk.PhotoImage(image=img)
lmain.imgtk = imgtk
lmain.configure(image=imgtk)

c5 = Canvas(root,bg='red',width=850,height=790)
c5.place(x=679,y=2)

c6 = Canvas(root,bg='white',width=450,height=350) 
c6.place(x=880,y=180)

labela = Label(root,text ="LOGIN",foreground="red",background='white',font =('Verdana',20))
labela.place(x=1060,y=200)

e1 = Entry(root,borderwidth=2, relief="groove",foreground="red",font =('Verdana', 15))
e1.place(height=60,width=350,x=930,y=280)

e2 = Entry(root,show='*',borderwidth=2, relief="groove",foreground="red",font =('Verdana', 15))
e2.place(height=60,width=350,x=930,y=360)

b1=Button(root,borderwidth=2, relief="flat",text="LOGIN", font="verdana 15", bg="red", fg="white",command=login_data)
b1.place(height=60,width=150,x=930,y=450)

b2=Button(root,borderwidth=2, relief="flat",text="RESET", font="verdana 15", bg="red", fg="white",command=data_reset)
b2.place(height=60,width=150,x=1130,y=450)

mainloop()
