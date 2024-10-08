'''
Created on 4 Jun 2020

@author: Nur Raudzah Binti Abdullah
'''
#Import required libraries && necessary modules
from tkinter import *
import tkinter.font as tkFont
from tkcalendar import Calendar,DateEntry
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
import csv
import datetime
from sklearn.preprocessing import MinMaxScaler
from tkinter import *
from PIL import ImageTk, Image
import os
#2ndInterface
class Window2(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.root3 = Toplevel(master)
        self.root3.title("Air Pollution Forecasting System")
        self.root3.geometry('750x600')
        self.root3.configure(bg='#C2CBAD')
        #set font style
        fontStyle = tkFont.Font(family="Garamond", size=40)
        fontStyle2 = tkFont.Font(family="Garamond", size=18)
        fontStyle3 = tkFont.Font(family="Arial Black", size=20)
        #create heading
        text9 = Label(self.root3, text = "Air Pollution Forecasting", font=fontStyle, bg="#83992A")
        text9.pack(fill=X)
        #create heading2
        t = Label(self.root3, font=fontStyle3, bg='#C2CBAD', fg="black", text="Result and Analysis of Air Pollution"+"\n on {}".format(app1.dataDate))
        t.place(x=120,y=70)
        #result--output
        t2 = Label(self.root3, text="Prediction Result(API)",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        t2.place(x=50,y=170,height=40, width=400)
        t3 = Label(self.root3, text="Air Quality Status",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        t3.place(x=50,y=220,height=40, width=400)
        e = Label(self.root3, font=14, borderwidth=2, relief="ridge",text=(", " . join(app1.sarr)))
        e.place(x=500,y=170, height=40, width=200)
        e1 = Label(self.root3, font=14, borderwidth=2, relief="ridge",text=app1.y)
        e1.place(x=500,y=220, height=40, width=200)
        load = Image.open("C:/api.png")
        render = ImageTk.PhotoImage(load)
        img = Label(self.root3, image=render)
        img.image = render
        img.place(x=220, y=270)
        #create butt
        button4 = Button(self.root3, text="BACK", font=fontStyle2, bg="#83992A", fg="white", borderwidth=4, relief="raised", command=self.root3.destroy)
        button4.place(x=180,y=520, height=33, width=120)
        button5 = Button(self.root3, text="EXIT", font=fontStyle2, bg="#83992A", fg="white", borderwidth=4, relief="raised", command=lambda x=master:x.destroy())
        button5.place(x=440,y=520, height=33, width=120)
#1stInterface
class Window1(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master
        self.configure(bg='#C2CBAD')
        self.master.title("Air Pollution Forecasting System")
        def number():
            try:
                int(entry2.get())
                entry2.config(text="")
            except ValueError:
                entry2.config(text="Invalid Input!Input must be a number") 
        #set font style
        fontStyle = tkFont.Font(family="Garamond", size=40)
        fontStyle2 = tkFont.Font(family="Garamond", size=18)
        #create header
        text1 = Label(self, text = "Air Pollution Forecasting",font=fontStyle, bg="#83992A", fg="white")
        text1.pack(fill=X)
        #create label for inputs
        text2 = Label(self, text="Date",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        text2.place(x=50,y=100,width=400)
        text3 = Label(self, text="SO2-Sulfur dioxide(ppm)",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        text3.place(x=50,y=200,width=400)
        text4 = Label(self, text="NO2-Nitrogen dioxide(ppm)",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        text4.place(x=50,y=300,width=400)
        text5 = Label(self, text="O3-Ozone(ppm)",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        text5.place(x=50,y=150,width=400)
        text6 = Label(self, text="CO-Carbon monoxide(ppm)",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        text6.place(x=50,y=250,width=400)
        text7 = Label(self, text=u"PM10-Particulate matter under 10(\u00B5g/m3)",font=fontStyle2, bg="#D6E49C", fg="black", borderwidth=2, relief="groove")
        text7.place(x=50,y=350,width=400)
        
        #create input entry 
        self.var=StringVar(self)
        cal = DateEntry(self,textvariable=self.var,year=2020,date_pattern='dd/mm/y',font=fontStyle2,
                 selectbackground="#83992A",
             selectforeground='black',
                 normalbackground='white',
                 normalforeground= "#83992A",
                 background="#83992A",
                 foreground='black',
                 bordercolor="#e6efc3",
                 othermonthforeground="#cdde87",
                 othermonthbackground='white',
                 othermonthweforeground="#cdde87",
                 othermonthwebackground='white',
                 weekendbackground="#eef4d7",
                 weekendforeground="#83992A",
                 headersbackground="#dde9af",
                 headersforeground="#56641b")
        cal.place(x=500,y=100, height=33, width=200)
        self.ozone = StringVar(self)
        entry2 = Entry(self, font=14, borderwidth=2, relief="ridge", textvariable=self.ozone)
        entry2.place(x=500,y=150, height=33, width=200)
        self.SO2 = StringVar(self)
        entry3 = Entry(self, font=14, borderwidth=2, relief="ridge", textvariable=self.SO2)
        entry3.place(x=500,y=200, height=33, width=200)
        self.CO = StringVar(self)
        entry4 = Entry(self, font=14, borderwidth=2, relief="ridge", textvariable=self.CO)
        entry4.place(x=500,y=250, height=33, width=200)
        self.NO2 = StringVar(self)
        entry5 = Entry(self, font=14, borderwidth=2, relief="ridge", textvariable=self.NO2)
        entry5.place(x=500,y=300, height=33, width=200)
        self.PM10 = StringVar(self)
        entry6 = Entry(self, font=14, borderwidth=2, relief="ridge", textvariable=self.PM10)
        entry6.place(x=500,y=350, height=33, width=200)
       
        #create button
        button1 = Button(self, text="PREDICT", font=fontStyle2, bg="#83992A", fg="white", borderwidth=4, relief="raised", command=lambda x=master:[number(),self.getData(),self.predict(), Window2(x)])
        button1.place(x=130,y=450, height=33, width=120)
        button2 = Button(self, text="CLEAR", font=fontStyle2, bg="#83992A", fg="white", borderwidth=4, relief="raised", command=self.clearFunc)
        button2.place(x=315,y=450, height=33, width=120)
        button3 = Button(self, text="EXIT", font=fontStyle2, bg="#83992A", fg="white", borderwidth=4, relief="raised", command=lambda y=master:y.destroy())
        button3.place(x=500,y=450, height=33, width=120)
        
    def getData(self):
        self.dataDate=self.var.get()
        self.oz=self.ozone.get()
        self.su=self.SO2.get()
        self.co=self.CO.get()
        self.no=self.NO2.get()
        self.pm=self.PM10.get()
        with open('FileData.csv', 'w') as f:
            w = csv.writer(f,quoting=csv.QUOTE_ALL)
            w.writerow([self.oz,self.su,self.co,self.no,self.pm]) # write Date/Time and the value
        
        #Make prediction on new data        
    def predict(self):
        #Load New Data
        real_df = pd.read_csv('C:/Users/User/eclipse-workspace/AirPollutionForecastingInKualaTerengganuUsingANN/src/FileData.csv',header=None)
        real_df.head()
       
        #Make prediction
        from keras.models import load_model
        saved_model = load_model('best13_model.h5')
        self.pred = saved_model.predict(real_df)
        print(self.pred)
        
        real_df['API'] = np.round(self.pred,0)
        self.final_result = real_df['API']
        self.sarr = [str(a) for a in self.final_result]
        print("API:",", " . join(self.sarr))
        print(np.round(self.final_result, 0))
        #if-else statement for air quality status 
        if np.round(self.pred,0) >=301:
            self.y = "Hazardous"
        
        elif np.round(self.pred,0) >=201:
            self.y = "Very Unhealthy"
        
        elif np.round(self.pred,0) >=101:
            self.y = "Unhealthy"
            
        elif np.round(self.pred,0) >=51:
            self.y = "Moderate"
        
        else:
            self.y = "Good"
       
        print("Air Quality Status:",self.y)
        
        now = datetime.datetime.now()
        real_df['Now'] = now
        print("Date and Time:",now)
        #Insert final_result into excel file
        real_df.to_csv('Prediction.csv', mode='a', header=False)

    def clearFunc(self):
        self.var.set("")  
        self.ozone.set("") 
        self.SO2.set("")  
        self.CO.set("") 
        self.NO2.set("") 
        self.PM10.set("") 
     
root1 = Tk()
root1.geometry('750x550')
app1 = Window1(root1)
app1.pack(fill="both", expand=True)
root1.mainloop()