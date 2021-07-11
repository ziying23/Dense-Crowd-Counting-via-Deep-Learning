import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfile, askopenfilename
import cv2
import numpy as np
import time
from PIL import Image, ImageTk
import os
import setup.model as modellib
from setup import utils, model
import torch
from matplotlib import cm as c
import cudatools as cuda
import torch
import torch.nn as nn
import torch.optim as optim

# creating main application window
root = tk.Tk()
root.geometry("720x1000") # size of the top_frame
root.title("People Counting Version 1.0")
root.resizable(width=False,height=False)
root.iconbitmap("icon.ico")


#  Frame 
logo_frame = Frame(root,bd=10)
logo_frame.pack()
top_frame = Frame(root, bd = 10)
top_frame.pack()
middle_frame = Frame(root, bd =10)
middle_frame.pack()
bottom_frame = Frame(root, bd = 10)
bottom_frame.pack()
notification_frame = Frame(root, bd = 10)
notification_frame.pack()
info_frame=Frame(root,bd=10)
info_frame.pack()

print('Torch', torch.version, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'))

"""User Defined Function"""
def load_weights():
    dialog_var.set("Loading weight")
    weight_path = "C:/Users/Dell/PycharmProjects/Testing/model_best.pth.tar"
    # torch.load(weight_path)

    torch.load(weight_path, map_location=torch.device('cpu'))
    print("yes1")
    dialog_var.set("Weight Uploaded")

# open a image file from hard-disk
def open_image(initialdir='/'):
    global file_path
    file_path = askopenfilename(initialdir=initialdir, filetypes = [('Image File', '*.*')])
    print(file_path)
    dialog_var.set("Waiting to load the image")
    img_var.set(file_path)
    image = Image.open(file_path)
    image = image.resize((320,240)) # resize image to 32x32
    photo = ImageTk.PhotoImage(image)
    img_label = Label(middle_frame, image=photo, padx=10, pady=10)
    img_label.image = photo # keep a reference!
    img_label.grid(row=3, column=1)
    return file_path

def load_image():
    dialog_var.set("Loading image")
    path = img_entry.get()
    global imgs
    imgs = cv2.imread(path)
    imgs = np.asarray(imgs,dtype="int32")
    print(imgs.shape)
    dialog_var.set("Image successful loaded, waiting to start counting")
    return

    # Test Image
def test_image():
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])
    net = modellib.apdn()
    print(net)

    # Specify a path to save to
    PATH = "C:/Users/Dell/PycharmProjects/Testing/model_best.pth.tar"
    img_path = img_entry.get()
    print(img_path)

    # Save
    torch.save(net.state_dict(), PATH)

    # Load
    device = torch.device('cpu')
    myModel = modellib.apdn()
    myModel.load_state_dict(torch.load(PATH, map_location=device))

    img = transform(Image.open(img_path).convert('RGB'))
    output = net(img.unsqueeze(0))
    # print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
    result_text = "Predicted Count: " + str(int(output.detach().cpu().sum().numpy()))
    test_result_var.set(result_text)
    dialog_var.set("Done counting")

def info_system():
    tk.messagebox.showinfo("Objective of People Counting System",
                           "This system is to accurately determine the number of people in aa image, and make the counting more faster as compared to traditional techniques.")

def close():
    root.destroy()

"""Logo Frame"""
logo = tk.PhotoImage(file="Logo_people.png")
photo = Label(logo_frame,image=logo)
photo.pack()

"""  Top Frame  """
# ##### H5 #################
btn_h5_confirm = ttk.Button(top_frame, text='Load Weights',  command = load_weights)
btn_h5_confirm.grid(row=2, column=2)
#######   IMAGE input #######
btn_img_fopen = ttk.Button(top_frame, text='Browse Image',  command =lambda: open_image(img_entry.get()))
btn_img_fopen.grid(row=7, column=1)
img_var = StringVar()
img_var.set("/")
img_entry = ttk.Entry(top_frame, textvariable=img_var, width=40)
img_entry.grid(row=7, column=2)
btn_img_confirm = ttk.Button(top_frame, text='Load Image',  command = load_image )
btn_img_confirm.grid(row=7, column=4)

""" middle Frame  """
ml = Label(middle_frame,bg="gray", fg="white", text="Browse Image Show Below").grid(row=1, column=1)

################ Test Image butttom #################
btn_test = ttk.Button(bottom_frame, text='Test Image',  command = test_image)
btn_test.pack()

test_result_var = StringVar()
test_result_var.set("Your result will be shown here")
test_result_label = Label(bottom_frame,font=40, height=3, textvariable=test_result_var, bg="white", fg="purple").pack()

"""" Notification frame """
# Define Text
dialog_var = StringVar()
dialog_var.set("Welcome to People Counting Version 1.0")
############# Label frame #############
labelframe1 = LabelFrame(notification_frame, text="Notification Box", bg="yellow")
labelframe1.pack()
toplabel = Label(labelframe1,font=30, height=2, textvariable=dialog_var, fg="black", bg="light sky blue")
toplabel.pack()

"""Info Frame"""
btn_info_system = ttk.Button(info_frame, text='Objective of People Counting', command = info_system)
btn_info_system.pack()
btn_close = ttk.Button(info_frame,text='Close',command = close)
btn_close.pack()

################# Entering the event mainloop ###############
top_frame.mainloop()