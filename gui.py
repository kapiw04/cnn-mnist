import tkinter as tk
import torch 
from cnn import model
from datasets import transform_grayscale
from torchvision.transforms import ToPILImage
from datasets import val_dataset_grayscale
import numpy as np

canvas_tensor = torch.zeros(28, 28)
model.load_state_dict(torch.load("mnist_cnn.pth"))

predicted = None
initial_touch_point = None

def whatModelSees():
    image = ToPILImage()(canvas_tensor.unsqueeze(0))
    model_input = transform_grayscale(image).unsqueeze(0)
    image = ToPILImage()(model_input.squeeze(0))
    image.show()

def showExample():
    image, _ = val_dataset_grayscale[0]
    image = ToPILImage()(image)
    image.show()

def update_label():
    output_label.config(text=f"Predicted: {predicted}")
    
def update_tensor(x, y, value):
    canvas_tensor[y, x] = value
    image = ToPILImage()(canvas_tensor.unsqueeze(0))
    model_input = transform_grayscale(image).unsqueeze(0)
    model_output = model(model_input)
    global predicted
    predicted = model_output.argmax(1).item()
    update_label()
    
def reset_initial_touch_point(event=None):
    global initial_touch_point
    initial_touch_point = None

def draw(event):
    x, y = (event.x // 21, event.y // 21)
    canvas.create_rectangle(x*21, y*21, (x+1)*21, (y+1)*21, fill="white", outline="white")
    update_tensor(x, y, 1) 
    
def erase(event):
    x, y = (event.x // 21, event.y // 21)
    canvas.create_rectangle(x*21, y*21, (x+1)*21, (y+1)*21, fill="black", outline="black")
    update_tensor(x, y, 0)

def clear():
    canvas.delete("all")
    canvas_tensor.zero_()
    update_label()

app = tk.Tk()
app.title("28x28 Drawing Canvas")
app.geometry("600x700")

frame = tk.Frame(app, width=28*21, height=28*21)
frame.pack_propagate(False)  # Prevent the frame from resizing
frame.pack()

canvas = tk.Canvas(frame, width=28*21, height=28*21, bg="black")
canvas.pack()
clear_button = tk.Button(app, text="Clear", command=clear)
clear_button.pack()
canvas.bind("<B1-Motion>", draw)
canvas.bind("<B3-Motion>", erase)
canvas.bind("<ButtonRelease-1>", reset_initial_touch_point)

font = ('Helvetica', 24)
output_label = tk.Label(app, text="Predicted: ", font=font)
output_label.pack()

app.mainloop()
# app.after(5000, whatModelSees)
# app.after(5000, showExample)