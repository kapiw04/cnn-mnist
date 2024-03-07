import tkinter as tk
import torch
import torch.nn.functional as F
from cnn import model
from datasets import transform_grayscale
from torchvision.transforms import ToPILImage
from datasets import val_dataset_grayscale

canvas_tensor = torch.zeros(128, 128)
model.load_state_dict(torch.load("mnist_cnn.pth"))

predicted = None

def whatModelSees():
    resized_tensor = F.interpolate(canvas_tensor.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False).squeeze(0)
    image = ToPILImage()(resized_tensor)
    model_input = transform_grayscale(image).unsqueeze(0)
    image = ToPILImage()(model_input.squeeze(0))
    image.show()

def showExample():
    image, _ = val_dataset_grayscale[0]
    image = ToPILImage()(image)
    image.show()

def update_label():
    output_label.config(text=f"Predicted: {predicted if predicted is not None else 'None'}")
    print(f"Predicted: {predicted if predicted is not None else 'None'}")
    
def update_tensor(x, y, value):
    for i in range(max(0, x*4-2), min(128, x*4+2)):
        for j in range(max(0, y*4-2), min(128, y*4+2)):
            canvas_tensor[j, i] = value
    resized_tensor = F.interpolate(canvas_tensor.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False).squeeze(0)
    model_input = transform_grayscale(ToPILImage()(resized_tensor)).unsqueeze(0)
    model_output = model(model_input)
    global predicted
    predicted = model_output.argmax(1).item()
    update_label()
    
def draw(event):
    x, y = (event.x // 16, event.y // 16) # Adjust for 128x128 canvas
    canvas.create_rectangle(x*16, y*16, (x+1)*16, (y+1)*16, fill="white", outline="white")
    update_tensor(x, y, 1)

def erase(event):
    x, y = (event.x // 16, event.y // 16) # Adjust for 128x128 canvas
    canvas.create_rectangle(x*16, y*16, (x+1)*16, (y+1)*16, fill="black", outline="black")
    update_tensor(x, y, 0)

def clear():
    canvas.delete("all")
    canvas_tensor.zero_()
    update_label()

app = tk.Tk()
app.title("128x128 Drawing Canvas")
app.geometry("800x800") 

frame = tk.Frame(app, width=128*16, height=128*16, bg="white")
frame.pack_propagate(False)  
frame.pack()

canvas = tk.Canvas(frame, width=128*16, height=128*16, bg="black")
canvas.pack()
clear_button = tk.Button(app, text="Clear", command=clear)
clear_button.pack()
canvas.bind("<B1-Motion>", draw)
canvas.bind("<B3-Motion>", erase)

font = ('Helvetica', 24)
output_label = tk.Label(app, text="Predicted: None", font=font)  
output_label.pack(side=tk.TOP, pady=10) 

app.after(5000, whatModelSees)

app.mainloop()