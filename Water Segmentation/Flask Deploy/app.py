from flask import Flask, render_template, request, redirect
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import rasterio
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, features=[32, 64, 128]):
        super(UNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        for feature in features:
            self.downs.append(self.conv_block(in_channels, feature))
            in_channels = feature

        self.bottleneck = self.conv_block(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self.conv_block(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            x = torch.cat((skip,x),dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)


device = torch.device("cpu")

model = UNet(in_channels=12,out_channels=1)
model.load_state_dict(torch.load(r"C:\Users\walaa\Desktop\Cellula CV\Water Segmentation\From Scratch\model\water_unet5.pth",map_location=device))
model.to(device)
model.eval()
print("Model loaded!")


means = np.array([396.4676, 494.62097, 822.32007, 973.67523, 2090.1118, 
                  1964.051, 1351.2747, 102.739655, 141.80382, 300.7412, 
                  35.102535, 9.753329], dtype=np.float32)
stds  = np.array([270.06662, 325.97922, 418.12158, 586.70294, 1055.9846, 
                  1191.4221, 961.76245, 48.804028, 1364.981, 496.03876, 
                  20.184526, 27.758299], dtype=np.float32)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect('/')
    
    file = request.files['image']
    if file.filename == '':
        return redirect('/')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    

    with rasterio.open(filepath) as src:
        image = src.read().astype(np.float32)  #shape(12, H, W)
    
    resized_bands = []
    for band in image:
        resized = cv2.resize(band, (128, 128), interpolation=cv2.INTER_LINEAR)
        resized_bands.append(resized)
    image_np = np.stack(resized_bands)  #shape(12, 128, 128)
    
    for i in range(12):
        image_np[i] = (image_np[i] - means[i]) / (stds[i] + 1e-8)
    
    input_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)  #shape(1,12,128,128)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255 
    
    mask_image = Image.fromarray(mask)
    mask_filename = "mask_" + file.filename.replace(".tif",".png")
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
    mask_image.save(mask_path)
    
    return render_template("index.html",uploaded_image=file.filename,mask_image=mask_filename)


if __name__ == "__main__":
    app.run(debug=True)