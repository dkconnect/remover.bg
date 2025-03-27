from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from io import BytesIO

app = FastAPI()

# Load U²-Net model (lightweight version: u2netp.pth)
model_path = "u2netp.pth"  # Download from U²-Net GitHub: https://github.com/xuebinqin/U-2-Net
device = torch.device("cpu")  # Free tiers don’t offer GPUs
net = torch.load(model_path, map_location=device)
net.eval()

# Preprocessing function (optimized for speed)
def preprocess_image(image: Image.Image, size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

# Postprocessing function
def postprocess_output(output, original_size):
    mask = output.squeeze().cpu().data.numpy()
    mask = Image.fromarray(mask).resize(original_size, Image.BILINEAR)
    mask = np.array(mask)
    mask = (mask > 0.5).astype(np.uint8) * 255  # Binary mask
    return mask

# Remove background
def remove_background(image: Image.Image):
    # Resize image to reduce memory usage (max 500px on larger side)
    max_size = 500
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    original_size = image.size
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = net(input_tensor)[0]  # Use first output of U²-Net
    
    mask = postprocess_output(output, original_size)
    
    # Apply mask
    image_np = np.array(image.convert("RGBA"))
    mask_3d = np.stack([mask] * 4, axis=2)
    result = np.where(mask_3d == 255, image_np, [0, 0, 0, 0]).astype(np.uint8)
    
    return Image.fromarray(result)

@app.post("/remove-background/")
async def remove_bg_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # Process image
    result_image = remove_background(image)
    
    # Return as PNG stream
    output_buffer = BytesIO()
    result_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    
    return StreamingResponse(output_buffer, media_type="image/png")

@app.get("/")
async def root():
    return {"message": "bg.remove API is running!"}
