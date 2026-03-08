import torch
import yaml
from PIL import Image
from torchvision import transforms
from model import build_model

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["fractured", "not fractured"]

# Load model
model = build_model(cfg, num_classes=2)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax(1).item()

    print("Prediction:", classes[pred])
    print("Confidence:", float(prob[0][pred]))

predict("demo_normal.jpg")