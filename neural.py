import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO

# --- Load image from URL ---
def load_img_from_url(url, max_size=400):
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)).convert('RGB')
    size = max(img.size)
    if size > max_size:
        scale = max_size / size
        img = img.resize((int(img.width*scale), int(img.height*scale)))
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225))
    ])
    return tf(img).unsqueeze(0).to(device)

# --- Gram matrix ---
def gram(x):
    b,c,h,w = x.size()
    f = x.view(c, h*w)
    return f @ f.t() / (c*h*w)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# URLs for content and style images
content_url = 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg''
style_url = 'https://upload.wikimedia.org/wikipedia/commons/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
'

content = load_img_from_url(content_url)
style   = load_img_from_url(style_url)
target  = content.clone().requires_grad_(True)

# Load VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

# Define layers
content_layers = ['21']  # relu4_2
style_layers = ['0','5','10','19','28']

# Extract features
def get_features(x):
    feats = {}
    for name,layer in vgg._modules.items():
        x = layer(x)
        if name in content_layers + style_layers:
            feats[name] = x
    return feats

# Precompute features
cont_feats = get_features(content)
style_feats = get_features(style)
style_grams = {l: gram(style_feats[l]) for l in style_layers}

# Optimization setup
opt = optim.Adam([target], lr=0.003)
style_weight, content_weight = 1e6, 1

# Optimize
for i in range(1, 501):
    opt.zero_grad()
    feats = get_features(target)
    c_loss = content_weight * torch.mean((feats['21'] - cont_feats['21'])**2)
    s_loss = sum(torch.mean((gram(feats[l]) - style_grams[l])**2)
                 for l in style_layers)
    s_loss *= style_weight / len(style_layers)
    loss = c_loss + s_loss
    loss.backward()
    opt.step()
    if i % 100 == 0:
        print(f"Step {i}, content_loss={c_loss.item():.2f}, style_loss={s_loss.item():.2f}")

# De-normalize and save
unorm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])
out = unorm(target.detach().cpu().squeeze())
out = torch.clamp(out, 0, 1)
transforms.ToPILImage()(out).save('stylized_out.jpg')
print("Saved stylized_out.jpg")

