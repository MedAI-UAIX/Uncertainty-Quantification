import numpy as np
import PIL.Image as Image
import os
import torch
from Models import Thynet
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
            ])

model = Thynet()
model1 = Thynet().resnet
model2 = Thynet().resnext
model3 = Thynet().desnet
model.eval()
model1.eval()
model2.eval()
model3.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)

preds = []
entropy = []
for image in tqdm(df['Image Path']):
  image_path = os.path.join(root, image)
  image = Image.open(image_path)
  image =transform(image).to(device).unsqueeze(0)
  pred =np.squeeze(model(image).softmax(-1).cpu().detach().numpy())
  pred1 =F.softmax(model.resnet(image).squeeze(),dim=0)
  pred2 =F.softmax(model.resnext(image).squeeze()，dim=0)
  pred3 =F.softmax(model.desnet(image).squeeze(),dim=0)
  loss1 = F.cross_entropy(pred1.unsqueeze(0), pred2.unsqueeze(0))
  Loss2 =F.cross_entropy(pred1.unsqueeze(0)，pred3.unsqueeze(0))
  Loss3 =F.cross_entropy(pred2.unsqueeze(0)，pred3.unsqueeze(0))
  loss =(loss1+ loss2 + loss3)/3
  preds.append(pred[1])entropy.append(loss.cpu().detach().numpy())
  entropy.append(loss.cpu().detach().numpy())
