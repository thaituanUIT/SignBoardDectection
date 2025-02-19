# Single inference

import torch
import numpy as np
import cv2
import os
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

saved_model = "C:\\Users\\USER\\Documents\\tempoRun-Remastered\\saved_model"  # Output directory of the save the model
filename = "image_002.jpg"    # Image filename
img_path = "datasets/test/" + filename

with open('labels.txt', 'r') as f:
	string = f.read()
	labels_dict = eval(string)

def get_model(num_classes):

	# Load an pre-trained object detectin model (in this case faster-rcnn)
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
		weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
	)

	# Number of input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# Replace the pre-trained head with a new head
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model

image = cv2.imread(img_path)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# img = torchvision.transforms.ToTensor()(img)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transform(img)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = get_model(1)
loaded_model.load_state_dict(torch.load(os.path.join(saved_model, 'faster_rcnn_model_20250126_003619.pth'), map_location = 'cpu'))
loaded_model.to(device)
loaded_model.eval()

img = img.to(device)
with torch.no_grad(): 
	prediction = loaded_model([img])

for element in range(len(prediction[0]['boxes'])):
	x_min, y_min, x_max, y_max = prediction[0]['boxes'][element].cpu().numpy().astype(int)
	score = np.round(prediction[0]['scores'][element].numpy(), decimals = 3)
	label_index = prediction[0]['labels'][element].numpy()
	label = labels_dict[int(label_index)]
	
	if score > 0.7:
		cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
		text = f"{label} {score}"
		cv2.putText(image, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
              (255, 255, 255), 1)
  
cv2.imshow("Predictions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()