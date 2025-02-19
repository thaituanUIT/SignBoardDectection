# Batch inference

import torch
import numpy as np
import cv2
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

saved_model = "C:\\Users\\USER\\Documents\\tempoRun-Remastered\\saved_model"  # Directory of the saved model
test_dataset_path = "datasets/test/"  # Directory containing test images
output_file_path = "detection_results.txt"  # Output file to save results

# Load labels
with open('labels.txt', 'r') as f:
	string = f.read()
	labels_dict = eval(string)

# Define model loading function
def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None  # Do not load pre-trained weights
    )

    # Number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Define image preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return image, transform(img_rgb)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = get_model(1)  # Update the number of classes accordingly
loaded_model.load_state_dict(torch.load(os.path.join(saved_model, 
                                                     'faster_rcnn_model_20250126_003619.pth'), map_location=device))
loaded_model.to(device)
loaded_model.eval()

# Process the dataset
with open(output_file_path, 'w') as output_file:
    for filename in os.listdir(test_dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only images
            image_path = os.path.join(test_dataset_path, filename)
            image, img_tensor = preprocess_image(image_path)

            # Move tensor to device
            img_tensor = img_tensor.to(device)

            # Make predictions
            with torch.no_grad():
                prediction = loaded_model([img_tensor])

            # Write predictions to the output file
            for element in range(len(prediction[0]['boxes'])):
                x_min, y_min, x_max, y_max = prediction[0]['boxes'][element].cpu().numpy().astype(int)
                score = prediction[0]['scores'][element].cpu().numpy()
                label_index = prediction[0]['labels'][element].cpu().numpy()

                if score > 0.3:  # Confidence threshold
                    label = labels_dict[int(label_index)]
                    output_file.write(f"{filename}, {x_min}, {y_min}, {x_max}, {y_max}, {label}\n")

print(f"Detection results saved to {output_file_path}")