#importing necessary libraries
import streamlit as st
import torch
import cv2
import numpy as np
from yolov5.utils.general import non_max_suppression, xywh2xyxy, xyxy2xywh
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
import os

# Set up the Streamlit app
st.title("Signature Detection using YOLOv5 - Tobacco 800 Dataset - Amogh B")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load the trained YOLOv5 model with the best.pt file generated from training the transfer learning model
weights_path = "D:\Signature-Detection-Yolov5\yolov5\\runs\\train\\Tobacco-run\\weights\\best.pt"
device = select_device('')
model = attempt_load(weights_path)

# Setting the model to evaluation mode thus deactivating the dropouts and batch norm to get accurate predictions on unseen data.
model.eval()

# Set threshold for object detection
conf_threshold = 0.4

# Set input image size
img_size = 640

if uploaded_file is not None:
    # Load input image
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Resize input image to model's expected size
    resized_img = cv2.resize(img, (img_size, img_size)).copy()

    # Convert BGR image to RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to range [0, 1]
    norm_img = rgb_img / 255.0

    # Convert image to torch tensor and add batch dimension
    tensor_img = torch.from_numpy(norm_img).to(device).float()
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0)

    # Get model output
    pred = model(tensor_img)

    # Apply non-maximum suppression to remove overlapping boxes
    pred = non_max_suppression(pred, conf_threshold, 0.5)

    import os

# Set path to labels directory
labels_dir = "D:\Signature-Detection-Yolov5\\tobacco_data_zhugy\\tobacco_data_zhugy\\tobacco_yolo_format\\labels\\valid"

# Define class names for the dataset
class_names = ['DLLogo','DLSignature']

# Get list of all label files in directory
label_files = os.listdir(labels_dir)

# Parse uploaded file name to get corresponding label file name
if uploaded_file is not None:
    file_name = uploaded_file.name
    label_file_name = os.path.splitext(file_name)[0] + ".txt"
    
    # Check if corresponding label file exists
    if label_file_name in label_files:
        # Load label file and extract coordinates
        label_path = os.path.join(labels_dir, label_file_name)
        with open(label_path, 'r') as f:
            labels = f.read().splitlines()
        boxes = []
        for label in labels:
            cls, x, y, w, h = map(float, label.split())
            x1 = int((x - w/2) * img.shape[1])
            y1 = int((y - h/2) * img.shape[0])
            x2 = int((x + w/2) * img.shape[1])
            y2 = int((y + h/2) * img.shape[0])
            boxes.append([x1, y1, x2, y2])

        # Draw boxes on the input image for each detected object
        if len(boxes) > 0:
            for box in boxes:
                # Get the class label for the detected object
                class_label = class_names[int(cls)]
                # Draw bounding box and class label on the input image
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img, class_label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the output image in Streamlit
    st.image(img, channels="BGR")
