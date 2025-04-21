import torch
from ultralytics import YOLO
from torchviz import make_dot

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(1, 3, 640, 640)

# Forward pass through the model to get the output
output = model.model(dummy_input)

# Ensure the output is a tensor by checking if it's a list or tuple
if isinstance(output, (list, tuple)):
    output_tensor = output[0]
else:
    output_tensor = output

# If the output tensor is still a list, convert it to a tensor
if isinstance(output_tensor, list):
    output_tensor = torch.stack(output_tensor)

# Convert the output tensor into a scalar by summing
scalar_output = output_tensor.sum()

# Generate the graph from the scalar output
dot = make_dot(scalar_output, params=dict(model.model.named_parameters()))

# Save the graph to a file and display it
dot.format = 'png'
dot.render('yolov8n_model_architecture')