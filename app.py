from streamlit_drawable_canvas import st_canvas
import torch
import streamlit as st
from PIL import Image
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from model import MNISTmodelv0, MNISTmodelv1



st.title("MNIST Digit Classifier")


# draw a imageing canvas
canvas_result = st_canvas(
    fill_color="#000000",  # Black
    stroke_width=20,
    stroke_color="#FFFFFF",  # White
    background_color="#000000",  # Black
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

st.write("Draw a digit (0-9) on the canvas above and see the processed image below.")



# Load the checkpoint
model1_checkpoint = torch.load("checkpoint_modelv2.pth", 
                        map_location=torch.device('cpu')) ## Small Custom Model
model2_checkpoint = torch.load("checkpoint_modelv3.pth", 
                        map_location=torch.device('cpu')) ## Deep Dustom Model
model3_checkpoint = torch.load("checkpoint_modelv4.pth", 
                        map_location=torch.device('cpu')) ## Mobilenet V3 transfer learning model

# Create the model instance
model1 = MNISTmodelv0(input_shape=1, hidden_units=10, output_shape=10)
model2 = MNISTmodelv1(input_shape=1, hidden_units=40, output_shape=10)
model3 = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

model3.classifier[3] = torch.nn.Linear(in_features=1024, out_features=10)
# Select which model to use in streamlit sidebar with little description about model
model_option = st.sidebar.selectbox(
    "Select Model",
    ("Small Custom Model", "Deep Custom Model", "MobileNet V3 Transfer Learning Model")
)

if model_option == "Small Custom Model":
    model = model1
    checkpoint = model1_checkpoint
    st.sidebar.write("Using a small custom CNN model with 3 convolutional layers.")

elif model_option == "Deep Custom Model":
    model = model2
    checkpoint = model2_checkpoint
    st.sidebar.write("Using a deeper custom CNN model with 4 convolutional layers and batch normalization.")

else:
    model = model3
    checkpoint = model3_checkpoint
    st.sidebar.write("Using MobileNet V3 small model pre-trained on ImageNet with transfer learning.")




# Load only the model state dict from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()
with torch.no_grad():
    if canvas_result.image_data is not None:
    # Convert RGBA properly - paste onto black background first
        img_rgba = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        img = Image.new("L", img_rgba.size, 0)  # Black background

        # for mobile net v3 convert to 3 channel else 1 channel
        if model_option == "MobileNet V3 Transfer Learning Model":
            img_rgba_rgb = img_rgba.convert("RGB")
            img.paste(img_rgba_rgb, mask=img_rgba.split()[3])  # Use alpha channel as mask
        else:
            img.paste(img_rgba.convert("L"), mask=img_rgba.split()[3])  # Use alpha channel as mask

        
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        
        # Normalize
        img_array = np.array(img) / 255.0
        
        # Reshape and convert to tensor
        if model_option == "MobileNet V3 Transfer Learning Model":
            input_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Convert to 3 channels
        else:
            input_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
        
        st.write(f"Predicted Digit: {predicted_label}")

# put a button to show classification report
    if st.button("Show Classification Report"):

# draw a drawer for classification report where all  probalities shown there
        st.image(img.resize((140, 140)), caption="Processed Image", use_column_width=False)
        st.write("Classification Probabilities:")
        probabilities = torch.softmax(output, dim=1).squeeze().numpy()
        for i, prob in enumerate(probabilities):
            st.write(f"Digit {i}: {prob*100:.2f}%")



# put a prediction probalities across alll models if button clicked but for mobile net we need 3 channel input
# just show for that particular digit recorded on canvas
    if st.button("Compare All Models"):
        st.image(img.resize((140, 140)), caption="Processed Image", use_column_width=False)
        st.write("Model Comparison Probabilities:")
        models = {
            "Small Custom Model": model1,
            "Deep Custom Model": model2,
            "MobileNet V3 Transfer Learning Model": model3
        }
        checkpoints = {
            "Small Custom Model": model1_checkpoint,
            "Deep Custom Model": model2_checkpoint,
            "MobileNet V3 Transfer Learning Model": model3_checkpoint
        }
        
        for model_name, mdl in models.items():
            # Load state dict
            mdl.load_state_dict(checkpoints[model_name]['model_state_dict'])
            mdl.eval()
            with torch.no_grad():
                if model_name == "MobileNet V3 Transfer Learning Model":
                    input_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Convert to 3 channels
                else:
                    input_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                output = mdl(input_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze().numpy()
                
                st.write(f"**{model_name}**")
                for i, prob in enumerate(probabilities):
                    st.write(f"Digit {i}: {prob*100:.2f}%")
                st.write("---")
