from streamlit_drawable_canvas import st_canvas
import torch
import streamlit as st
from PIL import Image
import numpy as np
from model import MNISTmodelv0



st.title("MNIST Image Classifier")


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
checkpoint = torch.load("checkpoint_modelv2.pth", 
                        map_location=torch.device('cpu'))

# Create the model instance
model = MNISTmodelv0(input_shape=1, hidden_units=10, output_shape=10)

# Load only the model state dict from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()
with torch.no_grad():
    if canvas_result.image_data is not None:
    # Convert RGBA properly - paste onto black background first
        img_rgba = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        img = Image.new("L", img_rgba.size, 0)  # Black background
        img.paste(img_rgba.convert("L"), mask=img_rgba.split()[3])  # Use alpha as mask
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.LANCZOS)
        
        
        # Normalize
        img_array = np.array(img) / 255.0
        
        # Reshape and convert to tensor
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
