import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

st.set_page_config(page_title="Skin Condition Classifier", layout="wide")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    try:
        # Enable unsafe deserialization for Lambda layers
        tf.keras.config.enable_unsafe_deserialization()
        
        # Ensemble model (for prediction)
        ensemble_model = tf.keras.models.load_model(
            "models/skin_condition_ensemble_model.keras", 
            safe_mode=False,
            compile=False
        )
        
        # Grad-CAM backbone model (DenseNet121)
        gradcam_model = tf.keras.models.load_model(
            "models/densenet121_skin_initial.h5",
            compile=False
        )
        
        return ensemble_model, gradcam_model, True
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        
        # Try loading just the DenseNet model
        try:
            gradcam_model = tf.keras.models.load_model(
                "models/densenet121_skin_initial.h5",
                compile=False
            )
            st.success("✓ Loaded DenseNet model for predictions")
            return gradcam_model, gradcam_model, False
        except Exception as e2:
            st.error(f"Could not load any models: {e2}")
            return None, None, False

# Load models
with st.spinner("Loading AI models..."):
    ensemble_model, gradcam_model, models_loaded = load_models()

if models_loaded:
    st.success("✓ All models loaded successfully!")

CLASS_NAMES = ['BLACK-HEADS', 'ACNE', 'PORES', 'HYPER-PIGMENTATION', 'WRINKLES']

# -------------------------------
# Skin Product Recommendations
# -------------------------------
recommendations = {
    "ACNE": [
        "🧴 Salicylic Acid Cleanser", 
        "💊 Niacinamide Serum",
        "🌿 Tea Tree Oil Spot Treatment",
        "🚫 Oil-Free Moisturizer"
    ],
    "BLACK-HEADS": [
        "🪨 Charcoal Face Mask",
        "🌀 BHA Exfoliant", 
        "💦 Salicylic Acid Toner",
        "✨ Clay Mask Weekly"
    ],
    "HYPER-PIGMENTATION": [
        "🍊 Vitamin C Serum",
        "🌟 Alpha Arbutin",
        "🛡️ Broad Spectrum Sunscreen SPF 50+",
        "💫 Tranexamic Acid"
    ],
    "PORES": [
        "🌀 Niacinamide Serum",
        "🪨 Clay Mask 2x Weekly",
        "💦 AHA/BHA Exfoliant",
        "❄️ Pore-Minimizing Toner"
    ],
    "WRINKLES": [
        "🕰️ Retinol Night Cream",
        "💎 Peptide Serum", 
        "🛡️ Sunscreen Daily",
        "💧 Hyaluronic Acid Serum"
    ]
}

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Handle different image formats
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------------------
# FIXED Grad-CAM function
# -------------------------------
def get_grad_cam(model, img_array, last_conv_layer_name):
    """Generate Grad-CAM heatmap - Fixed version"""
    try:
        st.write(f"Testing layer: {last_conv_layer_name}")
        
        # Get the last convolutional layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        st.write(f"✅ Layer found: {last_conv_layer.name}")
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        st.write("✅ Gradient model created")
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            st.write(f"✅ Conv outputs shape: {conv_outputs.shape}")
            st.write(f"✅ Predictions type: {type(predictions)}")
            
            # FIX: Handle the case where predictions might be a list
            if isinstance(predictions, list):
                st.write(f"✅ Predictions is a list with {len(predictions)} elements")
                # Take the first element if it's a list (usually the main output)
                predictions_tensor = predictions[0]
            else:
                predictions_tensor = predictions
                
            st.write(f"✅ Predictions tensor shape: {predictions_tensor.shape}")
            
            # Get the predicted class
            pred_index = tf.argmax(predictions_tensor[0])
            class_channel = predictions_tensor[:, pred_index]
            st.write(f"✅ Predicted class index: {pred_index}")
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        st.write(f"✅ Gradients shape: {grads.shape}")
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        st.write(f"✅ Pooled gradients shape: {pooled_grads.shape}")
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        
        st.write(f"✅ Conv outputs after [0]: {conv_outputs.shape}")
        
        # Create heatmap by weighting feature maps
        heatmap = tf.zeros(conv_outputs.shape[0:2])  # Initialize with spatial dimensions
        
        for i in range(pooled_grads.shape[0]):
            heatmap += pooled_grads[i] * conv_outputs[:, :, i]
        
        st.write(f"✅ Heatmap shape before processing: {heatmap.shape}")
        
        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / tf.reduce_max(heatmap)
        
        # Convert to numpy and resize
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Convert to RGB heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        st.write("✅ Grad-CAM successfully generated!")
        return heatmap
        
    except Exception as e:
        st.error(f"❌ Error in Grad-CAM with layer '{last_conv_layer_name}': {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Skin Condition Detection")

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"), horizontal=True)
uploaded_image = None

if option == "Upload Image":
    file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if file:
        uploaded_image = Image.open(file).convert("RGB")
elif option == "Use Webcam":
    picture = st.camera_input("Capture a photo")
    if picture:
        uploaded_image = Image.open(picture).convert("RGB")

if uploaded_image:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Image")
        st.image(uploaded_image, use_container_width=True)
    
    with col2:
        # Preprocess
        img_array = preprocess_image(uploaded_image)

        # Prediction
        with st.spinner("Analyzing skin condition..."):
            if ensemble_model is not None:
                preds = ensemble_model.predict(img_array, verbose=0)[0]
                model_type = "Ensemble Model"
            elif gradcam_model is not None:
                preds = gradcam_model.predict(img_array, verbose=0)[0]
                model_type = "DenseNet Model"
            else:
                # Mock prediction
                np.random.seed(hash(uploaded_image.tobytes()) % 1000)
                preds = np.random.dirichlet(np.ones(5), size=1)[0]
                model_type = "Demo Mode"
            
            class_idx = np.argmax(preds)
            confidence = preds[class_idx]
            pred_class = CLASS_NAMES[class_idx]

        st.subheader("Analysis Results")
        st.caption(f"Using: {model_type}")
        
        st.write(f"**Detected Condition:** {pred_class.replace('-', ' ').title()}")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Confidence bars
        st.write("**Detailed Confidence Scores:**")
        for i, class_name in enumerate(CLASS_NAMES):
            conf = preds[i]
            bar_length = int(conf * 20)
            bar = "█" * bar_length + "▒" * (20 - bar_length)
            emoji = "▷" if i == class_idx else "○"
            st.write(f"{emoji} {class_name.replace('-', ' ').title():<20} {bar} {conf:.1%}")

        # Recommendation
        st.subheader("Recommended Skincare")
        if pred_class in recommendations:
            for item in recommendations[pred_class]:
                st.write(f"• {item}")

    # Grad-CAM Visualization - FIXED LAYER NAMES
    if gradcam_model is not None:
        st.subheader("Heatmap Visualization")
        
        with st.spinner("Generating heatmap..."):
            # Use CONVOLUTIONAL layers only (not concatenation layers)
            densenet_layers = [
                "conv5_block16_2_conv",  # Primary - last conv layer before final concat
                "conv5_block16_1_conv",  # Secondary
                "conv5_block16_0_conv",  # Tertiary
                "conv5_block15_2_conv",  # Fallback options
                "conv5_block14_2_conv",
                "conv5_block13_2_conv"
            ]
            
            heatmap = None
            successful_layer = None
            
            for layer_name in densenet_layers:
                try:
                    heatmap = get_grad_cam(gradcam_model, img_array, layer_name)
                    if heatmap is not None:
                        successful_layer = layer_name
                        break
                except Exception as e:
                    continue
            
            if heatmap is not None:
                if successful_layer:
                    st.success(f"✓ Heatmap generated using layer: {successful_layer}")
                
                # Create overlay
                img_resized = cv2.resize(np.array(uploaded_image), (224, 224))
                overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
                
                # Display heatmaps
                col3, col4 = st.columns(2)
                with col3:
                    st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
                with col4:
                    st.image(overlay, caption="Overlay on Image", use_container_width=True)
            else:
                st.error("❌ Could not generate heatmap visualization")
                st.info("The prediction still works - this is just a visualization issue")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("AI skin condition analyzer with Grad-CAM visualization")
    
    st.header("Supported Conditions")
    for condition in CLASS_NAMES:
        st.write(f"• {condition.replace('-', ' ').title()}")
    
    st.header("Model Status")
    if models_loaded:
        st.success("✓ All Models Loaded")
    elif gradcam_model is not None:
        st.warning("⚠️ Basic Model Only")
    else:
        st.error("❌ Demo Mode")

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit | Educational purposes only")


