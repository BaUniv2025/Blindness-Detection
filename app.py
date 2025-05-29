import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from utils.model import BinaryCNN
from utils.visualisation import generate_gradcam, draw_aggressive_merged_boxes

# --- Настройки страницы ---
st.set_page_config(page_title="Диагностика ретинопатии", layout="wide")
st.title("🧠 Классификация глазного дна: Здоров / Диабетическая ретинопатия")

# --- Классы ---
class_names = ["Здоровое состояние", "Диабетическая ретинопатия"]

# --- Устройство ---
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# --- Модель ---


@st.cache_resource
def load_model():
    model = BinaryCNN()
    model.load_state_dict(torch.load("data/model1.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()

# --- Преобразование ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Загрузка файла ---
st.markdown("#### Загрузите изображение глазного дна (JPG/PNG):")
uploaded_file = st.file_uploader(
    label="", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((224, 224))
    image_tensor = transform(resized_image).to(device)

    with torch.no_grad():
        logit = model(image_tensor.unsqueeze(0))
        prob = torch.sigmoid(logit).item()
        prediction = int(prob >= 0.5)

    prob_dr = prob
    prob_healthy = 1 - prob
    pred_class = class_names[prediction]

    st.markdown(f"### 🩺 Предсказание: **{pred_class}**")
    st.markdown(f"""
    - 🟢 **{class_names[0]}**: {prob_healthy:.4f} ({prob_healthy * 100:.2f}%)
    - 🔴 **{class_names[1]}**: {prob_dr:.4f} ({prob_dr * 100:.2f}%)
    """)

    # --- Grad-CAM с прямоугольниками ---
    if prediction == 1:
        overlay, cam_resized = generate_gradcam(
            model, image_tensor, target_class=1)
        boxed_overlay = draw_aggressive_merged_boxes(
            np.array(resized_image), cam_resized)
    else:
        boxed_overlay = np.array(resized_image)

    # --- Отображение рядом: Оригинал + Боксы ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_image, caption="Оригинал", use_container_width=True)
    with col2:
        st.image(boxed_overlay, caption="С зонами внимания",
                 use_container_width=True)

    # --- Скачивание ---
    st.download_button(
        label="📥 Скачать изображение с зонами",
        data=cv2.imencode(".png", boxed_overlay)[1].tobytes(),
        file_name="gradcam_boxes.png",
        mime="image/png"
    )
