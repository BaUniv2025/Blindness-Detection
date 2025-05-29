import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2

from utils.model import BinaryCNN
from utils.visualisation import generate_gradcam, draw_aggressive_merged_boxes

# --- Настройки интерфейса ---
st.set_page_config(page_title="Диагностика DR", layout="centered")
st.title("🧠 Классификация глазного дна: Healthy / Diabetic Retinopathy")

class_names = ["Diabetic Retinopathy", "Healthy"]

# --- Определение устройства ---
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# --- Загрузка модели ---


@st.cache_resource
def load_model():
    model = BinaryCNN()
    model.load_state_dict(torch.load(
        "data/model1.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()

# --- Трансформация изображения ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Загрузка изображения ---
uploaded_file = st.file_uploader(
    "Загрузите изображение глаза (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение",
             use_container_width=True)

    image_tensor = transform(image).to(device)  # type: ignore

    # --- Инференс ---
    with torch.no_grad():
        logit = model(image_tensor.unsqueeze(0))
        prob = torch.sigmoid(logit).item()
        prediction = int(prob >= 0.5)

    # --- Вывод вероятностей ---
    prob_dr = 1 - prob
    prob_healthy = prob
    pred_class = class_names[prediction]

    st.markdown(f"### Предсказание: **{pred_class}**")
    st.markdown(f"""
    - 🟢 **Healthy**: {prob_healthy:.4f} ({prob_healthy*100:.2f}%)
    - 🔴 **Diabetic Retinopathy**: {prob_dr:.4f} ({prob_dr*100:.2f}%)
    """)

    # --- Grad-CAM ---
    overlay, cam_resized = generate_gradcam(
        model, image_tensor, target_class=prediction)
    boxed_overlay = draw_aggressive_merged_boxes(overlay, cam_resized)

    st.image(boxed_overlay, caption="Grad-CAM с зонами внимания",
             use_container_width=True)

    # --- Скачивание результата ---
    st.download_button(
        label="📥 Скачать Grad-CAM",
        data=cv2.imencode(".png", boxed_overlay)[1].tobytes(),
        file_name="gradcam_result.png",
        mime="image/png"
    )
