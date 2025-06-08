import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from utils.model import BinaryCNN, BinaryImprovedCNN
from utils.visualisation import generate_gradcam, draw_merged_boxes

# Функция для русификации дополнительных элементов интерфейса


def style_language_uploader():
    st.markdown("""
    <style>
    /* Скрытие оригинальных надписей */
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        display: none !important;
    }

    /* Контейнер с кастомными строками */
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.25rem;
    }

    /* Инструкция */
    div[data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "Перетащите изображение сюда или нажмите Обзор";
        display: block;
        font-size: 14px;
        color: #6c757d;
    }

    /* Ограничения */
    div[data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "Максимальный размер файла — 200 МБ • JPG, JPEG, PNG";
        display: block;
        font-size: 12px;
        color: #6c757d;
    }

     /* Скрытие надписи на кнопке загрузки */
    section[data-testid="stFileUploaderDropzone"] > button[data-testid="stBaseButton-secondary"] {
        color: transparent !important;
        position: relative;
    }

    /* Добавление новой надписи */
    section[data-testid="stFileUploaderDropzone"] > button[data-testid="stBaseButton-secondary"]::after {
        content: "Обзор";
        color: black;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        pointer-events: none;
    }
    
    /* Скрытие верхней панели Streamlit полностью */
    div[data-testid="stToolbar"] {
        display: none !important;
    }

    /* Скрыть кнопку Deploy */
    div[data-testid="stAppDeployButton"] {
        display: none !important;
    }

    /* Скрыть меню (три точки) */
    #MainMenu {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Настройки страницы
st.set_page_config(page_title="Диагностика ретинопатии", layout="wide")
st.title("Классификация глазного дна: Здоров / Диабетическая ретинопатия")

# Cайдбар с информацией о модели
with st.sidebar:
    st.markdown("### Используемая модель")
    st.markdown("- Файл: `model2.pth`")
    st.markdown(
        "- Классы: \n   - Здоровое состояние \n   - Диабетическая ретинопатия"
    )

# Классы
class_names = ["Здоровое состояние", "Диабетическая ретинопатия"]

# Выбор устройства
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# Загрузка модели


@st.cache_resource
def load_model():
    model = BinaryImprovedCNN()
    model.load_state_dict(torch.load("data/model2.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()

# Преобразование изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Отображение изображений бок о бок


def show_side_by_side(left_img, right_img, captions=("Оригинал", "С зонами внимания")):
    col1, col2 = st.columns(2)

    with col1:
        st.image(left_img, caption=captions[0], use_container_width=True)
    with col2:
        st.image(right_img, caption=captions[1], use_container_width=True)


# Русификация интерфейса загрузки
style_language_uploader()

# Интерфейс загрузки
st.markdown("#### Загрузите изображение глазного дна:")
uploaded_file = st.file_uploader(
    label="Загрузите изображение глазного дна",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# Обработка изображения
if uploaded_file:
    # Оригинальное изображение
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((224, 224))
    image_tensor = transform(resized_image).to(device)  # type: ignore

    # Предсказание
    with torch.no_grad():
        logit = model(image_tensor.unsqueeze(0))
        prob = torch.sigmoid(logit).item()
        prediction = int(prob >= 0.5)

    prob_dr = prob
    prob_healthy = 1 - prob
    pred_class = class_names[prediction]

    # Вывод вероятностей
    st.markdown(f"### Предсказание: **{pred_class}**")
    st.markdown(f"""
    - 🟢 **{class_names[0]}**: {prob_healthy:.4f} ({prob_healthy * 100:.2f}%)
    - 🔴 **{class_names[1]}**: {prob_dr:.4f} ({prob_dr * 100:.2f}%)
    """)

    # Grad-CAM для визуализации зон внимания
    if prediction == 1:
        _, cam_resized = generate_gradcam(model, image_tensor, target_class=1)
        boxed_overlay = draw_merged_boxes(
            np.array(resized_image),
            cam_resized,
            threshold=0.2,
            dilation_iter=3,
            min_area=200,
            merge_distance=20,
            line_width=1)
    else:
        boxed_overlay = np.array(resized_image)

    # Сравнение изображений
    st.markdown("### Сравнение изображений:")
    show_side_by_side(resized_image, boxed_overlay)

    # Скачивание результата
    boxed_bgr = cv2.cvtColor(boxed_overlay, cv2.COLOR_RGB2BGR)

    st.download_button(
        label="📥 Скачать изображение с зонами внимания",
        data=cv2.imencode(".png", boxed_bgr)[1].tobytes(),
        file_name="gradcam_boxes.png",
        mime="image/png"
    )
