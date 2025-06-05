import torch
import numpy as np
import cv2


def generate_gradcam(model, image_tensor, target_class, conv_layer_name='conv3'):
    model.eval()  # Переводим модель в режим оценки
    # Определяем устройство для вычислений
    device = torch.device("mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Перемещаем модель на выбранное устройство
    # Добавляем batch размерность и переносим на устройство
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Словари для хранения активаций и градиентов слоя
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        # Сохраняем выход слоя при прямом проходе
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        # Сохраняем градиенты слоя при обратном проходе
        gradients['value'] = grad_output[0]

    target_layer = dict(model.named_modules())[
        conv_layer_name]  # Получаем нужный слой по имени
    forward_handle = target_layer.register_forward_hook(
        forward_hook)  # Регистрируем forward hook
    backward_handle = target_layer.register_backward_hook(
        backward_hook)  # Регистрируем backward hook

    output = model(image_tensor)  # Прямой проход изображения через модель
    # Получаем скор для нужного класса
    class_score = output[0] if output.shape[-1] == 1 else output[0, target_class]
    model.zero_grad()  # Обнуляем градиенты модели
    class_score.backward()  # Запускаем обратное распространение для вычисления градиентов

    # Получаем активации слоя
    acts = activations['value'].detach().cpu().squeeze(0)
    # Получаем градиенты слоя
    grads = gradients['value'].detach().cpu().squeeze(0)
    # Усредняем градиенты
    weights = grads.mean(dim=(1, 2))

    # Инициализируем карту активаций и накладываем веса
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.clamp(cam, min=0)  # Оставляем только положительные значения
    cam = cam - cam.min()  # Нормируем карту
    cam = cam / cam.max() if cam.max() != 0 else cam  # Масштабируем к [0, 1]
    # Изменяем размер до 224x224
    cam_resized = cv2.resize(cam.numpy(), (224, 224))

    # Создаём цветовую карту (тепловую карту)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)\

    # Переводим тензор изображения в numpy
    img_np = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img_np = np.uint8(255 * img_np)  # Масштабируем значения к [0, 255]
    # Накладываем тепловую карту на изображение
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    forward_handle.remove()  # Удаляем forward hook
    backward_handle.remove()  # Удаляем backward hook

    return overlay, cam_resized  # Возвращаем наложенное изображение и карту активаций


def draw_aggressive_merged_boxes(overlay, cam_resized, threshold=0.5, dilation_iter=3, min_area=300, merge_distance=30, line_width=2):
    # Бинаризуем карту активаций по порогу
    binary_map = np.uint8(cam_resized > threshold)
    kernel = np.ones((3, 3), np.uint8)  # Ядро для дилатации
    dilated = cv2.dilate(binary_map, kernel, iterations=dilation_iter)
    # Находим контуры на дилатированной карте
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        # Фильтруем маленькие области
        if cv2.contourArea(cnt) >= min_area:
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(cnt)
            # Сохраняем координаты прямоугольника
            boxes.append((x, y, x + w, y + h))

    merged = []
    while boxes:
        x1, y1, x2, y2 = boxes.pop(0)  # Берём первый прямоугольник
        changed = True
        while changed:
            changed = False
            # Проверяем пересечение с другими прямоугольниками
            for other in boxes[:]:
                ox1, oy1, ox2, oy2 = other
                # Если пересекаются или близко
                if not (ox1 > x2 + merge_distance or ox2 < x1 - merge_distance or
                        oy1 > y2 + merge_distance or oy2 < y1 - merge_distance):
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    # Удаляем объединённый прямоугольник
                    boxes.remove(other)
                    changed = True
        # Добавляем объединённый прямоугольник
        merged.append((x1, y1, x2, y2))

    boxed = overlay.copy()  # Копируем изображение для рисования
    # Рисуем прямоугольники
    for x1, y1, x2, y2 in merged:
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 255, 255),
                      line_width)

    return boxed  # Возвращаем изображение с нарисованными прямоугольниками
