import torch
import numpy as np
import cv2


def generate_gradcam(model, image_tensor, target_class, conv_layer_name='conv3'):
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    target_layer = dict(model.named_modules())[conv_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    class_score = output[0] if output.shape[-1] == 1 else output[0, target_class]
    model.zero_grad()
    class_score.backward()

    acts = activations['value'].detach().cpu().squeeze(0)
    grads = gradients['value'].detach().cpu().squeeze(0)
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() != 0 else cam
    cam_resized = cv2.resize(cam.numpy(), (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET)  # type: ignore
    img_np = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img_np = np.uint8(255 * img_np)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)  # type: ignore

    forward_handle.remove()
    backward_handle.remove()

    return overlay, cam_resized


def draw_aggressive_merged_boxes(overlay, cam_resized, threshold=0.5, dilation_iter=3, min_area=300, merge_distance=30):
    binary_map = np.uint8(cam_resized > threshold)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_map, kernel,  # type: ignore
                         iterations=dilation_iter)  # type: ignore

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))

    merged = []
    while boxes:
        x1, y1, x2, y2 = boxes.pop(0)
        changed = True
        while changed:
            changed = False
            for other in boxes[:]:
                ox1, oy1, ox2, oy2 = other
                if not (ox1 > x2 + merge_distance or ox2 < x1 - merge_distance or
                        oy1 > y2 + merge_distance or oy2 < y1 - merge_distance):
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    boxes.remove(other)
                    changed = True
        merged.append((x1, y1, x2, y2))

    boxed = overlay.copy()
    for x1, y1, x2, y2 in merged:
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return boxed
