import os
import sys
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

URL = "https://upload.wikimedia.org/wikipedia/commons/6/66/President_Trump_at_the_White_House_Summit_on_Human_Trafficking_%2849469604658%29.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8m.pt")
RESULT_PATH = os.path.join(BASE_DIR, "result.jpg")

PERSON_CLASS = 0
CONFIDENCE = 0.35
IOU = 0.5
TARGET_WIDTH = 1280


def get_remote_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req)
    except Exception as e:
        print("Can't load the image from this site:", e)
        return None
    data = np.frombuffer(resp.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_image(source):
    if os.path.isfile(source):
        return cv2.imread(source)
    return get_remote_image(source)


def resize(image):
    scale = TARGET_WIDTH / image.shape[1]
    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def adjust_gamma(image):
    mean = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
    print(f"Mean brightness: {mean:.1f}")

    if 80 <= mean <= 180:
        return image

    gamma = np.log(128 / 255) / np.log(max(mean, 1) / 255)
    gamma = float(np.clip(gamma, 0.7, 1.5))
    print(f"Gamma correction: {gamma:.2f}")
    table = np.array([((i / 255) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def preprocess(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

    gamma = adjust_gamma(denoised)

    lab = cv2.cvtColor(gamma, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def detect_people(model, image):
    result = model(image, classes=[PERSON_CLASS], conf=CONFIDENCE, iou=IOU,
                   imgsz=TARGET_WIDTH, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    return boxes, scores


def draw_people(image, boxes, scores):
    output = image.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"{score:.2f}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    text = f"People: {len(boxes)}"
    cv2.rectangle(output, (0, 0), (220, 45), (0, 0, 0), -1)
    cv2.putText(output, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return output


def print_stats(name, scores):
    if len(scores) > 0:
        mean = scores.mean()
    else:
        mean = 0
    print(f"{name}: {len(scores)} people, mean confidence {mean:.2f}")


def main():
    source = URL
    if len(sys.argv) > 1:
        source = sys.argv[1]

    image = load_image(source)
    if image is None:
        sys.exit("Error while loading the image")
    print("Image size:", image.shape)

    image = resize(image)
    processed = preprocess(image)

    model = YOLO(MODEL_PATH)

    boxes_orig, scores_orig = detect_people(model, image)
    boxes_proc, scores_proc = detect_people(model, processed)

    print_stats("Original", scores_orig)
    print_stats("Preprocessed", scores_proc)
    print("\nPeople in the room:", len(boxes_proc))

    result_orig = draw_people(image, boxes_orig, scores_orig)
    result_proc = draw_people(processed, boxes_proc, scores_proc)
    cv2.imwrite(RESULT_PATH, result_proc)

    plt.figure("People counting", figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(result_orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Original: {len(boxes_orig)} people")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title("Preprocessed (denoise, gamma, CLAHE)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_proc, cv2.COLOR_BGR2RGB))
    plt.title(f"Preprocessed: {len(boxes_proc)} people")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
