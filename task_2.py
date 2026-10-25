import sys
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np

URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/960px-Lemon.jpg"


def get_remote_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req)
    except Exception as e:
        print("Can't load the image from this site:", e)
        return None
    data = np.frombuffer(resp.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def show_histograms(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure("Histograms", figsize=(12, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist, color="black")
    plt.title("Whole image")
    plt.xlim(0, 255)

    plt.subplot(2, 3, 3)
    colors = ["blue", "green", "red"]
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=colors[i])
    plt.title("All channels")
    plt.xlim(0, 255)

    for i in range(3):
        plt.subplot(2, 3, 4 + i)
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=colors[i])
        plt.title("Channel " + colors[i])
        plt.xlim(0, 255)

    plt.tight_layout()


def check_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    pixels = gray.size

    mean = gray.mean()
    std = gray.std()
    dark = hist[0:5].sum() / pixels * 100
    bright = hist[251:256].sum() / pixels * 100
    low = np.percentile(gray, 1)
    high = np.percentile(gray, 99)

    score = 100

    # exposure
    if mean < 60 or mean > 195:
        score -= 30
    elif mean < 90 or mean > 165:
        score -= 15

    # clipping
    if dark > 5:
        score -= 20
    elif dark > 1:
        score -= 10

    if bright > 5:
        score -= 20
    elif bright > 1:
        score -= 10

    # contrast
    if high - low < 100 or std < 30:
        score -= 20
    elif high - low < 160:
        score -= 10

    if score < 0:
        score = 0

    if score >= 85:
        quality = "very good"
    elif score >= 70:
        quality = "good"
    elif score >= 50:
        quality = "average"
    else:
        quality = "bad"

    print("Score:", score, "/ 100")
    print("Quality:", quality)


def main():
    url = URL
    if len(sys.argv) > 1:
        url = sys.argv[1]

    image = get_remote_image(url)
    if image is None:
        sys.exit("Error while loading the image")
    print("Image size:", image.shape)

    check_quality(image)
    show_histograms(image)
    plt.show()


if __name__ == "__main__":
    main()
