import sys
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np

URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/960px-Lemon.jpg"


def get_remote_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    image = get_remote_image(URL)
    if image is None:
        sys.exit("Error while loading the image")
    print("Original:", image.shape)

    plt.figure("Original image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original {image.shape[1]}x{image.shape[0]}")
    plt.axis("off")

    resolution_reduction = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    grayscale = cv2.cvtColor(resolution_reduction, cv2.COLOR_BGR2GRAY)

    flip = cv2.rotate(grayscale, cv2.ROTATE_90_CLOCKWISE)

    plt.figure("Image result")
    plt.imshow(flip, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Result {flip.shape[1]}x{flip.shape[0]} (50%, grayscale, 90°)")
    plt.axis("off")

    print("\nMatrix")
    print(flip)

    plt.show()


if __name__ == "__main__":
    main()
