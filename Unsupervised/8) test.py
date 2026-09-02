import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import io
from sklearn.decomposition import TruncatedSVD


#import thr image      image matrix store in (img)
img = Image.open("D:\ML Course\Datasets\image.jpg")

target_kb=float(input("Compress size is:"))

# Convert image to RGB
img = img.convert("RGB")

img_array = np.array(img)
print(img_array.shape)

plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.show()

import os
image_path = r"D:\ML Course\Datasets\image.jpg"
size_kb = os.path.getsize(image_path) / 1024
print("Original Image Size:", size_kb, "KB")

img_array = np.array(img)

total_components = img_array.size
print("Total Components:", total_components)

Component_per_KB=total_components/size_kb
print("Component per KB:",Component_per_KB)

component_for_Compress=Component_per_KB*target_kb
print("component_for_Compress is:",component_for_Compress)

n_components = int(component_for_Compress)
# Maximum allowed components

def compress_channel(channel, k):
    svd = TruncatedSVD(
        n_components=k,
        random_state=42
    )

    compressed = svd.fit_transform(
        channel
    )

    reconstructed = svd.inverse_transform(
        compressed
    )

    return reconstructed


def get_image_size_kb(img):
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG"
    )
    size_kb = buf.tell() / 1024
    return size_kb

from PIL import Image
import numpy as np

def compress_image(img, target_kb):

    # Convert PIL image to NumPy array
    img = np.array(img)

    # Find dimensions
    height, width, channels = img.shape

    print(height, width)

    original_size_kb = img.nbytes / 1024

    ratio = target_kb / original_size_kb

    k = int(ratio * min(height, width))

    k = max(1, min(k, min(height, width) - 1))

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    R_rec = compress_channel(R, k)
    G_rec = compress_channel(G, k)
    B_rec = compress_channel(B, k)

    compressed_img = np.dstack((R_rec, G_rec, B_rec))

    return compressed_img


compressed_image = compress_image(img, target_kb)

compressed_image = np.clip(
    compressed_image,
    0,
    255
).astype(np.uint8)

# Compressed Image
plt.subplot(1,2,2)
plt.imshow(compressed_image)
plt.title("Compressed Image")
plt.axis("off")
plt.show()


from PIL import Image

# Convert NumPy array to image
compressed_pil = Image.fromarray(compressed_image)

# Save image
compressed_pil.save(
    r"D:\ML Course\Datasets\compressed_image.jpg"
)

print("Compressed image saved successfully")