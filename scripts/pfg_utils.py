#このコードはhttps://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.pyを参考にしていますというかパクっています。

from huggingface_hub import hf_hub_download
import os
import cv2
from PIL import Image
import numpy as np

IMAGE_SIZE = 448

TAGGER_REPO = "furusu/wd-v1-4-tagger-pytorch"
TAGGER_FILE = "wd-v1-4-vit-tagger-v2.ckpt"

def download(path):
    print(f"downloading onnx model from hf_hub.")
    hf_hub_download(TAGGER_REPO, TAGGER_FILE, cache_dir=path, force_download=True, force_filename=TAGGER_FILE)

def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image