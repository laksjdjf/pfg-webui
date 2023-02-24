#このコードはhttps://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.pyを参考にしていますというかパクっています。

from huggingface_hub import hf_hub_download
import os

DEFAULT_WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-vit-tagger-v2'
TAGGER_DIR = 'wd-v1-4-vit-tagger-v2'
FILES = ["keras_metadata.pb", "saved_model.pb"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]

def download(path):
    model_dir = os.path.join(path, TAGGER_DIR)
    if not os.path.exists(model_dir):
        print(f"downloading wd14 tagger model from hf_hub. id: {DEFAULT_WD14_TAGGER_REPO}")
        for file in FILES:
            hf_hub_download(DEFAULT_WD14_TAGGER_REPO, file, cache_dir=model_dir, force_download=True, force_filename=file)
        for file in SUB_DIR_FILES:
            hf_hub_download(DEFAULT_WD14_TAGGER_REPO, file, subfolder=SUB_DIR, cache_dir=os.path.join(
                model_dir, SUB_DIR), force_download=True, force_filename=file)
    else:
        print("using existing wd14 tagger model")
