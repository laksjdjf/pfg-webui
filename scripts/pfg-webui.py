from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.processing import StableDiffusionProcessing

from scripts.pfg_utils import download, preprocess_image, TAGGER_FILE
from scripts.pfg_model import ViT
from PIL import Image

import os

# extensions/pfg-webui直下のパス
CURRENT_DIRECTORY = scripts.basedir()

class Script(scripts.Script):

    def __init__(self):
        self.model_list = [file for file in os.listdir(os.path.join(CURRENT_DIRECTORY, "models/")) if file != "put_models_here.txt"]
        if not os.path.exists(os.path.join(CURRENT_DIRECTORY, TAGGER_FILE)):
            download(CURRENT_DIRECTORY)

    def title(self):
        return "PFG for webui"

    # どうやらこれをやるとタブに常に表示されるらしい。
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("PFG", open=False):
                enabled = gr.Checkbox(value=False, label="Enable")
                with gr.Row():
                    image = gr.Image(type="pil", label="guide image")
                with gr.Row():
                    pfg_scale = gr.Slider(minimum=0, maximum=3, step=0.05, label="pfg scale", value=1.0)
                with gr.Row():
                    pfg_path = gr.Dropdown(self.model_list, label="pfg model", value=None)
                with gr.Row():
                    pfg_num_tokens = gr.Slider(minimum=0, maximum=20, step=1.0, value=10.0, label="pfg num tokens")
                with gr.Row():
                    sub_image = gr.Image(type="pil", label="sub image for latent couple")

        return enabled, image, pfg_scale, pfg_path, pfg_num_tokens, sub_image

    # wd-14-taggerの推論関数
    @torch.no_grad()
    def infer(self, img: Image):
        img = preprocess_image(img)
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        print("inferencing by torch model.")
        probs = self.tagger(img).squeeze(0)
        return probs

    # CFGのdenoising step前に起動してくれるらしい。
    def denoiser_callback(self, params: CFGDenoiserParams):
        if self.enabled:
            # (batch_size*num_prompts, cond_tokens, dim)
            cond = params.text_cond
            couple = self.batch_size * 3 == cond.shape[0]
            # (batch_size*num_prompts, uncond_tokens, dim)
            uncond = params.text_uncond

            # (1, num_tokens, dim)
            pfg_cond = self.pfg_cond.to(cond.device, dtype=cond.dtype)
            if couple:
                pfg_cond_sub = self.pfg_cond_sub.to(cond.device, dtype=cond.dtype)
                pfg_cond_zero = torch.zeros_like(pfg_cond_sub)
                # (3, num_tokens, dim) - >  (batch size * 3, num_tokens, dim)
                pfg_cond = torch.cat([pfg_cond, pfg_cond_sub, pfg_cond_zero]).repeat(self.batch_size, 1, 1)
            else:
                pfg_cond = pfg_cond.repeat(cond.shape[0], 1, 1)
            # concatenate
            params.text_cond = torch.cat([cond, pfg_cond], dim=1)

            # copy zero
            pfg_uncond_zero = torch.zeros(uncond.shape[0], self.pfg_num_tokens, uncond.shape[2]).to(uncond.device, dtype=uncond.dtype)
            params.text_uncond = torch.cat([uncond, pfg_uncond_zero], dim=1)

            if params.sampling_step == 0:
                print(f"Apply pfg num_tokens:{self.pfg_num_tokens}(this message will be duplicated)")

    def process(
            self,
            p: StableDiffusionProcessing,
            enabled: bool,
            image: Image,
            pfg_scale: float,
            pfg_path: str,
            pfg_num_tokens: int,
            sub_image: Image = None
    ):

        self.enabled = enabled
        if not self.enabled:
            return

        self.image = image
        self.sub_image = sub_image
        self.pfg_scale = pfg_scale
        self.pfg_num_tokens = pfg_num_tokens
        self.batch_size = p.batch_size

        pfg_weight = torch.load(os.path.join(CURRENT_DIRECTORY, "models/" + pfg_path))
        self.weight = pfg_weight["pfg_linear.weight"].cpu()  # 大した計算じゃないのでcpuでいいでしょう
        self.bias = pfg_weight["pfg_linear.bias"].cpu()

        if not hasattr(self, 'tagger'):
            self.tagger = ViT(3, 448, 9083)
            self.tagger.load_state_dict(torch.load(os.path.join(CURRENT_DIRECTORY, TAGGER_FILE)))
            self.tagger.eval()

        pfg_feature = self.infer(self.image)
        # (768,) -> (dim * num_tokens, )
        self.pfg_cond = (self.weight @ pfg_feature + self.bias) * self.pfg_scale

        # (dim * num_tokens, ) -> (1, num_tokens, dim)
        self.pfg_cond = self.pfg_cond.reshape(1, self.pfg_num_tokens, -1)

        if sub_image is not None:
            pfg_feature_sub = self.infer(self.sub_image)
            self.pfg_cond_sub = (self.weight @ pfg_feature_sub + self.bias) * self.pfg_scale
            self.pfg_cond_sub = self.pfg_cond_sub.reshape(1, self.pfg_num_tokens, -1)
        else:
            self.pfg_feature_sub = None

        if not hasattr(self, 'callbacks_added'):
            on_cfg_denoiser(self.denoiser_callback)
            self.callbacks_added = True

        return

    def postprocess(self, *args):
        return
