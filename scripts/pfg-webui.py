from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from modules import devices

import modules.scripts as scripts
import gradio as gr

from modules.script_callbacks import CFGDenoisedParams, on_cfg_denoiser

from modules.processing import StableDiffusionProcessing

from scripts.dbimutils import smart_imread_pil, smart_24bit, make_square, smart_resize
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model, Model

import os

CURRENT_DIRECTORY = scripts.basedir()

class Script(scripts.Script):

    def __init__(self):
        self.model_list = os.listdir(os.path.join(CURRENT_DIRECTORY, "models/"))

    def title(self):
        return "PFG for webui"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("PFG", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                with gr.Row():
                    image = gr.Image(type="pil", label="guide images")
                with gr.Row():
                    pfg_scale = gr.Slider(minimum=0, maximum=3, step=0.05, label="pfg scale", value=1.0)
                with gr.Row():
                    tagger_path = gr.Textbox(value="wd-v1-4-vit-tagger-v2", label="WD14tagger-vit-v2 path")
                    pfg_path = gr.Dropdown(self.model_list, label="pfg model", value = None)
                    pfg_num_tokens = gr.Slider(minimum=0, maximum=20, step=1.0, value=10.0, label="pfg num tokens")
                    
        return enabled, image, pfg_scale, tagger_path, pfg_path, pfg_num_tokens
                                      
    def infer(self, img:Image):
        img = smart_imread_pil(img)
        img = smart_24bit(img)
        img = make_square(img, 448)
        img = smart_resize(img, 448)
        img = img.astype(np.float32)
        probs = self.tagger(np.array([img]), training=False)
        return torch.tensor(probs.numpy()).squeeze(0).cpu()

    def denoiser_callback(self, params: CFGDenoisedParams):
        cond = params.tensor
        uncond = params.uncond
        pfg_feature = self.infer(self.image)
        pfg_cond = self.weight @ pfg_feature + self.bias
        pfg_cond = pfg_cond.reshape(1, self.pfg_num_tokens, -1).repeat(cond.shape[0],1,1)
        pfg_cond = pfg_cond.to(cond.device, dtype = cond.dtype)
        params.tensor = torch.cat([cond,pfg_cond],dim=1)
        params.uncond = torch.cat([uncond,uncond[:,-1:,:].repeat(1,self.pfg_num_tokens,1)],dim=1)
                                     

    def process(self, p: StableDiffusionProcessing, enabled:bool, image: Image, pfg_scale:float, tagger_path:str, pfg_path: str, pfg_num_tokens:int):
        self.enabled = enabled
        if not self.enabled:
            return
        self.image = image
        self.pfg_scale = pfg_scale
        self.pfg_num_tokens = pfg_num_tokens
        
        pfg_weight = torch.load(os.path.join(CURRENT_DIRECTORY, "models/" + pfg_path))
        self.weight = pfg_weight["pfg_linear.weight"].cpu() #大した計算じゃないのでcpuでいいでしょう
        self.bias = pfg_weight["pfg_linear.bias"].cpu()

        if not hasattr(self, 'tagger'):
            #なんもいみわかっとらんけどこれしないとVRAMくわれる。対応するバージョンもよくわからない
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
            else:
                print("Not enough GPU hardware devices available")

            self.tagger = load_model(os.path.join(CURRENT_DIRECTORY, tagger_path))
            self.tagger = Model(self.tagger.layers[0].input, self.tagger.layers[-3].output) #最終層手前のプーリング層の出力を使う
                                     
        if not hasattr(self, 'callbacks_added'):
            on_cfg_denoiser(self.denoiser_callback)
            self.callbacks_added = True

        return

    def postprocess(self, *args):
        return
