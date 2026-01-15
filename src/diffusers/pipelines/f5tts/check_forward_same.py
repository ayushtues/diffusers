import sys
sys.path.append('/Users/ayushmangal/f5_contri/F5-TTS/src')

def seed_everything(seed):
    import random
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True)

    random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# training script.
import sys
sys.path.append('/Users/ayushmangal/f5_contri/F5-TTS/src')
import os
from importlib.resources import files
import torch
import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)
cfg_path = '/Users/ayushmangal/f5_contri/F5-TTS/src/f5_tts/configs/F5TTS_v1_Base.yaml'

model_cfg = OmegaConf.load(cfg_path)
model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch
tokenizer = model_cfg.model.tokenizer
mel_spec_type = model_cfg.model.mel_spec.mel_spec_type


# set text tokenizer
if tokenizer != "custom":
    tokenizer_path = model_cfg.datasets.name
else:
    tokenizer_path = model_cfg.model.tokenizer_path
vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

# set model
model = CFM(
    transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
    mel_spec_kwargs=model_cfg.model.mel_spec,
    vocab_char_map=vocab_char_map,
)

# print('x:', model.state_dict()['transformer.text_embed.text_embed.weight'])
# save model
ckpt_path = '/Users/ayushmangal/f5_contri/F5-TTS/ckpts/model_1250000.safetensors'
from safetensors.torch import load_file

checkpoint = load_file(ckpt_path)
checkpoint = {"ema_model_state_dict": checkpoint}

checkpoint["model_state_dict"] = {
    k.replace("ema_model.", ""): v
    for k, v in checkpoint["ema_model_state_dict"].items()
    if k not in ["initted", "update", "step"]
}
model.load_state_dict(checkpoint["model_state_dict"], strict=True)


with open('/Users/ayushmangal/f5_contri/F5-TTS/ckpts/vocab.txt', "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)

from diffusers.pipelines.f5tts.pipeline_f5tts import F5DiTModel, F5ConditioningEncoder, F5FlowPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

dit_config = {
"dim": 1024,
"depth": 22,
"heads": 16,
"ff_mult": 2,
"text_dim": 512,
"text_num_embeds": 256,
"text_mask_padding": True,
"qk_norm": None,  # null | rms_norm
"conv_layers": 4,
"pe_attn_head": None,
"attn_backend": "torch",  # torch | flash_attn
"attn_mask_enabled": False,
"checkpoint_activations": False,  # recompute activations and save memory for extra compute
}


mel_spec_config = {
    "target_sample_rate": 24000,
    "n_mel_channels": 100,
    "hop_length": 256,
    "win_length": 1024,
    "n_fft": 1024,
}


dit = F5DiTModel(**dit_config)
print("DiT model initialized with config:", dit_config)

conditioning_encoder_config = {
    'dim': 1024,
    'text_num_embeds': vocab_size,
    'text_dim': 512,
    'text_mask_padding': True,
    'conv_layers': 4,
    'mel_dim': mel_spec_config['n_mel_channels'],
}
conditioning_encoder = F5ConditioningEncoder(**conditioning_encoder_config)
print("Conditioning Encoder initialized with config:", conditioning_encoder_config)

# f5_pipeline = F5FlowPipeline(
#     transformer=dit,
#     conditioning_encoder=conditioning_encoder,
#     vocab_char_map=vocab_char_map,
#     scheduler=FlowMatchEulerDiscreteScheduler()
# )
f5_pipeline = F5FlowPipeline.from_pretrained('/Users/ayushmangal/f5_contri/diffusers/src/diffusers/pipelines/f5tts/f5tts_test_pipeline')
print("F5FlowPipeline initialized with DiT and Conditioning Encoder.")
# push to hub
f5_pipeline.push_to_hub("f5_tts", private=False)
exit()

def load_pipeline_components_from_state_dict(state_dict, f5_pipeline):
    """
    Load the components of the F5FlowPipeline from a state_dict.
    """
    # print('state_dict, ', state_dict)
    conditioning_encoder_state_dict = {}
    for key in f5_pipeline.conditioning_encoder.state_dict().keys():
        if 'transformer.' + key in state_dict:
            if 'grn' not in key:
                conditioning_encoder_state_dict[key] = state_dict['transformer.' + key]
            else:
                grn_param = state_dict['transformer.' + key]
                grn_param = grn_param.unsqueeze(0) 
                conditioning_encoder_state_dict[key] = grn_param
    f5_pipeline.conditioning_encoder.load_state_dict(conditioning_encoder_state_dict)


    transformer_state_dict = {}
    # Load transformer
    for key in f5_pipeline.transformer.state_dict().keys():
        if key in state_dict:
            transformer_state_dict[key] = state_dict[key]
        if 'transformer.' + key in state_dict:
            transformer_state_dict[key] = state_dict['transformer.' + key]
    f5_pipeline.transformer.load_state_dict(transformer_state_dict)
    
    return f5_pipeline

# f5_pipeline = load_pipeline_components_from_state_dict(model.state_dict(), f5_pipeline)

# check what keys have not changed 
for key in f5_pipeline.conditioning_encoder.state_dict().keys():
    if key in model.state_dict():
        if not torch.allclose(f5_pipeline.conditioning_encoder.state_dict()[key], model.state_dict()[key], atol=1e-3):
            print(f"Key {key} has changed in the conditioning encoder state dict.")
    # Check if the key exists in the model state dict with a 'transformer.' prefix
    elif 'transformer.' + key in model.state_dict():
        if not torch.allclose(f5_pipeline.conditioning_encoder.state_dict()[key], model.state_dict()['transformer.' + key], atol=1e-3):
            print(f"Key {key} has changed in the conditioning encoder state dict.")

for key in f5_pipeline.transformer.state_dict().keys():
    if key in model.state_dict():
        if not torch.allclose(f5_pipeline.transformer.state_dict()[key], model.state_dict()[key], atol=1e-3):
            print(f"Key {key} has changed in the transformer state dict.")
            print(f"Key {key} in model state dict: {model.state_dict()[key]}")
            print(f"Key {key} in f5_pipeline state dict: {f5_pipeline.transformer.state_dict()[key]}")
            break
    elif 'transformer.' + key in model.state_dict():
        if not torch.allclose(f5_pipeline.transformer.state_dict()[key], model.state_dict()['transformer.' + key], atol=1e-3):
            print(f"Key {key} has changed in the transformer state dict.")
            print(f"Key {key} in model state dict: {model.state_dict()['transformer.' + key]}")
            print(f"Key {key} in f5_pipeline state dict: {f5_pipeline.transformer.state_dict()[key]}")
            break
        
model.eval()
f5_pipeline.transformer.eval()

# save the pipeline locally
# f5_pipeline.save_pretrained("/Users/ayushmangal/f5_contri/diffusers/src/diffusers/pipelines/f5tts/f5tts_test_pipeline")


import torch 
# ref_audio = torch.randn(1, 16000)  # Dummy reference audio
import torchaudio 

ref_audio_path = "/Users/ayushmangal/Downloads/joker_trimmed.wav"
ref_audio, sr = torchaudio.load(ref_audio_path)
if sr != 24000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
    ref_audio = resampler(ref_audio)
ref_audio = ref_audio.mean(dim=0, keepdim=True)  # convert to mono

duration = 2500

ref_text = "See their morals, their code, its a bad joke. Dropped at the first sign of trouble"  # Dummy reference text
gen_text = "This is a generated sentence. This is a generated sentence. This is a generated sentence."  # Dummy generated text

text = [ref_text+" "+gen_text]  # Combine reference and generated text
seed_everything(42)

ref_audio_len = ref_audio.shape[1] // 256

# calculate output length based on speed

speed = 1.0
total_text = ref_text + " " + gen_text
output_len = ref_audio_len  + int(ref_audio_len * (1.0) * (len(total_text)/len(ref_text)))

num_steps = 2
cfg_strength = 2.0

x1, _ = model.sample(cond=ref_audio,
    text=text,
    duration=output_len,
    steps=num_steps,
    cfg_strength=cfg_strength,
    sway_sampling_coef=-1.0,
    )




import torch 

seed_everything(42)
x2 = f5_pipeline(ref_audio=ref_audio,
                   ref_text=ref_text,
                   gen_text=gen_text,
                   speed=torch.tensor([1.0]), num_inference_steps=num_steps, guidance_scale=cfg_strength)


x2 = x2['audios']
import torch

# from vocos import Vocos

# vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
# 1, 100, 250
print("x1:", x1.shape)
print('x2:', x2.shape)

x1 = x1.permute(0, 2, 1)
# audio_1 = vocoder.decode(x1)
# audio_2 = vocoder.decode(x2)
# print('x1:', torch.mean(x1).item())
# print('x2:', torch.mean(x2).item())
# print(torch.allclose(x1, x2, atol=1e-3))
# print("Audio 1:", audio_1.shape, torch.mean(audio_1).item())
# print("Audio 2:", audio_2.shape, torch.mean(audio_2).item())
# print(torch.allclose(audio_1, audio_2, atol=1e-3))
# # save audio
# torchaudio.save("/Users/ayushmangal/f5_contri/diffusers/src/diffusers/pipelines/f5tts/audio_1.wav", audio_1.cpu(), 24000)
# torchaudio.save("/Users/ayushmangal/f5_contri/diffusers/src/diffusers/pipelines/f5tts/audio_2.wav", audio_2.cpu(), 24000)