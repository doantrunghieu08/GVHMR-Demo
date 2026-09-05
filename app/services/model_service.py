from pathlib import Path
import torch
import hydra
from hydra import initialize_config_module, compose
from hmr4d.configs import register_store_gvhmr

model = None
hydra_initalized = False

def get_api_config(video_path: Path, static_cam: bool = False, use_dpvo: bool = False):
    global hydra_initalized

    if not hydra_initalized:
        initialize_config_module(
            version_base="1.3",
            config_module="hmr4d.configs"
        )
        hydra_initalized = True
         
    overrides = [
        f"video_name='{video_path.stem}'",
        f"static_cam={str(static_cam).lower()}",
        f"verbose=False",
        f"use_dpvo={str(use_dpvo).lower()}"
    ]
        
    register_store_gvhmr()
    
    config = compose(config_name="demo", overrides=overrides)
        
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.preprocess_dir).mkdir(parents=True, exist_ok=True)
        
    return config

def load_model_into_gpu():
    global model
    ckpt_path = "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"

    if not Path(ckpt_path).exists():
        print(f"[Cảnh báo]: Không tìm thấy checkpoint tại {ckpt_path}. Vui lòng tải về trước.")
        return
    print("Loading GVHMR model into GPU...")

    global hydra_initalized
    if not hydra_initalized:
        initialize_config_module(version_base="1.3", config_module="hmr4d.configs")
        hydra_initalized = True

        register_store_gvhmr()
        config = compose(config_name="demo")
        
        model = hydra.utils.instantiate(config.model, _recursive_=False)
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()
        print("Model loaded successfully")

def get_model():
    return model
