from functools import partial

import torch
from benchmarking_utils import BenchmarkMixin, BenchmarkScenario, model_init_fn

from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
from diffusers.utils.testing_utils import torch_device


CKPT_ID = "black-forest-labs/FLUX.1-dev"
RESULT_FILENAME = "flux.csv"


def get_input_dict(**device_dtype_kwargs):
    # resolution: 1024x1024
    # maximum sequence length 512
    hidden_states = torch.randn(1, 4096, 64, **device_dtype_kwargs)
    encoder_hidden_states = torch.randn(1, 512, 4096, **device_dtype_kwargs)
    pooled_prompt_embeds = torch.randn(1, 768, **device_dtype_kwargs)
    image_ids = torch.ones(512, 3, **device_dtype_kwargs)
    text_ids = torch.ones(4096, 3, **device_dtype_kwargs)
    timestep = torch.tensor([1.0], **device_dtype_kwargs)
    guidance = torch.tensor([1.0], **device_dtype_kwargs)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "img_ids": image_ids,
        "txt_ids": text_ids,
        "pooled_projections": pooled_prompt_embeds,
        "timestep": timestep,
        "guidance": guidance,
    }


if __name__ == "__main__":
    scenarios = [
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=model_init_fn,
            compile_kwargs={"fullgraph": True},
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-bnb-nf4",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
                ),
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=model_init_fn,
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-layerwise-upcasting",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(model_init_fn, layerwise_upcasting=True),
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-group-offload-leaf",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(
                model_init_fn,
                group_offload_kwargs={
                    "onload_device": torch_device,
                    "offload_device": torch.device("cpu"),
                    "offload_type": "leaf_level",
                    "use_stream": True,
                    "non_blocking": True,
                },
            ),
        ),
    ]

    runner = BenchmarkMixin()
    runner.run_bencmarks_and_collate(scenarios, filename=RESULT_FILENAME)
