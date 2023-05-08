from ...utils import (
    OptionalDependencyNotAvailable,
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)


try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    print("to-do")
#    from ...utils.dummy_torch_and_transformers_objects import UnCLIPImageVariationPipeline, UnCLIPPipeline
else:
    from .pipeline_kandinsky import KandinskyPipeline, KandinskyPriorPipeline
    from .text_proj import KandinskyTextProjModel
