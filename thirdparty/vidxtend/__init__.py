import os
import sys
from diffusers import pipelines

# Register models with diffusers, ensuring `from_pretrained` works
for dirname in ["models"]:
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), dirname)):
        if filename.endswith(".py") and not filename.startswith("__"):
            model_name = filename[:-3]
            sys.modules[f"{__name__}.{dirname}.{model_name}"] = __import__(f"{__name__}.{dirname}.{model_name}", fromlist=[model_name])
            setattr(pipelines, f"{__name__}.{dirname}.{model_name}", sys.modules[f"{__name__}.{dirname}.{model_name}"])

# Re-export pipeline
from vidxtend.pipelines import VidXTendPipeline
__all__ = ["VidXTendPipeline"]
