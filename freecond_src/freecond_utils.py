import numpy as np
import torch
from PIL import Image

from .freecond import fc_config, get_pipeline

# Modified from
from .freecond_controlnet import FreeCondControlNetPipeline, ControlNetModel

# Modified from 
from hdpainter_src import models
from hdpainter_src.methods import fc_rasg, rasg, sd, sr
from hdpainter_src.utils import IImage, resize


PROMPT="University"
NPROMPT="money, love, hope"
def get_pipeline_forward(method="sd", variant="sd15",device="cuda", **kwargs):
    """_summary_

    Args:
        fc_control (fc_config): FreeCond control.
        method (str, optional): Currently support ["sd","cn","hdp","pp","bn"]. Defaults to "sd".
        checkpoint (str, optional): Mainly designed for SDs currently support ["sd15","sd2","sdxl","ds8"]  . Defaults to "sd15".
        **kwargs specify the hyperparameter for method
    Returns:
        pipeline (Depending on the method, but mainly diffuser.pipeline): the object of pipeline for adjusting scheduler?
          or printing model details
        forward (): generalized forward function across different baselines
    """
    print("❗❗❗ Be sure using correct python environment, the python environment are different for methods ")
    if method=="cn":
        print("🔄 Building ConrtrolNet-Inpainting FreeCond control...")
        controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        )
        pipe = FreeCondControlNetPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
        )
        def make_inpaint_condition(image, image_mask):
            image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

            assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
            image[image_mask > 0.5] = -1.0  # set as masked pixel
            image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
            return image
        
        
        def forward(fc_control, init_image, mask_image,
                    prompt=PROMPT, negative_prompt=NPROMPT,
                    guidance_scale=15, num_inference_steps=50, generator=None, **kwargss):
            
            control_image = make_inpaint_condition(init_image, mask_image)
            return pipe.freecond_forward_staged(
                fc_control,
                prompt, init_image, mask_image,
                control_image=control_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps, 
                **kwargss
                )


    elif method=="hdp":
        print("🔄 Building HD-Painter FreeCond control...")

        if "hdp_methods" in kwargs:
            hdp_methods=kwargs["hdp_methods"]
        else:
            hdp_methods="painta+fc_rasg"

        if "rasg-eta" in kwargs:
            hdp_rasg_eta=kwargs["rasg-eta"]
        else:
            hdp_rasg_eta=0.1
        
        pipe = models.load_inpainting_model(variant, device='cuda:0', cache=True)
        runner=fc_rasg

        hdp_negative_prompt = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"
        positive_prompt = "Full HD, 4K, high quality, high resolution"

        
        def forward(fc_control, init_image, mask_image,
                    prompt=PROMPT, negative_prompt=NPROMPT,
                    guidance_scale=15, num_inference_steps=50, generator=None, **kwargss):
            return runner.run(
                    fc_control,
                    ddim=pipe,
                    seed=1234,
                    method=method,
                    prompt=prompt,
                    image=IImage(init_image),
                    mask=IImage(mask_image),
                    eta=hdp_rasg_eta,
                    negative_prompt=negative_prompt+hdp_negative_prompt,
                    positive_prompt=positive_prompt,
                    guidance_scale=guidance_scale,
                    num_steps=num_inference_steps,
                ).pil(), None




    elif method=="pp":
        print("🔄 Building PowerPaint FreeCond control...")
    elif method=="bn":
        print("🔄 Building BrushNet FreeCond control...")

    else:
        print("🔄 Building Stable-Diffusion-Inpainting FreeCond control...")

        pipe = get_pipeline(variant).to(device)
        def forward(fc_control, init_image, mask_image,
                    prompt=PROMPT, negative_prompt=NPROMPT,
                    guidance_scale=15, num_inference_steps=50, generator=None, **kwargss):
            return pipe.freecond_forward_staged(
                fc_control,
                prompt, init_image, mask_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps, 
                **kwargss
                )
    try:
        pipe.to(device)
    except:
        print("Unknown error happen when setting device")
    return pipe, forward