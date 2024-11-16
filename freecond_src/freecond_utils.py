from .freecond import fc_config, get_pipeline

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

    elif method=="hdp":
        print("🔄 Building HD-painter FreeCond control...")

    elif method=="pp":
        print("🔄 Building PowerPaint FreeCond control...")
    elif method=="bn":
        print("🔄 Building BrushNet FreeCond control...")
        
    else:
        print("🔄 Building Stable-Diffusion-Inpainting FreeCond control...")

        pipe = get_pipeline(variant).to(device)
        def forward(fc_control, init_image, mask_image, prompt="National Chiao Tung University", negative_prompt="females, money, hope", guidance_scale=15, num_inference_steps=50, generator=None, **kwargss):
            return pipe.freecond_forward_staged(
                fc_control,
                prompt, init_image, mask_image,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps, 
                **kwargss
                )
    
    return pipe, forward