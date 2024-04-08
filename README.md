# sd-webui-perturbed-attention-guidance
[Perturbed Attention Guidance](https://github.com/sunovivid/Perturbed-Attention-Guidance) for stable-diffusion-webui

The basic implementation is finished.

Generation times are doubled because it uses a separate forward pass.

An optimized implementation would require more invasive changes to the webui's denoising loop. PRs welcome.

This extension breaks Seshelle's [CFG Rescale](https://github.com/Seshelle/CFG_Rescale_webui) extension, so an alternative CFG rescale slider is provided.
