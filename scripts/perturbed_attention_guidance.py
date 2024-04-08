from types import MethodType

import gradio as gr
import torch

from ldm.util import default  # type: ignore

from modules import scripts, script_callbacks, shared  # type: ignore
from modules.sd_samplers_cfg_denoiser import CFGDenoiser  # type: ignore
from modules.hypernetworks import hypernetwork  # type: ignore


def perturbed_forward(self, x, context=None, mask=None, **kwargs):
    """Perturbed Attention"""

    batch_size, sequence_length, inner_dim = x.shape
    if mask is not None:
        mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
        mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])

    context = default(context, x)
    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    v_in = self.to_v(context_v)
    hidden_states = v_in

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states


class PAGScript(scripts.Script):
    pag_scale = 0.0
    rescale = 0.0
    old_combine_denoised = None
    denoised_perturb = None

    def __init__(self):
        super().__init__()
        script_callbacks.on_cfg_denoiser(self.on_cfg_denoiser)
        # TODO: xyz grid support

    def title(self):
        return "Perturbed Attention Guidance"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion("Perturbed Attention Guidance", open=True, elem_id="pag"):
            pag_scale = gr.Slider(label="PAG Scale", minimum=0.0, maximum=10.0, step=0.5, value=0.0)
            rescale = gr.Slider(label="CFG Rescale", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
        return [pag_scale, rescale]

    def process(self, p, pag_scale: float, rescale: float):
        # SDXL not supported
        if shared.sd_model.is_sdxl:
            return

        self.pag_scale = pag_scale
        self.rescale = rescale
        # TODO: extra generation params

        # override CFGDenoiser.combine_denoised
        def pagrescale_combine_denoised(denoiser, x_out, conds_list, uncond, cond_scale):
            denoised_uncond = x_out[-uncond.shape[0]:]
            denoised = torch.clone(denoised_uncond)
            if self.pag_scale > 0:
                assert self.denoised_perturb is not None, "pag_scale > 0 but no denoised_perturb saved"
                assert self.denoised_perturb.shape == denoised.shape, "denoised_perturb shape doesn't match denoised"

            for i, conds in enumerate(conds_list):
                for cond_index, weight in conds:
                    xcfg = denoised[i] + (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)
                    # apply PAG
                    if self.pag_scale > 0:
                        pag_delta = x_out[cond_index] - self.denoised_perturb[cond_index]
                        xcfg += pag_delta * (weight * self.pag_scale)
                    # apply cfg rescale
                    if self.rescale > 0:
                        # adapted from https://github.com/Seshelle/CFG_Rescale_webui
                        xrescaled = torch.std(x_out[cond_index]) / torch.std(xcfg)
                        xfinal = self.rescale * xrescaled + (1.0 - self.rescale)
                        xcfg *= xfinal
                    denoised[i] = xcfg

            return denoised

        self.old_combine_denoised = CFGDenoiser.combine_denoised
        CFGDenoiser.combine_denoised = pagrescale_combine_denoised

    def on_cfg_denoiser(self, params: script_callbacks.CFGDenoiserParams):
        """Perform PAG forward pass and save result for later"""

        # SDXL not supported
        if shared.sd_model.is_sdxl or self.pag_scale == 0:
            return

        # prevent DeepCache from caching this step
        deepcache = None
        for script in scripts.scripts_txt2img.scripts:
            if script.title().lower() == "deepcache":
                deepcache = script
        if deepcache is not None:
            deepcache_timestep = deepcache.session.enumerated_timestep["value"]
            deepcache.session.enumerated_timestep["value"] = -1

        # patch unet mid block self attention
        mid_block = shared.sd_model.model.diffusion_model.middle_block
        for _, module in mid_block.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                mid_attn1 = module.attn1
                break  # there should only be one
        orig_forward = mid_attn1.forward
        mid_attn1.forward = MethodType(perturbed_forward, mid_attn1)

        # perturbed forward pass. save result for combine_denoised
        if isinstance(params.text_uncond, dict):
            make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
        else:
            make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}
        uncond_size = params.text_uncond.shape[0]
        self.denoised_perturb = params.denoiser.inner_model(
            params.x[:-uncond_size],
            params.sigma[:-uncond_size],
            cond=make_condition_dict(params.text_cond, params.image_cond[:-uncond_size])
        )

        # restore mid block
        mid_attn1.forward = orig_forward

        # restore DeepCache
        if deepcache is not None:
            deepcache.session.enumerated_timestep["value"] = deepcache_timestep


    def postprocess(self, p, processed, pag: float, rescale: float):
        CFGDenoiser.combine_denoised = self.old_combine_denoised
        self.pag_scale = 0.0
        self.rescale = 0.0
        self.old_combine_denoised = None
        self.denoised_perturb = None
