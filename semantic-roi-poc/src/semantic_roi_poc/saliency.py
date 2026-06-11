"""Stage A: obtain an image and its saliency map.

Two providers share one interface (:class:`SaliencyResult`):

* :class:`DiffusersAttentionExtractor` -- the *real* method. It runs a
  Stable Diffusion text-to-image generation and aggregates the cross-attention
  maps of the chosen subject tokens over a set of denoising steps and U-Net
  layers, producing a saliency map *for free* as a by-product of generation.
  Requires ``torch`` + ``diffusers`` + a GPU; imported lazily so the rest of
  the package works without them.

* :class:`SyntheticSaliency` -- a dependency-free fixture that renders a
  subject (a bright shape) over a textured background and returns an exact
  saliency map. It exists so the *whole* downstream pipeline (Stage B/C/D) can
  be smoke-tested on any machine before the real generator is plugged in.
  It is a test fixture, **not** a stand-in for real results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SaliencyResult", "SyntheticSaliency", "DiffusersAttentionExtractor"]


@dataclass
class SaliencyResult:
    """An image plus its saliency map and a derived ROI mask.

    image:
        ``(H, W, 3)`` uint8 RGB image.
    saliency:
        ``(H, W)`` float32 map normalised to ``[0, 1]``.
    prompt:
        The text prompt used (or a label for synthetic data).
    """

    image: np.ndarray
    saliency: np.ndarray
    prompt: str

    def roi_mask(self, quantile: float = 0.7) -> np.ndarray:
        """Boolean ROI mask = saliency above the given quantile threshold."""
        thr = float(np.quantile(self.saliency, quantile))
        return self.saliency >= thr


def _normalise(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class SyntheticSaliency:
    """Render a subject-on-background image with a known saliency map."""

    def __init__(self, height: int = 256, width: int = 256, seed: int = 0):
        self.height = height
        self.width = width
        self.rng = np.random.default_rng(seed)

    def generate(self, prompt: str = "synthetic subject on textured background") -> SaliencyResult:
        h, w = self.height, self.width
        # Textured (high-frequency) background so ROI bit redistribution matters.
        bg = self.rng.integers(60, 196, size=(h, w, 3), dtype=np.int16)
        noise = self.rng.integers(-30, 30, size=(h, w, 3), dtype=np.int16)
        img = np.clip(bg + noise, 0, 255).astype(np.uint8)

        # Subject: a smooth bright elliptical blob, off-centre.
        cy, cx = int(h * 0.42), int(w * 0.55)
        ry, rx = h * 0.18, w * 0.14
        yy, xx = np.mgrid[0:h, 0:w]
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        blob = np.exp(-dist * 1.5)

        subject_color = np.array([235, 90, 70], dtype=np.float32)
        alpha = blob[..., None]
        img = (img * (1 - alpha) + subject_color * alpha).astype(np.uint8)

        saliency = _normalise(blob)
        return SaliencyResult(image=img, saliency=saliency.astype(np.float32), prompt=prompt)


class DiffusersAttentionExtractor:
    """Generate an image with Stable Diffusion and extract attention saliency.

    Parameters
    ----------
    model_id:
        Hugging Face model id of a Stable Diffusion pipeline.
    device:
        ``"cuda"`` (recommended) or ``"cpu"``.
    steps:
        Number of denoising steps.
    layer_resolutions:
        Cross-attention spatial resolutions (token-grid side lengths) to
        aggregate. Mid resolutions such as 16 and 32 carry the cleanest
        semantic localisation; very coarse/fine layers are noisier.
    step_range:
        Fractional ``(start, end)`` window of denoising steps to average over.
        Mid/late steps localise the subject better than the first noisy steps.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        layer_resolutions: tuple[int, ...] = (16, 32),
        step_range: tuple[float, float] = (0.2, 0.9),
    ):
        self.model_id = model_id
        self.device = device
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.layer_resolutions = layer_resolutions
        self.step_range = step_range
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch  # noqa: F401  (validated lazily)
        from diffusers import StableDiffusionPipeline

        import torch as _torch

        dtype = _torch.float16 if self.device.startswith("cuda") else _torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        pipe = pipe.to(self.device)
        pipe.safety_checker = None
        self._pipe = pipe

    def _subject_token_indices(self, prompt: str, subject: str) -> list[int]:
        """Token positions (in the CLIP sequence) for the subject words."""
        tokenizer = self._pipe.tokenizer
        ids = tokenizer(prompt).input_ids
        sub_ids = tokenizer(subject, add_special_tokens=False).input_ids
        idx = [i for i, t in enumerate(ids) if t in set(sub_ids)]
        return idx or list(range(1, min(len(ids) - 1, 4)))

    def generate(self, prompt: str, subject: str | None = None) -> SaliencyResult:
        """Run generation and return the image + aggregated attention saliency.

        ``subject`` defaults to the first noun-ish word of the prompt.
        """
        self._load()
        import torch
        from diffusers.models.attention_processor import Attention

        subject = subject or prompt.strip().split(",")[0].split()[-1]

        start_step = int(self.step_range[0] * self.steps)
        end_step = int(self.step_range[1] * self.steps)
        step_counter = {"i": 0}

        # We record cross-attention probabilities by swapping in a recorder
        # processor on every attention module, then restore the originals.
        attn_maps: list[tuple[int, np.ndarray]] = []

        def patched_call(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
            is_cross = encoder_hidden_states is not None
            residual = hidden_states
            q = attn.to_q(hidden_states)
            ctx = encoder_hidden_states if is_cross else hidden_states
            k = attn.to_k(ctx)
            v = attn.to_v(ctx)
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)
            probs = attn.get_attention_scores(q, k, attention_mask)
            if is_cross:
                res = int(round(probs.shape[1] ** 0.5))
                if res in self.layer_resolutions and start_step <= step_counter["i"] < end_step:
                    p = probs.detach().float().mean(0).cpu().numpy()  # (hw, tokens)
                    attn_maps.append((res, p))
            out = torch.bmm(probs, v)
            out = attn.batch_to_head_dim(out)
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            return out + residual if attn.residual_connection else out

        def on_step_end(pipe, step, timestep, cbk):
            step_counter["i"] = step + 1
            return cbk

        # Apply the patched processor to all attention modules.
        originals = []
        for name, module in self._pipe.unet.named_modules():
            if isinstance(module, Attention):
                originals.append((module, module.processor))

        # Use diffusers' AttnProcessor override via a thin shim object.
        class _RecorderProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
                return patched_call(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)

        for module, _ in originals:
            module.set_processor(_RecorderProcessor())

        try:
            g = torch.Generator(device=self.device).manual_seed(0)
            result = self._pipe(
                prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                height=self.height,
                width=self.width,
                generator=g,
                callback_on_step_end=on_step_end,
            )
            image = np.asarray(result.images[0])
        finally:
            for module, proc in originals:
                module.set_processor(proc)

        token_idx = self._subject_token_indices(prompt, subject)
        sal = self._aggregate(attn_maps, token_idx)
        sal = self._resize(sal, image.shape[0], image.shape[1])
        return SaliencyResult(image=image, saliency=_normalise(sal).astype(np.float32), prompt=prompt)

    @staticmethod
    def _aggregate(attn_maps: list[tuple[int, np.ndarray]], token_idx: list[int]) -> np.ndarray:
        if not attn_maps:
            raise RuntimeError(
                "no cross-attention captured; check layer_resolutions/step_range "
                "and the installed diffusers version"
            )
        acc = None
        count = 0
        target = max(res for res, _ in attn_maps)
        for res, p in attn_maps:
            # p: (hw, tokens) -> select subject tokens, reshape to (res, res).
            sel = p[:, [t for t in token_idx if t < p.shape[1]]]
            if sel.size == 0:
                continue
            m = sel.mean(axis=1).reshape(res, res)
            m = DiffusersAttentionExtractor._resize(m, target, target)
            acc = m if acc is None else acc + m
            count += 1
        if acc is None or count == 0:
            raise RuntimeError("subject tokens not found in captured attention")
        return acc / count

    @staticmethod
    def _resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """Bilinear-ish resize using NumPy (no PIL dependency on the hot path)."""
        ah, aw = arr.shape
        ys = np.linspace(0, ah - 1, h)
        xs = np.linspace(0, aw - 1, w)
        y0 = np.floor(ys).astype(int)
        y1 = np.minimum(y0 + 1, ah - 1)
        x0 = np.floor(xs).astype(int)
        x1 = np.minimum(x0 + 1, aw - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]
        top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
        bot = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
        return top * (1 - wy) + bot * wy
