"""
Vision-language model for interpreting objects and environment context.
Supports BLIP-2 (lightweight) and optional LLaVA-style VLMs via transformers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class VisionLanguageModel:
    """
    Wraps a VLM to answer questions about an image (e.g. "What objects are visible?",
    "Describe the scene.", "Is there a person?").
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[str] = None,
        use_float16: bool = True,
    ):
        self._model_name = model_name
        self._device = device
        self._use_float16 = use_float16
        self._model = None
        self._processor = None

    def load(self) -> None:
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "Vision-language requires transformers and torch. "
                "Install: pip install transformers torch"
            )
        if "blip2" in self._model_name.lower():
            self._processor = Blip2Processor.from_pretrained(self._model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(self._model_name)
        else:
            raise ValueError(f"Unsupported VLM: {self._model_name}")
        self._torch = torch
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(dev)
        if self._use_float16 and dev == "cuda":
            self._model = self._model.half()
        self._device = dev

    def describe(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        """
        Describe the image. If prompt is None, uses default caption.
        image: BGR (H, W, 3) numpy array.
        """
        if self._model is None:
            self.load()
        from PIL import Image
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        if prompt:
            inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt").to(self._device)
            if self._use_float16 and self._device == "cuda":
                inputs = {k: v.half() if hasattr(v, "half") else v for k, v in inputs.items()}
            out = self._model.generate(**inputs, max_new_tokens=100)
        else:
            inputs = self._processor(images=pil_image, return_tensors="pt").to(self._device)
            if self._use_float16 and self._device == "cuda":
                inputs = {k: v.half() if hasattr(v, "half") else v for k, v in inputs.items()}
            out = self._model.generate(**inputs, max_new_tokens=100)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()

    def query(self, image: np.ndarray, question: str) -> str:
        """Answer a question about the image (e.g. 'What objects are in this scene?')."""
        return self.describe(image, prompt=question)


def create_vlm(
    model_name: str = "Salesforce/blip2-opt-2.7b",
    device: Optional[str] = None,
) -> VisionLanguageModel:
    """Factory for VLM. Use 'Salesforce/blip2-opt-2.7b' for GPU or 'Salesforce/blip2-opt-2.7b-coco' for smaller."""
    return VisionLanguageModel(model_name=model_name, device=device)
