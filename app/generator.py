"""Dedicated Image Generation & Scene Synthesis API Service (`api.py`).

Provides text-to-image synthesis, stylized scene generation, and background creation.
Architecture:
- Pluggable multi-provider engine (`ImageGeneratorService`)
- Tier 1: Gemini Image Generation (Google GenAI SDK via gemini-3.1-flash-image / gemini-2.5-flash-image)
- Tier 2: Secondary Cloud Text-to-Image Generation (Pollinations AI)
- Tier 3: Offline Semantic Vector/Raster Synthesizer (Pillow + OpenCV)
"""
from __future__ import annotations
import io
import math
import os
import random
import re
import urllib.parse
import urllib.request
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _load_env_file():
    from pathlib import Path
    env_p = Path(__file__).resolve().parent.parent / ".env"
    if env_p.exists():
        try:
            for line in env_p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


class ImageGeneratorService:
    def __init__(self, api_key: Optional[str] = None):
        _load_env_file()
        self.api_key = api_key

    def _get_api_key(self) -> str:
        """Fetch API key dynamically from environment on every request."""
        key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        return key

    def generate_image(
        self,
        prompt: str,
        style: str = "realistic",
        width: int = 1024,
        height: int = 768,
        negative_prompt: Optional[str] = None
    ) -> Tuple[bytes, dict]:
        """Generate an image using Tier 1 (Gemini AI) -> Tier 2 (Pollinations) -> Tier 3 (Offline Synthesizer).
        Returns (png_bytes, metadata_dict).
        """
        # 1. Try Google Gemini Image Generation
        try:
            return self._generate_with_google_gemini(prompt, style, width, height)
        except Exception as e:
            print(f"[IMAGE_GENERATOR_ERROR] Google Gemini image generation failed ({e}). Falling back to Pollinations AI...")

        # 2. Try Pollinations AI (Cloud Diffusion Fallback)
        try:
            return self._generate_with_pollinations(prompt, style, width, height)
        except Exception as e:
            print(f"[IMAGE_GENERATOR_ERROR] Pollinations AI failed ({e}). Falling back to local synthesizer...")

        # 3. Offline Local Synthesizer
        return self._generate_local_scene(prompt, style, width, height)

    def generate_background(
        self,
        prompt: str,
        style: str = "realistic",
        width: int = 1024,
        height: int = 768
    ) -> Tuple[bytes, dict]:
        """Generate an environment or background for composite layouts."""
        return self.generate_image(
            prompt=prompt,
            style=style,
            width=width,
            height=height
        )

    def _generate_with_google_gemini(self, prompt: str, style: str, width: int, height: int) -> Tuple[bytes, dict]:
        """Generate image using Google AI Studio Gemini API models."""
        from google import genai
        from google.genai import types

        api_key = self._get_api_key()
        client = genai.Client(api_key=api_key)
        enhanced_prompt = self._enhance_prompt_for_style(prompt, style)
        aspect_ratio = "4:3" if width > height else ("3:4" if height > width else "1:1")

        # Active Gemini image generation models
        gemini_image_models = [
            "gemini-3.1-flash-image",
            "gemini-2.5-flash-image",
            "gemini-3-pro-image"
        ]

        last_err = None
        for model_name in gemini_image_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=enhanced_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio
                        ) if hasattr(types, "ImageConfig") else None
                    )
                )

                # Extract image bytes from returned content parts
                if hasattr(response, "candidates") and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            return part.inline_data.data, {
                                "provider": f"google-{model_name}",
                                "model": model_name,
                                "prompt": prompt,
                                "style": style,
                                "width": width,
                                "height": height
                            }
            except Exception as ex:
                last_err = ex
                continue

        # Fallback for Vertex AI / Enterprise Mode with Imagen 3
        try:
            result = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=enhanced_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    output_mime_type="image/png"
                )
            )
            img_bytes = result.generated_images[0].image.image_bytes
            return img_bytes, {
                "provider": "google-imagen-3.0",
                "model": "imagen-3.0-generate-002",
                "prompt": prompt,
                "style": style,
                "width": width,
                "height": height
            }
        except Exception as ex:
            last_err = ex

        raise RuntimeError(f"Google Gemini image generation failed. Last error: {last_err}")

    def _generate_with_pollinations(self, prompt: str, style: str, width: int, height: int) -> Tuple[bytes, dict]:
        """Generate using Pollinations AI (free cloud Flux/SDXL image synthesis)."""
        enhanced_prompt = self._enhance_prompt_for_style(prompt, style)
        encoded = urllib.parse.quote(enhanced_prompt)
        w = max(512, min(1024, width))
        h = max(384, min(1024, height))
        seed = random.randint(1000, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&seed={seed}&nologo=true"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ImageCatalog/2.0 (Modern AI Compositing Assistant)"}
        )
        with urllib.request.urlopen(req, timeout=18) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Pollinations returned status {resp.status}")
            raw_bytes = resp.read()

        pil_im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if pil_im.size != (width, height):
            pil_im = pil_im.resize((width, height), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil_im.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return png_bytes, {
            "provider": "pollinations-ai",
            "model": "flux",
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "style": style,
            "width": width,
            "height": height
        }

    def _enhance_prompt_for_style(self, prompt: str, style: str) -> str:
        s = style.lower()
        if "cartoon" in s:
            return f"{prompt}, vibrant colorful 2D cartoon style, crisp outlines, vector illustration, Disney animation aesthetic, clean flat colors"
        elif "illustration" in s:
            return f"{prompt}, gorgeous digital art illustration, painterly textures, rich lighting, artistic aesthetic, masterpiece"
        elif "product" in s or "mockup" in s:
            return f"{prompt}, professional product photography mockup, clean commercial studio backdrop, softbox lighting, high-end catalog"
        else:
            return f"{prompt}, photorealistic, detailed photograph, 8k resolution, cinematic lighting, sharp focus"

    def _generate_local_scene(self, prompt: str, style: str, width: int, height: int) -> Tuple[bytes, dict]:
        """Synthesize a visually appealing semantic scene locally when offline."""
        p_lower = prompt.lower()
        im = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(im)

        # Base Palette Selection
        if "futuristic" in p_lower or "neon" in p_lower or "cyber" in p_lower:
            top_color, mid_color, bot_color = (15, 10, 35), (40, 15, 70), (10, 25, 45)
            accent, accent2 = (0, 240, 255), (255, 0, 128)
        elif "garden" in p_lower or "park" in p_lower or "nature" in p_lower or "outdoor" in p_lower:
            top_color, mid_color, bot_color = (135, 206, 235), (180, 225, 200), (34, 139, 34)
            accent, accent2 = (255, 215, 0), (255, 105, 180)
        elif "urban" in p_lower or "street" in p_lower or "dark" in p_lower or "black" in p_lower:
            top_color, mid_color, bot_color = (25, 28, 35), (45, 48, 55), (20, 22, 26)
            accent, accent2 = (220, 220, 220), (255, 165, 0)
        elif "t-shirt" in p_lower or "mockup" in p_lower:
            top_color, mid_color, bot_color = (245, 246, 248), (235, 238, 242), (220, 224, 230)
            accent, accent2 = (50, 50, 60), (100, 120, 150)
        elif "table" in p_lower or "wood" in p_lower or "fruit" in p_lower or "banana" in p_lower or "apple" in p_lower:
            top_color, mid_color, bot_color = (240, 230, 215), (210, 185, 155), (140, 100, 65)
            accent, accent2 = (255, 215, 0), (220, 50, 50)
        else:
            top_color, mid_color, bot_color = (230, 238, 245), (205, 220, 235), (170, 185, 205)
            accent, accent2 = (108, 99, 255), (249, 115, 22)

        # Smooth vertical gradient backdrop
        for y in range(height):
            t = y / max(1, height - 1)
            if t < 0.5:
                sub_t = t * 2.0
                r = int(top_color[0] * (1 - sub_t) + mid_color[0] * sub_t)
                g = int(top_color[1] * (1 - sub_t) + mid_color[1] * sub_t)
                b = int(top_color[2] * (1 - sub_t) + mid_color[2] * sub_t)
            else:
                sub_t = (t - 0.5) * 2.0
                r = int(mid_color[0] * (1 - sub_t) + bot_color[0] * sub_t)
                g = int(mid_color[1] * (1 - sub_t) + bot_color[1] * sub_t)
                b = int(mid_color[2] * (1 - sub_t) + bot_color[2] * sub_t)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

        # Scene Backdrops
        if "garden" in p_lower or "park" in p_lower:
            draw.ellipse([width * 0.75, height * 0.08, width * 0.9, height * 0.28], fill=(255, 245, 170, 240))
            pts = [(0, int(height * 0.62))]
            for x in range(0, width + 50, 50):
                pts.append((x, int(height * 0.58 + 25 * math.sin(x * 0.015))))
            pts.extend([(width, height), (0, height)])
            draw.polygon(pts, fill=(46, 125, 50, 255))
            rng = random.Random(42)
            for _ in range(30):
                fx = rng.randint(20, width - 20)
                fy = rng.randint(int(height * 0.68), height - 20)
                draw.ellipse([fx - 6, fy - 6, fx + 6, fy + 6], fill=(255, 105, 180, 255))

        elif "futuristic" in p_lower or "neon" in p_lower or "cyber" in p_lower:
            horizon = int(height * 0.6)
            for x in range(0, width, 60):
                draw.line([(x, horizon), (int(width * 0.5 + (x - width * 0.5) * 2.5), height)], fill=accent, width=1)
            for y_step in range(horizon, height, 25):
                draw.line([(0, y_step), (width, y_step)], fill=accent2, width=1)
            draw.ellipse([int(width * 0.4), int(height * 0.2), int(width * 0.6), int(height * 0.47)], outline=accent2, width=4)

        # Semantic Object Rendering
        cx = width // 2
        cy = int(height * 0.55)

        if any(w in p_lower for w in ("banana", "apple", "orange", "fruit", "table")):
            table_y = int(height * 0.52)
            draw.rectangle([0, table_y, width, height], fill=(139, 90, 44, 255))
            draw.line([(0, table_y), (width, table_y)], fill=(180, 120, 70, 255), width=4)

        if "banana" in p_lower:
            bx = cx + 80 if "apple" in p_lower else cx
            by = cy
            b_pts = []
            for angle_deg in range(30, 150, 5):
                rad = math.radians(angle_deg)
                px = bx + math.cos(rad) * 140
                py = by + math.sin(rad) * 90 - 40
                b_pts.append((px, py))
            for angle_deg in range(145, 25, -5):
                rad = math.radians(angle_deg)
                px = bx + math.cos(rad) * 155
                py = by + math.sin(rad) * 115 - 40
                b_pts.append((px, py))
            if len(b_pts) > 4:
                draw.ellipse([bx - 120, by + 40, bx + 120, by + 65], fill=(40, 25, 10, 100))
                draw.polygon(b_pts, fill=(255, 225, 53, 255), outline=(218, 165, 32, 255))
                if b_pts:
                    draw.ellipse([b_pts[0][0] - 8, b_pts[0][1] - 8, b_pts[0][0] + 8, b_pts[0][1] + 8], fill=(90, 55, 20, 255))
                    draw.ellipse([b_pts[-1][0] - 6, b_pts[-1][1] - 6, b_pts[-1][0] + 6, b_pts[-1][1] + 6], fill=(70, 110, 30, 255))

        if "apple" in p_lower:
            ax = cx - 90 if "banana" in p_lower else cx
            ay = cy
            draw.ellipse([ax - 55, ay + 35, ax + 55, ay + 55], fill=(40, 25, 10, 100))
            draw.ellipse([ax - 50, ay - 45, ax + 50, ay + 45], fill=(220, 38, 38, 255), outline=(185, 28, 28, 255))
            draw.line([(ax, ay - 45), (ax + 5, ay - 68)], fill=(101, 67, 33, 255), width=4)
            draw.polygon([(ax + 5, ay - 62), (ax + 24, ay - 75), (ax + 20, ay - 60)], fill=(34, 197, 94, 255))

        if "girl" in p_lower or "person" in p_lower or "boy" in p_lower:
            gx, gy = cx, int(height * 0.48)
            draw.polygon([(gx - 45, gy + 110), (gx + 45, gy + 110), (gx, gy + 20)], fill=(236, 72, 153, 255) if "girl" in p_lower else (59, 130, 246, 255))
            draw.ellipse([gx - 25, gy - 30, gx + 25, gy + 20], fill=(254, 215, 170, 255))
            if "hat" in p_lower:
                hat_color = (220, 38, 38, 255) if "red" in p_lower else (234, 179, 8, 255)
                draw.ellipse([gx - 45, gy - 38, gx + 45, gy - 20], fill=hat_color)
                draw.rectangle([gx - 22, gy - 55, gx + 22, gy - 32], fill=hat_color)

        # Style post-processing
        rgb_im = im.convert("RGB")
        np_arr = np.array(rgb_im)
        if "cartoon" in style.lower():
            bgr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
            filtered = cv2.bilateralFilter(bgr, 9, 75, 75)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            np_arr = cv2.cvtColor(cv2.bitwise_and(filtered, edges_bgr), cv2.COLOR_BGR2RGB)
        elif "illustration" in style.lower():
            bgr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
            blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
            np_arr = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        out_im = Image.fromarray(np_arr)
        buf = io.BytesIO()
        out_im.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return png_bytes, {
            "provider": "local-semantic-synthesizer",
            "prompt": prompt,
            "style": style,
            "width": width,
            "height": height
        }


# Global default service instance
IMAGE_GENERATOR = ImageGeneratorService()
