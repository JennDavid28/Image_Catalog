"""Intelligent Natural-Language Prompt Understanding & Planning Layer.

Uses Google Gemini Flash API (or configured LLM / local heuristic fallback) to:
1. Interpret user intent and determine the optimal generation mode:
   - Mode 1: Assets only (collage composition using existing catalog items)
   - Mode 2: Hybrid (catalog assets + AI generated background/elements)
   - Mode 3: Pure AI generation (complete semantic scene synthesized from prompt)
2. Extract objects, colors, counts, sizes, relations, anchors, style, and background.
3. Map entities to existing catalog assets with fuzzy/semantic matching.
4. Detect ambiguity or missing assets and ask clarification questions when necessary.
5. Produce a structured JSON plan ready for execution.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are an expert AI Composition & Image Generation Planner for an Image Catalog system.
Your job is to interpret user natural-language prompts and create a structured JSON plan.

The system supports three image generation modes:
- Mode 1 ("assets_only"): The user wants to build an image purely from existing catalog items (e.g. "place my logo on the shirt", "an apple beside an orange"). No AI-generated stand-in pixels.
- Mode 2 ("hybrid"): The user wants to combine fixed catalog assets (e.g. uploaded logo, a cartoon girl) with newly generated background or decorative elements (e.g. "put my logo on a futuristic neon background", "place the boy in a sunny park with flowers").
- Mode 3 ("ai_generated"): The user wants a completely new semantic scene created with text-to-image AI (e.g. "a girl wearing a red hat standing in a garden", "a watercolor painting of a mountain").

Catalog Assets Available:
{catalog_assets}

Instructions:
1. Determine `generation_mode`: "assets_only", "hybrid", or "ai_generated".
2. If the user explicitly asks for catalog assets or pure composition without AI background, pick "assets_only".
3. If the user wants to use a catalog asset (like a logo, boy, apple) but also requested a new background, environment, or decorative elements that do NOT exist in the catalog, pick "hybrid".
4. If the user describes a cohesive new scene from scratch without referencing specific catalog assets, pick "ai_generated".
5. Match requested items to catalog assets when applicable. If a requested catalog asset is ambiguous or missing in Mode 1 or 2, set `clarification_needed` to true and provide `clarification_question` and `clarification_options`.
6. For Mode 2 and 3, provide `ai_prompt` and artistic `ai_style` ("realistic", "cartoon", "illustration", "product mockup").
7. Produce a clean JSON object conforming strictly to this schema:
{{
  "generation_mode": "assets_only" | "hybrid" | "ai_generated",
  "explanation": "short explanation of why this mode was chosen",
  "clarification_needed": false | true,
  "clarification_question": "question text if ambiguous or missing, or null",
  "clarification_options": ["option 1", "option 2"] or [],
  "canvas": {{
    "width": 1024,
    "height": 768,
    "background_type": "color" | "transparent" | "ai_generated" | "catalog_image",
    "background_value": "white" | "#hex" | "ai prompt description"
  }},
  "ai_generation": {{
    "prompt": "detailed prompt for the image generator",
    "style": "realistic" | "cartoon" | "illustration" | "product mockup",
    "role": "background" | "full_scene" | "elements"
  }} or null,
  "asset_placements": [
    {{
      "asset_query": "apple",
      "matched_asset_id": "optional id",
      "matched_label": "apple",
      "matched_image_name": "apple",
      "count": 1,
      "color": "red",
      "size": 1.0,
      "anchor": "left" | "right" | "center" | "top" | "bottom" | null,
      "relation": "adjacent" | "left_of" | "right_of" | "above" | "below" | "in_front" | "behind",
      "locked": false
    }}
  ]
}}
Return ONLY JSON without markdown fences.
"""


def _load_env_file():
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


def _get_api_key() -> Optional[str]:
    _load_env_file()
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    )


def _call_gemini_flash(prompt: str, api_key: str, model_name: str = DEFAULT_GEMINI_MODEL) -> str:
    """Call Google Gemini Flash via modern google-genai SDK."""
    from google import genai
    client = genai.Client(api_key=api_key)

    for m in [model_name, "gemini-2.5-flash", "gemini-1.5-flash"]:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
            )
            if response and response.text:
                return response.text
        except Exception:
            continue

    raise RuntimeError("google-genai could not fulfill the request with the configured model.")


def format_catalog_summary(catalog, user_id: str | None = None) -> str:
    """Summarize available catalog assets for the LLM context."""
    if catalog is None:
        return "Catalog is empty."
    live = [o for o in catalog.live_objects(user_id) if o.get("image_kind") == "source" and not o.get("deleted")]
    if not live:
        return "No source assets currently in catalog."
    summary_lines = []
    seen = set()
    for o in live:
        key = (o["label"], o.get("color"), o.get("image_name"))
        if key not in seen:
            seen.add(key)
            color_str = f" (color: {o['color']})" if o.get("color") else ""
            summary_lines.append(f"- '{o['label']}'{color_str} from image '{o['image_name']}' [id: {o['id']}]")
    return "\n".join(summary_lines)


def parse_prompt_heuristic(instruction: str, catalog=None, user_id: str | None = None) -> dict:
    """High-quality deterministic fallback planner when no LLM key is configured."""
    text = instruction.strip().lower()
    live_objects = [o for o in catalog.live_objects(user_id) if o.get("image_kind") == "source" and not o.get("deleted")] if catalog else []

    ai_keywords = [
        "generate a", "create a picture of", "realistic", "photograph", "scene",
        "standing in", "wearing a", "garden", "futuristic", "background", "landscape", "painting"
    ]
    has_ai_intent = any(k in text for k in ai_keywords)

    matched_placements = []

    for obj in live_objects:
        lbl = obj["label"].lower()
        if lbl in text and not any(p["matched_asset_id"] == obj["id"] for p in matched_placements):
            obj_color = (obj.get("color") or "").lower()
            color = obj_color if obj_color and obj_color in text else None
            matched_placements.append({
                "asset_query": lbl,
                "matched_asset_id": obj["id"],
                "matched_label": obj["label"],
                "matched_image_name": obj["image_name"],
                "count": 1,
                "color": obj["color"],
                "size": 1.0,
                "anchor": "left" if "on the left" in text else ("right" if "on the right" in text else "center"),
                "relation": "adjacent",
                "locked": "logo" in lbl or "logo" in text
            })

    if matched_placements and ("background" in text or "futuristic" in text or "park" in text or "garden" in text or "urban" in text):
        mode = "hybrid"
        explanation = "Detected catalog assets combined with a request for a newly generated environment/background."
        ai_gen = {
            "prompt": instruction,
            "style": "product mockup" if "t-shirt" in text or "logo" in text else "realistic",
            "role": "background"
        }
    elif matched_placements and not has_ai_intent:
        mode = "assets_only"
        explanation = "User requested composition of existing catalog assets."
        ai_gen = None
    elif not matched_placements and has_ai_intent:
        mode = "ai_generated"
        explanation = "User requested a completely new scene synthesized with AI."
        ai_gen = {
            "prompt": instruction,
            "style": "cartoon" if "cartoon" in text else "realistic",
            "role": "full_scene"
        }
    else:
        if matched_placements:
            mode = "hybrid" if ("with" in text and len(matched_placements) < 2) else "assets_only"
            explanation = "Using matched catalog assets with compositing."
            ai_gen = {"prompt": instruction, "style": "realistic", "role": "background"} if mode == "hybrid" else None
        else:
            mode = "ai_generated"
            explanation = "No matching catalog assets found; generating new image with AI."
            ai_gen = {"prompt": instruction, "style": "realistic", "role": "full_scene"}

    clarification_needed = False
    clarification_q = None
    clarification_opts = []
    if "logo" in text and not any("logo" in p["asset_query"] for p in matched_placements):
        clarification_needed = True
        clarification_q = "You mentioned 'logo', but no logo image was found in your catalog. Would you like to upload one or continue?"
        clarification_opts = ["Upload logo to catalog", "Generate a placeholder logo with AI", "Continue without logo"]
    elif any(phrase in text for phrase in ("place it", "put it", "move it")) or text.strip() in ("it", "left", "right"):
        clarification_needed = True
        clarification_q = "Which catalog asset would you like to place?"
        clarification_opts = [o["label"] for o in live_objects[:4]] if live_objects else ["Upload an asset", "Specify object name"]

    return {
        "generation_mode": mode,
        "explanation": explanation,
        "clarification_needed": clarification_needed,
        "clarification_question": clarification_q,
        "clarification_options": clarification_opts,
        "canvas": {
            "width": 1024,
            "height": 768,
            "background_type": "ai_generated" if mode in ("hybrid", "ai_generated") else "color",
            "background_value": "white" if mode == "assets_only" else instruction
        },
        "ai_generation": ai_gen,
        "asset_placements": matched_placements,
        "_planner": "heuristic"
    }


def plan_prompt(instruction: str, catalog=None, model_name: str = DEFAULT_GEMINI_MODEL, api_key: Optional[str] = None,
                user_id: str | None = None) -> dict:
    """Main entrypoint: interpret prompt into structured plan using Gemini Flash or heuristic fallback."""
    key = api_key or _get_api_key()
    catalog_summary = format_catalog_summary(catalog, user_id)

    if key:
        full_prompt = SYSTEM_PROMPT.format(catalog_assets=catalog_summary) + f"\n\nUser Request: {instruction}\nJSON Response:"
        try:
            raw_text = _call_gemini_flash(full_prompt, key, model_name)
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                data["_planner"] = f"gemini ({model_name})"
                data.setdefault("generation_mode", "assets_only")
                data.setdefault("clarification_needed", False)
                data.setdefault("canvas", {"width": 1024, "height": 768, "background_type": "color", "background_value": "white"})
                data.setdefault("asset_placements", [])
                return data
        except Exception as e:
            fallback = parse_prompt_heuristic(instruction, catalog, user_id)
            fallback["explanation"] += f" (LLM note: {type(e).__name__}, used heuristic planner)"
            return fallback

    return parse_prompt_heuristic(instruction, catalog, user_id)
