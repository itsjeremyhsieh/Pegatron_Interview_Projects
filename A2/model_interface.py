from PIL import Image
import base64
import io
from openai import OpenAI


class BenchmarkModel:
    """
    OpenAI-based multimodal inference using a vision-capable chat model.
    Set OPENAI_API_KEY in the environment.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name or "gpt-4o-mini"
        print(f"Using OpenAI model: {self.model_name}")
        self.client = OpenAI()

    def _pil_to_data_url(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def predict(self, question: str, image: Image.Image, options: list[str]):
        # Build options text A., B., ...
        options_text = "".join(
            [f"{chr(ord('A')+i)}. {opt}\n" for i, opt in enumerate(options)])

        prompt = (
            "You are answering a multiple-choice question.\n"
            "Return EXACTLY ONE letter from [A|B|C|D|E|F]. No other text.\n\n"
            f"Question: {question}\n\nOptions:\n{options_text}\n"
            "Answer (one letter only):"
        )

        img_url = self._pil_to_data_url(image) if isinstance(
            image, Image.Image) else None

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ] + ([{"type": "image_url", "image_url": {"url": img_url}}] if img_url else []),
                }
            ]

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            result = resp.choices[0].message.content or ""
            print(f"Detail - OpenAI response: {result}")
            return result
        except Exception as e:
            print(f"Debug - OpenAI prediction error: {e}")
            return ""
