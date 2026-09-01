import base64
import json
import os
import sys

import requests


API_URL = os.getenv("API_URL", "http://localhost:8000")
IMAGE_PATH = os.getenv("IMAGE_PATH", "/client/images/bus.jpg")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/client/output/result.json")


def encode_image(image_path: str) -> str:
    """Lê uma imagem e converte seu conteúdo para Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    print(f"[INFO] API: {API_URL}")
    print(f"[INFO] Imagem: {IMAGE_PATH}")

    if not os.path.exists(IMAGE_PATH):
        print(f"[ERRO] Imagem não encontrada: {IMAGE_PATH}")
        sys.exit(1)

    try:
        image_base64 = encode_image(IMAGE_PATH)

        payload = {
            "image_base64": image_base64,
            "confidence": 0.25,
            "model_name": "yolov8n.pt",
        }

        print("[INFO] Enviando imagem para /predict...")

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        print("\n[OK] Inferência concluída!")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        os.makedirs(
            os.path.dirname(OUTPUT_PATH),
            exist_ok=True,
        )

        with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
            json.dump(
                result,
                output_file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n[OK] Resultado salvo em: {OUTPUT_PATH}")

    except requests.RequestException as error:
        print(f"[ERRO] Falha na comunicação com a API: {error}")
        sys.exit(1)

    except Exception as error:
        print(f"[ERRO] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
