import cv2

from preprocessing.utils.evaluate import evaluate_pipeline

if __name__ == "__main__":
    evaluate_pipeline("E2-A", lambda image: cv2.resize(image, (640, 640)))
    # Ultralytics' validated default path performs the reference letterbox and
    # adjusts labels internally, so the original dataset is the correct E2-B.
    evaluate_pipeline("E2-B")
