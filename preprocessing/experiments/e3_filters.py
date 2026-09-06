import cv2

from preprocessing.utils.evaluate import evaluate_pipeline

if __name__ == "__main__":
    evaluate_pipeline("E3-A", lambda image: image.copy())
    evaluate_pipeline("E3-B", lambda image: cv2.GaussianBlur(image, (3, 3), 0.8))
    evaluate_pipeline("E3-C", lambda image: cv2.GaussianBlur(image, (5, 5), 1.5))
    evaluate_pipeline("E3-D", lambda image: cv2.medianBlur(image, 3))
