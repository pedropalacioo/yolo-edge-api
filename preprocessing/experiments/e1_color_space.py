import cv2

from preprocessing.utils.evaluate import evaluate_pipeline

if __name__ == "__main__":
    # Files are decoded as BGR by Ultralytics. Swapping before saving makes its
    # later BGR->RGB conversion feed the original BGR order to the network.
    evaluate_pipeline("E1-A", lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    evaluate_pipeline("E1-B", lambda image: image.copy())
