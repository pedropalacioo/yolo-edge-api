import cv2
import numpy as np

from preprocessing.utils.evaluate import evaluate_pipeline


def dark(image):
    table = ((np.arange(256) / 255.0) ** 1.8 * 255).astype("uint8")
    return cv2.LUT(image, table)


def equalize(image):
    lab = cv2.cvtColor(dark(image), cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((cv2.equalizeHist(luminance), a, b)), cv2.COLOR_LAB2BGR)


def clahe(image):
    lab = cv2.cvtColor(dark(image), cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)
    operator = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((operator.apply(luminance), a, b)), cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    evaluate_pipeline("E4-A", dark)
    evaluate_pipeline("E4-B", equalize)
    evaluate_pipeline("E4-C", clahe)
