import cv2, os
from scipy import signal
from numpy.linalg import norm
import numpy as np

fs = 30
bpf_div = 60 * fs / 2
b_BPF40220, a_BPF40220 = signal.butter(10, ([40 / bpf_div, 220 / bpf_div]), 'bandpass')


def bandpass_filter(sig):
    return signal.filtfilt(b_BPF40220, a_BPF40220, sig)


def crop_to_boundingbox(bb, frame):
    y, h, x, w = [int(c) for c in bb]
    return frame[y:y + h, x:x + w]


def draw_face_roi(face, img):
    try:
        x, y, w, h = [int(c) for c in face]
        delta = int(w * 0.2)
        thickness = int(w * 0.025)
        color = (255, 0, 0)

        image = cv2.line(img, (x, y), (x + delta, y), color, thickness)
        image = cv2.line(image, (x, y), (x, y + delta), color, thickness)

        image = cv2.line(img, (x + w, y), (x + w - delta, y), color, thickness)
        image = cv2.line(image, (x + w, y), (x + w, y + delta), color, thickness)

        image = cv2.line(img, (x, y + h), (x + delta, y + h), color, thickness)
        image = cv2.line(image, (x, y + h), (x, y + h - delta), color, thickness)

        image = cv2.line(img, (x + w, y + h), (x + w - delta, y + h), color, thickness)
        image = cv2.line(image, (x + w, y + h), (x + w, y + h - delta), color, thickness)
    except:
        pass


def normalize(data):
    _range = np.max(data) - np.min(data)
    return round((data - np.min(data)) / _range) * 255

