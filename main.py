import cv2, time
from PPGMap import utils
from PPGMap.chrom import get_chrom_value
import numpy as np
from scipy import signal

WINDOW_TITLE = 'Face_Detector'
haar_cascade_path = "../haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
Debug = True
part_size = 128
img_list = []
fs = 30
mean_colors = []
timestamps = []


def run_observer(webcam):
    fps = int(webcam.get(cv2.CAP_PROP_FPS))
    while cv2.getWindowProperty(WINDOW_TITLE, 0) == 0:
        ret, img_frame = webcam.read()
        frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

        face_list = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if len(face_list) > 0:
            face_box = face_list[0]
            # utils.draw_face_roi(face_box, img_frame)
            face = utils.crop_to_boundingbox(face_box, img_frame)
            if face.shape[0] > 0 and face.shape[1] > 0:
                get_chrom_map(face)

        cv2.imshow('Camera', img_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def get_chrom_map(face_roi):
    # 分成32个区域
    height = face_roi.shape[0]
    sub_area_height = int(height/32)
    timestamps += [time.time()]
    # utils.draw_face_roi(face_box, img_frame)
    t = np.arange(timestamps[0], timestamps[-1], 1 / fs)
    mean_colors_resampled = np.zeros((3, t.shape[0]))

    chrom_arr = np.zeros((32, 3, t.shape[0]))

    for i in range(32):
        if len(mean_colors) < (i + 1):
            mean_colors.append([])

        sub_area = face_roi[(i*sub_area_height):((i+1)*sub_area_height), :, :]
        # if Debug:
        #     x_start = 0
        #     y = (i+1)*sub_area_height
        #     x_end = face_roi.shape[1]
        #     cv2.line(img, (x_start, y), (x_end, y), (255, 0, 0))

        mean_colors[i] += [sub_area.mean(axis=0).mean(axis=0)]

        for j in range(3):
            resampled = np.interp(t, timestamps, np.array(mean_colors[i])[:, j])
            mean_colors_resampled[j] = resampled
        # 记录
        chrom_arr[i] = mean_colors_resampled

    if t.shape[0] > part_size:
        chrom_values = []
        for i in range(32):
            cur_mean_colors = chrom_arr[i]
            chrom_value = get_chrom_value(cur_mean_colors, part_size)
            chrom_values.append(chrom_value)
        # 归一化
        img_values = utils.normalize(chrom_value)
        return img_values


# Clean up
def shut_down(webcam):
    webcam.release()
    cv2.destroyAllWindows()
    exit(0)


def split_img(path):
    img = cv2.imread(path)
    get_chrom_map(img, img)
    cv2.imshow('ww', img)
    cv2.waitKey(0)


def main():
    webcam = cv2.VideoCapture('../video/2.mp4')
    if not webcam.isOpened():
        print('ERROR:  Unable to open webcam.  Verify that webcam is connected and try again.  Exiting.')
        webcam.release()
        return

    cv2.namedWindow(WINDOW_TITLE)
    cv2.resizeWindow(WINDOW_TITLE, 640, 480)
    run_observer(webcam)
    shut_down(webcam)


if __name__ == '__main__':
    split_img('../0.jpg')
