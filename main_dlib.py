import cv2, time
from PPGMap import utils
from PPGMap.chrom import get_chrom_value
import numpy as np
import dlib
import os
from scipy import signal
from PPGMap.delaunay import perspective_transoform

Debug = True
part_size = 128
img_list = []
fs = 30
mean_colors = []
timestamps = []
PICTURE_INDEX = 10


def run_observer(directory, filename):
    path = os.path.join(directory, filename)
    webcam = cv2.VideoCapture(path)
    fps = int(webcam.get(cv2.CAP_PROP_FPS))

    detector = dlib.get_frontal_face_detector()

    try:
        predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')
    except RuntimeError as e:
        print('ERROR:  \'shape_predictor_68_face_landmarks.dat\' was not found in current directory.   ' \
              'Download it from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2')
        return

    while True:
        ret, img_frame = webcam.read()
        points = []

        faces = detector(img_frame, 0)
        if len(faces) >= 1:
            face_points = predictor(img_frame, faces[0])
            # print(face_points.parts())
            for part in face_points.parts():
                points.append((int(part.x), int(part.y)))
        if points:
            m_pts = [points[36], points[45], points[10], points[6]]
            # points = points[27:36]
            face_box = [m_pts[0][0], m_pts[0][1], points[35][0] - points[39][0], points[31][1] - points[39][1]]
            # face = utils.crop_to_boundingbox(face_box, img_frame)
            face_roi = perspective_transoform(img_frame, m_pts)
            utils.draw_face_roi(face_box, img_frame)
            map_values = get_chrom_map(face_roi)
            if map_values is not None:
                cv2.imshow('map', map_values)
                path_2 = directory + '_map/' + filename.split('.')[0] + '.jpg'
                cv2.imwrite(path_2, map_values)
                global mean_colors, timestamps
                mean_colors = []
                timestamps = []
                break
        # img_frame = cv2.resize(img_frame, (320, 480))
        # cv2.imshow('Camera', img_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cv2.waitKey(10)
    return webcam


def get_chrom_map(face_roi, fps=None):
    cv2.imshow('face', face_roi)
    # 分成32个区域
    height = face_roi.shape[0]
    sub_area_height = int(height / 32)
    global timestamps
    timestamps += [time.time()]
    # utils.draw_face_roi(face_box, img_frame)
    if not fps:
        time_elapsed = timestamps[-1] - timestamps[0]
        fps = (len(timestamps) / time_elapsed) if time_elapsed else 1
    t = np.arange(timestamps[0], timestamps[-1], 1 / fps)
    mean_colors_resampled = np.zeros((3, t.shape[0]))

    chrom_arr = np.zeros((32, 3, t.shape[0]))

    for i in range(32):
        if len(mean_colors) < (i + 1):
            mean_colors.append([])

        sub_area = face_roi[(i * sub_area_height):((i + 1) * sub_area_height), :, :]
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
        welch_values = []
        for i in range(32):
            cur_mean_colors = chrom_arr[i]
            chrom_value = get_chrom_value(cur_mean_colors, part_size)
            welch_value = get_welch_value(chrom_value, fps)
            chrom_values.append(chrom_value)
            welch_values.append(welch_value)
        # 归一化
        img_values = chrom_value + welch_value
        img_values = utils.normalize(img_values)
        img_values = img_values.astype(np.uint8)
        return img_values
    return None


def get_welch_value(x, fs):
    # print(x.shape)
    f, Pxx_spec = signal.welch(x, fs,  window='hann', nperseg=1, scaling='density')
    print(Pxx_spec.shape)
    return Pxx_spec


def matlab_ppg(origin):
    origin = np.array(origin)
    origin = np.transpose(origin)
    xs = 0.77 * origin[:, 0] - 0.51 * origin[:, 1]
    ys = 0.77 * origin[:, 0] + 0.51 * origin[:, 1] - 0.77 * origin[:, 2]
    alpha = std_ratio_sliding_win(xs, ys)
    iPPG = xs - alpha * ys
    # print(iPPG)
    return iPPG


def std_ratio_sliding_win(x, y):
    nx = np.std(x)
    ny = np.std(y)
    alpha_chrom = nx / ny
    return alpha_chrom


# Clean up
def shut_down(webcam):
    webcam.release()
    cv2.destroyAllWindows()
    exit(0)


def main(directory, filename):
    # path = os.path.join(directory, filename)
    # webcam = cv2.VideoCapture(path)
    # if not webcam.isOpened():
    #     print('ERROR:  Unable to open webcam.  Verify that webcam is connected and try again.  Exiting.')
    #     webcam.release()
    #     return

    webcam = run_observer(directory, filename)
    # shut_down(webcam)
    return webcam


if __name__ == '__main__':
    directory = ['fake']
    for d in directory:
        file_info = os.walk('../' + d)
        for p, _, file_list in file_info:
           for i, file in enumerate(file_list):
               webcam = main(p, file)
               print('* '*10 + str(i) + ' *'*10)

    shut_down(webcam)
