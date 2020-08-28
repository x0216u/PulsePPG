import cv2, time
from PPGMap import utils
import numpy as np
import dlib
import os
from PPGMap.delaunay import perspective_transoform
from concurrent.futures import ThreadPoolExecutor, as_completed  # 线程池


part_size = 128
img_list = []
fs = 30
PICTURE_INDEX = 10
VIDEO_FRAME = 1

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')


def run_observer(directory, filename):
    path = os.path.join(directory, filename)
    webcam = cv2.VideoCapture(path)

    global detector, predictor

    mean_colors = []
    timestamps = []
    pic_index = 0
    while True:
        ret, img_frame = webcam.read()
        points = []
        try:
            faces = detector(img_frame, 0)
        except Exception as e:
            print(filename + ' error 跳过')
            return
        if len(faces) >= 1:
            face_points = predictor(img_frame, faces[0])
            # print(face_points.parts())
            for part in face_points.parts():
                points.append((int(part.x), int(part.y)))
        if points:
            m_pts = [points[36], points[45], points[10], points[6]]
            # face_box = [m_pts[0][0], m_pts[0][1], points[35][0] - points[39][0], points[31][1] - points[39][1]]
            face_roi = perspective_transoform(img_frame, m_pts)
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2YUV)
            map_values = get_chrom_map(face_roi, mean_colors, timestamps)
            if map_values is not None:
                map_values = cv2.cvtColor(map_values, cv2.COLOR_YUV2RGB)
                # directory_save = 'G:\\mmy\\mmy_study\\code\\deepfake\\YUVmap_0818\\c23_map\\fake\\'
                directory_save = '../fake_map/'
                if not os.path.exists(directory_save):
                    os.makedirs(directory_save)
                path_2 = directory_save + 'deepfakes_' + filename.split('.')[0] + '_' + str(pic_index) + '.jpg'
                cv2.imwrite(path_2, map_values)
                mean_colors = []
                timestamps = []
                pic_index += 1
                if pic_index > VIDEO_FRAME:
                    break
    webcam.release()
    return


def get_chrom_map(face_roi, mean_colors, timestamps):
    # 分成32个区域
    height = face_roi.shape[0]
    width = face_roi.shape[1]
    row = 6
    sub_area_height = int(height / row)
    sub_area_width = int(width / row)
    timestamps += [time.time()]

    cur_frame_colors = np.zeros((row ** 2, 3))
    for i in range(row):
        for j in range(row):
            if len(mean_colors) < (i * row + 1 + j):
                mean_colors.append([])
            sub_area = face_roi[(j * sub_area_height):((j + 1) * sub_area_height),
                       (i * sub_area_width):((i + 1) * sub_area_width), :]
            # 求平均值
            area = sub_area.shape[0] * sub_area.shape[1]
            y1 = np.sum(sub_area[:, :, 0]) / area
            y2 = np.sum(sub_area[:, :, 1]) / area
            y3 = np.sum(sub_area[:, :, 2]) / area
            cur_frame_colors[(i * row + j)] = np.array([y1, y2, y3])
    # 归一化当前帧
    cur_frame_colors = utils.normalize(cur_frame_colors)
    # 将当前帧添加到总数组里面
    for i in range(row ** 2):
        mean_colors[i] += [list(cur_frame_colors[i])]

    if len(timestamps) >= part_size:
        img_values = np.array(mean_colors)
        np.nan_to_num(img_values)
        img_values = img_values.astype(np.uint8)
        return img_values
    return None


def run_images(directory):
    global detector, predictor
    mean_colors = []
    timestamps = []
    pic_index = 0
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        img_frame = cv2.imread(path)
        points = []
        try:
            faces = detector(img_frame, 0)
        except Exception as e:
            print(filename + ' error 跳过')
            continue
        if len(faces) >= 1:
            face_points = predictor(img_frame, faces[0])
            for part in face_points.parts():
                points.append((int(part.x), int(part.y)))
        if points:
            m_pts = [points[36], points[45], points[10], points[6]]
            # face_box = [m_pts[0][0], m_pts[0][1], points[35][0] - points[39][0], points[31][1] - points[39][1]]
            face_roi = perspective_transoform(img_frame, m_pts)
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2YUV)
            map_values = get_chrom_map(face_roi, mean_colors, timestamps)
            if map_values is not None:
                map_values = cv2.cvtColor(map_values, cv2.COLOR_YUV2RGB)
                directory_save = 'G:\\mmy\\mmy_study\\code\\deepfake\\YUVmap_0818\\c23_map\\fake_images\\'
                if not os.path.exists(directory_save):
                    os.makedirs(directory_save)
                path_2 = directory_save + 'deepfakes_' + filename.split('.')[0] + '_' + str(pic_index) + '.jpg'
                cv2.imwrite(path_2, map_values)
                mean_colors = []
                timestamps = []
                pic_index += 1
                if pic_index > VIDEO_FRAME:
                    break


# Clean up
def shut_down():
    # webcam.release()
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    directory = r'H:\FaceForensics++\Deepfakes\c23\images'
    file_info = os.listdir(directory)

    import time
    start = time.time()

    skip_index = 1
    thread_pool = ThreadPoolExecutor(5)
    task_list = []
    for i, _d in enumerate(file_info):
        if skip_index and i < (skip_index - 1):
            continue
        path = os.path.join(directory, _d)
        obj = thread_pool.submit(run_images, path)
        task_list.append(obj)

    for i, future in enumerate(as_completed(task_list)):
        print('* ' * 10 + str(i) + ' *' * 10)
        if i == 5:
            print(time.time() - start)
