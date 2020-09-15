import cv2
import dlib
import os

# 单视频图片数量限制
PIC_LENGTH_ONE_VIDEO_MAX_LIMIT = 521
detector = dlib.get_frontal_face_detector()

try:
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
except RuntimeError as e:
    print('ERROR:  \'shape_predictor_68_face_landmarks.dat\' was not found in current directory.   ')


def extract_image_includes_face_from_video(path, save_dir=''):
    # 视频名称作为待保存的子文件夹名称
    cur_directory = path.split('.')[-2].split('/')[-1]
    webcam = cv2.VideoCapture(path)
    # 图片序号
    i = 0
    # 已保存的图片数量
    save_index = 0
    while True:
        ret, img_frame = webcam.read()
        # 视频结束
        if not ret:
            break
        # 检测是否有人脸
        try:
            faces = detector(img_frame, 0)
        except Exception as e:
            continue
        if len(faces) >= 1:
            # 图片序号名称 (4位数序号,不足4位 用0补足)
            temp_name = str(i)
            if len(temp_name) < 4:
                temp_name = (4 - len(temp_name)) * '0' + str(i)

            # 保存图片
            file_name = temp_name + '.jpg'
            directory_save = save_dir + '/' + cur_directory
            if not os.path.exists(directory_save):
                os.makedirs(directory_save)
            file_path = os.path.join(directory_save, file_name)
            cv2.imwrite(file_path, img_frame)
            save_index += 1
            # 如果设置了单个视频最大保存限制
            if PIC_LENGTH_ONE_VIDEO_MAX_LIMIT and save_index >= PIC_LENGTH_ONE_VIDEO_MAX_LIMIT:
                # 下一个视频
                break
        i += 1


if __name__ == '__main__':
    video_directory = r'Deepfakes/c40/videos'
    save_directory = r'Deepfakes/c40/images'
    file_info = os.walk(video_directory)
    for p, d, file_list in file_info:
        for i, file in enumerate(file_list):
            print('****  ' + str(i) + '  *****')
            _d = (video_directory + '/' + d[0]) if d else video_directory
            _path = os.path.join(_d, file)
            extract_image_includes_face_from_video(_path, save_directory)
    print('end........')
