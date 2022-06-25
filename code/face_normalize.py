import math

import os
import cv2
import dlib
import numpy as np
from assure_path import assure_path_exists

predictor = dlib.shape_predictor("../data/dlib-data/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
font = cv2.FONT_HERSHEY_SIMPLEX


class FaceNormalize:
    def __init__(self):
        self.NORMALIZE_SIZE = 64
        self.pt_center = (0, 0)
        self.pt_left_eye = (0, 0)
        self.pt_right_eye = (0, 0)
        self.rotate_angle = 0
        assure_path_exists("../data/normalized-imgs/")
        self.write_path = "../data/normalized-imgs/"
        self.read_path = "../data/face-imgs/"

    def face_rotate(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray, 1)
        if len(faces) != 0:
            for i, face in enumerate(faces):
                features = np.matrix([[p.x, p.y] for p in predictor(img, face).parts()])
                for idx, point in enumerate(features):
                    # 68点的坐标
                    pos = (point[0, 0], point[0, 1])
                    # print(idx, pos)
                    if idx == 36:
                        self.pt_left_eye = pos
                    elif idx == 45:
                        self.pt_right_eye = pos
                    self.pt_center = ((self.pt_left_eye[0] + self.pt_right_eye[0]) / 2,
                                      (self.pt_left_eye[1] + self.pt_right_eye[1]) / 2)
                    # cv2.circle(img, pos, 4, (0, 255, 0), 1)
                    # cv2.putText(img, str(idx), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # print(self.feature_dict)
            width, height = img.shape[:2]
            eye_width = abs(self.pt_right_eye[0] - self.pt_left_eye[0])
            eye_height = self.pt_right_eye[1] - self.pt_left_eye[1]
            self.rotate_angle = int(math.atan2(eye_height / 2, eye_width / 2) * 180 / math.pi)
            new_height = int(width * math.fabs(math.sin(math.radians(self.rotate_angle))) +
                             height * math.fabs(math.cos(math.radians(self.rotate_angle))))
            new_width = int(height * math.fabs(math.sin(math.radians(self.rotate_angle))) +
                            width * math.fabs(math.cos(math.radians(self.rotate_angle))))
            mat_rotate = cv2.getRotationMatrix2D((self.pt_center[1], self.pt_center[0]), self.rotate_angle, 1)
            mat_rotate[0, 2] += (new_width - width) / 2  # 重点在这步，目前不懂为什么加这步
            mat_rotate[1, 2] += (new_height - height) / 2  # 重点在这步
            dst = cv2.warpAffine(img, mat_rotate, (new_width, new_height), borderValue=(0, 0, 0))
            # print(self.rotate_angle)
            # print(self.pt_left_eye, self.pt_right_eye, self.pt_center)
            return dst
        else:
            return None

    def face_normalize(self, img):
        dst = img
        size = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) != 0:
            for i, face in enumerate(faces):
                if face.left() > 0:
                    left = face.left()
                else:
                    left = 0
                if face.top() > 0:
                    top = face.top()
                else:
                    top = 0
                if face.right() < size:
                    right = face.right()
                else:
                    right = size
                if face.bottom() < size:
                    bottom = face.bottom()
                else:
                    bottom = size
                # right = face.right()
                # top = face.top()
                # bottom = face.bottom()
                after_cut_face = img[top: bottom, left: right]

                after_resize = cv2.resize(after_cut_face, (self.NORMALIZE_SIZE, self.NORMALIZE_SIZE))
                dst = cv2.cvtColor(after_resize, cv2.COLOR_RGB2GRAY)
                # dst = cv2.equalizeHist(gray)
        return dst

    def write_normalized_face(self, img, face_num, person_name):
        cv2.imwrite(self.write_path + person_name + '/' + str(face_num) + '.jpg', img)

    def get_faces_by_name(self, person_name):
        all_faces_path = []
        path = self.read_path + person_name
        for fpath, dirname, fnames in os.walk(path):
            for f in fnames:
                filename = fpath + '/' + f
                all_faces_path.append(filename)
        return all_faces_path

    def run(self, person_name):
        all_faces_path = self.get_faces_by_name(person_name)
        os.mkdir(self.write_path + person_name)
        for face_path in all_faces_path:
            img = cv2.imread(face_path)
            after_rotate = self.face_rotate(img)

            if after_rotate is not None:
                after_normalize = self.face_normalize(after_rotate)
                self.write_normalized_face(after_normalize, all_faces_path.index(face_path) + 1, person_name)
                # print(face_path)


if __name__ == '__main__':
    face_normalize = FaceNormalize()
    face_normalize.run('00007')
    # face = FaceNormalize()
    # path = "data/screenshots/person_1/face_16_2021-06-07-14-41-14.jpg"
    # img = cv2.imread(path)
    # after_rotate = face.face_rotate(img)
    # gray = cv2.cvtColor(after_rotate, cv2.COLOR_BGR2GRAY)
    # faces = detector(gray, 1)
    # if len(faces) != 0:
    #     for i, face in enumerate(faces):
    #         left = face.left()
    #         right = face.right()
    #         top = face.top()
    #         bottom = face.bottom()
    #         print(left, right, top, bottom)
    #         after_cut_face = img[top: bottom, left: right]
    #
    #         cv2.namedWindow("image", 0)
    #         cv2.resizeWindow("image", 360, 360)
    #         cv2.imshow("image", after_cut_face)
    #         cv2.waitKey(0)
