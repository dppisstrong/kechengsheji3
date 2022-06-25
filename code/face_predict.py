import numpy as np
import cv2
import dlib
import keras

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/dlib-data/shape_predictor_68_face_landmarks.dat")


class Predict:
    def __init__(self):
        self.model = None
        self.face_id_list = []
        self.my_name = ''
        self.get_name_flag = True
        self.delay_flag = True
        self.delay_frame = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.mouth_height = 0
        self.open_mouth_flag = False
        self.turn_right_flag = False
        self.turn_left_flag = False
        self.p_mouth_top = (0, 0)
        self.p_mouth_bottom = (0, 0)
        self.p_left_face = (0, 0)
        self.p_right_face = (0, 0)
        self.p_nose_center = (0, 0)
        self.turn_left_width = 0
        self.turn_right_width = 0
        self.rect_height = 0
        self.rect_width = 0

    def face_recognition(self, model_path, stream):
        # model = Model()
        self.load_model(model_path)
        frame = 0
        while stream.isOpened():
            ret, img_camera = stream.read()
            img_gray = cv2.cvtColor(img_camera, cv2.COLOR_BGR2GRAY)
            faces = detector(img_gray, 1)
            if len(faces) != 0:
                for i, face in enumerate(faces):
                    left = face.left()
                    right = face.right()
                    top = face.top()
                    bottom = face.bottom()
                    self.rect_height = bottom - top
                    self.rect_width = right - top

                    features = np.matrix([[p.x, p.y] for p in predictor(img_camera, face).parts()])
                    for idx, point in enumerate(features):
                        # 68点的坐标
                        pos = (point[0, 0], point[0, 1])
                        if idx == 51:
                            self.p_mouth_top = pos
                        if idx == 57:
                            self.p_mouth_bottom = pos
                        self.mouth_height = self.p_mouth_bottom[1] - self.p_mouth_top[1]
                        # print(mouth_height)

                        if idx == 1:
                            self.p_right_face = pos
                        if idx == 15:
                            self.p_left_face = pos
                        if idx == 30:
                            self.p_nose_center = pos
                        self.turn_left_width = self.p_left_face[0] - self.p_nose_center[0]
                        self.turn_right_width = self.p_nose_center[0] - self.p_right_face[0]

                    img_cut = img_camera[top: bottom, left: right]
                    if img_cut.shape[0] != 0 and img_cut.shape[1] != 0:
                        # img_rotate = normalize.face_rotate(img_cut)
                        # img_normalize = normalize.face_normalize(img_cut)
                        after_resize = cv2.resize(img_cut, (64, 64))
                        # gray = cv2.cvtColor(after_resize, cv2.COLOR_RGB2GRAY)
                        # print(img_normalize.shape)
                        cv2.rectangle(img_camera, (left, top), (right, bottom), (0, 255, 0), 2)
                        predict_probability, faceID = self.face_predict(after_resize)
                        # print(predict_probability, faceID)
                        if len(self.face_id_list) <= 20:
                            self.face_id_list.append(faceID)

            if frame > 10:
                print(self.face_id_list)
                max_label = max(self.face_id_list, key=self.face_id_list.count)
                img_set_id = "{}".format(max_label)
                if self.get_name_flag:
                    self.my_name = str(img_set_id)
                    self.get_name_flag = False
                cv2.putText(img_camera, self.my_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 1)
                # if self.open_mouth_flag is False and self.turn_right_flag is False and self.turn_left_flag is False:
                #     cv2.putText(img_camera, "Please open mouth", (10, 30), self.font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                #     radio_mouth = self.mouth_height / self.rect_height
                #     if radio_mouth > 0.2:
                #         self.open_mouth_flag = True
                #         # print(self.open_mouth_flag)
                # elif self.open_mouth_flag is True and self.turn_right_flag is False and self.turn_left_flag is False:
                #     cv2.putText(img_camera, "Please turn right", (10, 30), self.font, 1, (0, 255, 0), 1,
                #                 cv2.LINE_AA)
                #     radio_right_face = self.turn_right_width / self.rect_width
                #     if radio_right_face < 0.3:
                #         self.turn_right_flag = True
                #         print(self.turn_right_flag)
                # elif self.open_mouth_flag is True and self.turn_right_flag is True and self.turn_left_flag is False:
                #     cv2.putText(img_camera, "Please turn left", (10, 30), self.font, 1, (0, 255, 0), 1,
                #                 cv2.LINE_AA)
                #     radio_left_face = self.turn_left_width / self.rect_width
                #     if radio_left_face < 0.3:
                #         self.turn_left_flag = True
                #         print(self.turn_right_flag)
                # elif self.open_mouth_flag is True and self.turn_right_flag is True and self.turn_left_flag is True:
                #     if self.delay_flag:
                #         self.delay_frame = frame
                #         self.delay_flag = False
                #     cv2.putText(img_camera, "Check in successfully", (10, 30), self.font, 1, (0, 255, 0), 2,
                #                 cv2.LINE_AA)
                #     if frame - self.delay_frame > 20:
                #         #DAO.staff_check_in(self.my_name)
                #         break

            cv2.imshow("camera", img_camera)
            frame += 1
            k = cv2.waitKey(10)
            if k == 27:
                break

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def face_predict(self, image):
        image = np.array(image)
        image = image.reshape((1, 64, 64, 3))
        image = image.astype('float32')
        image /= 255
        predict_probability = self.model.predict_proba(image)
        result = self.model.predict_classes(image)

        # return max(predict_probability[0]), result[0]
        return predict_probability[0], result[0]

    def run(self):
        model_path = '../data/model/model.h5'
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.face_recognition(model_path, cap)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    predict = Predict()
    predict.run()
    # path = "data/normalized-faces/person_2/4.jpg"
    # image = cv2.imread(path)
    # image = np.array(image)
    # image = image.reshape((1, 64, 64, 3))
    #
    # image = image.astype('float32')
    # image /= 255
    # model = Model()
    # model.load_model('data/model/model.h5')
    # my_model = model.model
    # result = my_model.predict_proba(image)
    # print('result:', result)
    # result = my_model.predict_classes(image)
    # print(result)
