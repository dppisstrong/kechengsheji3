import os.path
import time
import dlib
import cv2
from assure_path import assure_path_exists
# 默认的人脸检测器
detector = dlib.get_frontal_face_detector()


class GetFaceImg:
    def __init__(self, img_set_id):
        assure_path_exists("../data/face-imgs/")
        self.screenshots_path = "../data/face-imgs/"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_count = 1
        self.frame = 0
        self.person_num = 0
        self.stream_width = 0
        self.stream_height = 0
        self.img_set_id = img_set_id

    def make_dir(self):
        # os.mkdir(self.screenshots_path + "person_" + str(self.person_num))
        # 有则删除，无则添加
        id_list = []
        for ID in os.listdir(self.screenshots_path):
            id_list.append(ID)
            if ID == self.img_set_id:
                for img in os.listdir(self.screenshots_path + ID):
                    os.remove(self.screenshots_path + self.img_set_id + '/' +img)
        if self.img_set_id not in id_list:
            os.mkdir(self.screenshots_path + self.img_set_id)

    # def check_person_num(self):
    #     if os.listdir(self.screenshots_path):
    #         person_list = os.listdir(self.screenshots_path)
    #         person_list_count = []
    #         for person in person_list:
    #             person_list_count.append(int(person.split('_')[-1]))
    #         self.person_num = max(person_list_count) + 1
    #     else:
    #         self.person_num = 0

    def get_face_from_camera(self, stream):
        self.stream_width = stream.get(3)
        self.stream_height = stream.get(4)
        # self.check_person_num()
        self.make_dir()
        while stream.isOpened():
            flag, img_origin = stream.read()
            ret, img_camera = stream.read()
            img_gray = cv2.cvtColor(img_camera, cv2.COLOR_RGB2GRAY)
            faces = detector(img_gray, 1)  # 返回人脸检测矩形框4点坐标
            if len(faces) == 1:
                for i, face in enumerate(faces):
                    height = face.bottom() - face.top()
                    width = face.right() - face.left()
                    left = int(face.left() - width / 6)
                    right = int(face.right() + width / 6)
                    top = int(face.top() - height / 6)
                    bottom = int(face.bottom() + height / 6)
                    # print("面部矩阵：%d %d %d %d" % (left, right, top, bottom))

                    if bottom < self.stream_height and right < self.stream_width and top > 0 and left > 0:
                        cv2.rectangle(img_camera, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(img_camera, "Face #{}".format(i + 1), (left - 10, top - 10), self.font, 0.5,
                                    (0, 255, 0), 1,
                                    cv2.LINE_AA)
                        # 收集人脸信息
                        if self.face_count <= 100:
                            img_face = img_origin[top: bottom, left: right]
                            current_person_path = self.screenshots_path + self.img_set_id
                            cv2.imwrite(
                                current_person_path + "/face" + "_" + str(self.face_count) + "_" + time.strftime(
                                    "%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", img_face)
                            print("获取图像:", self.face_count)
                            self.face_count += 1
                    else:
                        cv2.putText(img_camera, "out of screen", (10, 30), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(img_camera, "Not Found Face", (10, 30), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(img_camera, "press ESC to quit", (20, 450), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_camera)
            self.frame += 1
            k = cv2.waitKey(1)
            # 按下ESC退出
            if k == 27:
                break
            if self.face_count > 100:
                break

    def run(self):
        # 采用cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)通过摄像头捕捉图形一次使用花费时间在0.08秒左右
        # cap=cv2.VideoCapture(0）直接调用摄像头检测速度在0.045秒左右
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.get_face_from_camera(cap)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Face_Img = GetFaceImg('000003')
    Face_Img.run()
