import cv2
import numpy as np
import pafy

# yaş ve cinsiyetin tahmin edileceği url
url = 'https://www.youtube.com/watch?v=BIXkZk7tSZQ'
vPafy = pafy.new(url)

play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)

#cap = cv2.VideoCapture(0) #WebCam üzerinden çalışmak için bu satırı aktif edin

cap.set(3, 720)  # videonun genişliği
cap.set(4, 1280)  # videonun yüksekliği

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

gender_list = ['KADIN', 'ERKEK']


def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net


def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    while True:
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            print("{} Yüz Bulundu".format(str(len(faces))))

            for (x, y, w, h) in faces:
                # Resimdeki yüzleri bul
                face_img = image[y:y + h, h:h + w].copy()

                # yüz nesesini yeniden boyutlandır
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Cinsiyetleri tahmin et
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print("Cinsiyet : " + gender)

                # Yaşı tahmin et
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print("Yaş Aralığı: " + age)

                overlay_text = "%s %s" % (gender, age)

                # Orjinal resme kare çizelim
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

                # Orjinal resme yazıyı ekleyelim
                cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Resmi gösterelim
        cv2.imshow('Resim', image)

        # Klavyeden q harfine basılınca programı bitirelim
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)
