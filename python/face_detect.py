"""Face and eye detection with OpenCV.

Two detectors are provided:
  * haar : Viola-Jones Haar cascades shipped with OpenCV (no download needed)
  * dnn  : ResNet-10 SSD face detector (weights are downloaded on first use)

Usage:
  python face_detect.py --image ../examples/astronaut.jpg
  python face_detect.py --image group.jpg --method dnn --out result.jpg
  python face_detect.py --webcam
"""
import argparse
import os
import urllib.request

import cv2

DNN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DNN_FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}


def load_cascade(name):
    # read through Python so non-ASCII install paths (e.g. C:\Users\gönül) also work on Windows
    with open(os.path.join(cv2.data.haarcascades, name), encoding="utf-8") as f:
        fs = cv2.FileStorage(f.read(), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    clf = cv2.CascadeClassifier()
    if not clf.read(fs.getFirstTopLevelNode()):
        raise RuntimeError(f"cannot load cascade {name}")
    return clf


class HaarDetector:
    def __init__(self):
        self.face = load_cascade("haarcascade_frontalface_default.xml")
        self.eye = load_cascade("haarcascade_eye.xml")

    def detect(self, img):
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        faces = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        out = []
        for (x, y, w, h) in faces:
            roi = gray[y:y + h // 2 + h // 8, x:x + w]
            eyes = self.eye.detectMultiScale(roi, 1.1, 8, minSize=(w // 10, w // 10))
            out.append(((x, y, w, h), 1.0, [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]))
        return out


class DnnDetector:
    def __init__(self, conf=0.5):
        os.makedirs(DNN_DIR, exist_ok=True)
        for name, url in DNN_FILES.items():
            path = os.path.join(DNN_DIR, name)
            if not os.path.exists(path):
                print(f"downloading {name} ...")
                urllib.request.urlretrieve(url, path)
        self.net = cv2.dnn.readNetFromCaffe(os.path.join(DNN_DIR, "deploy.prototxt"),
                                            os.path.join(DNN_DIR, "res10_300x300_ssd_iter_140000.caffemodel"))
        self.conf = conf

    def detect(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        det = self.net.forward()[0, 0]
        out = []
        for d in det:
            if d[2] < self.conf:
                continue
            x1, y1, x2, y2 = (d[3:7] * [w, h, w, h]).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            out.append(((x1, y1, x2 - x1, y2 - y1), float(d[2]), []))
        return out


def draw(img, detections):
    for (x, y, w, h), score, eyes in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(img, f"face {score:.2f}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 120, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="input image")
    ap.add_argument("--webcam", action="store_true", help="use the default camera")
    ap.add_argument("--method", choices=["haar", "dnn"], default="haar")
    ap.add_argument("--out", default="result.jpg", help="output image (image mode)")
    args = ap.parse_args()

    detector = HaarDetector() if args.method == "haar" else DnnDetector()

    if args.webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("face detection (q: quit)", draw(frame, detector.detect(frame)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    if not args.image:
        ap.error("give --image or --webcam")
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    dets = detector.detect(img)
    cv2.imwrite(args.out, draw(img, dets))
    print(f"{len(dets)} face(s) found -> {args.out}")
    for (x, y, w, h), s, eyes in dets:
        print(f"  box=({x},{y},{w},{h}) score={s:.2f} eyes={len(eyes)}")


if __name__ == "__main__":
    main()
