"""Yüz Tanıma Sistemi - face enrollment, live multi-face recognition and attendance log.

Detector  : YuNet (OpenCV DNN, ONNX)
Recognizer: SFace 128-d embeddings, cosine similarity (accept >= 0.363, OpenCV's
            recommended threshold for SFace)
Storage   : database/people.json, database/faces/<ID>/*.jpg, database/embeddings.npz,
            database/sightings.csv  (all local; database/ is git-ignored)

Run:
  python face_app.py
"""
import csv
import json
import os
import threading
import time
import tkinter as tk
import urllib.request
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DB_DIR = os.path.join(HERE, "database")
FACES_DIR = os.path.join(DB_DIR, "faces")
PEOPLE_JSON = os.path.join(DB_DIR, "people.json")
EMB_FILE = os.path.join(DB_DIR, "embeddings.npz")
LOG_CSV = os.path.join(DB_DIR, "sightings.csv")

MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
THRESHOLD = 0.363          # cosine similarity
SAMPLES_PER_PERSON = 15    # captured during camera enrollment
LOG_COOLDOWN_S = 60        # one sighting per person per minute
MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


# ----------------------------------------------------------------------------- utilities
def imread_u(path):
    """cv2.imread that also works with non-ASCII paths on Windows."""
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_u(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if ok:
        buf.tofile(path)
    return ok


def tr_datetime(ts):
    d = datetime.fromisoformat(ts)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year} {DAYS[d.weekday()]}, saat {d:%H:%M:%S}"


def sighting_sentence(row):
    return f"{row['name']} ({row['id']}), {tr_datetime(row['time'])}'de kamerada görüldü."


def load_font(size):
    for f in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ----------------------------------------------------------------------------- engine
class FaceEngine:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, url in MODELS.items():
            path = os.path.join(MODEL_DIR, name)
            if not os.path.exists(path):
                print(f"downloading {name} ...")
                urllib.request.urlretrieve(url, path)
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(
            os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx"), "")
        self.lock = threading.Lock()

    def detect(self, img):
        h, w = img.shape[:2]
        with self.lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
        return [] if faces is None else list(faces)

    def embed(self, img, face):
        with self.lock:
            crop = self.recognizer.alignCrop(img, face)
            f = self.recognizer.feature(crop).ravel()
        return f / (np.linalg.norm(f) + 1e-12), crop


class Database:
    def __init__(self):
        os.makedirs(FACES_DIR, exist_ok=True)
        self.people = json.load(open(PEOPLE_JSON, encoding="utf-8")) if os.path.exists(PEOPLE_JSON) else {}
        if os.path.exists(EMB_FILE):
            d = np.load(EMB_FILE)
            self.ids, self.vecs = list(d["ids"]), d["vecs"]
        else:
            self.ids, self.vecs = [], np.zeros((0, 128), np.float32)

    def save(self):
        with open(PEOPLE_JSON, "w", encoding="utf-8") as f:
            json.dump(self.people, f, ensure_ascii=False, indent=2)
        np.savez(EMB_FILE, ids=np.array(self.ids, dtype=str), vecs=self.vecs)

    def next_id(self):
        n = max([int(k[1:]) for k in self.people] + [0]) + 1
        return f"P{n:03d}"

    def add_person(self, name, crops, vecs):
        pid = self.next_id()
        folder = os.path.join(FACES_DIR, pid)
        os.makedirs(folder, exist_ok=True)
        for i, c in enumerate(crops):
            imwrite_u(os.path.join(folder, f"{i + 1:02d}.jpg"), c)
        self.people[pid] = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
                            "samples": len(crops)}
        self.ids += [pid] * len(vecs)
        self.vecs = np.vstack([self.vecs, np.array(vecs, np.float32)])
        self.save()
        return pid

    def remove_person(self, pid):
        """Removes the person from the active database; face images are moved to faces/_removed."""
        self.people.pop(pid, None)
        keep = [i for i, p in enumerate(self.ids) if p != pid]
        self.ids = [self.ids[i] for i in keep]
        self.vecs = self.vecs[keep]
        src = os.path.join(FACES_DIR, pid)
        if os.path.isdir(src):
            dst_root = os.path.join(FACES_DIR, "_removed")
            os.makedirs(dst_root, exist_ok=True)
            os.replace(src, os.path.join(dst_root, f"{pid}_{datetime.now():%Y%m%d_%H%M%S}"))
        self.save()

    def rename(self, pid, name):
        self.people[pid]["name"] = name
        self.save()

    def photo(self, pid):
        path = os.path.join(FACES_DIR, pid, "01.jpg")
        return imread_u(path) if os.path.exists(path) else None

    def identify(self, vec):
        if not self.ids:
            return None, 0.0
        sims = self.vecs @ vec
        best = {}
        for pid, s in zip(self.ids, sims):
            best[pid] = max(best.get(pid, -1.0), float(s))
        pid = max(best, key=best.get)
        return (pid, best[pid]) if best[pid] >= THRESHOLD else (None, best[pid])


class SightingLog:
    FIELDS = ["time", "id", "name", "similarity"]

    def __init__(self):
        self.rows = []
        if os.path.exists(LOG_CSV):
            with open(LOG_CSV, encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
        self.last_seen = {}

    def save(self):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    def add(self, pid, name, sim):
        now = time.time()
        if now - self.last_seen.get(pid, 0) < LOG_COOLDOWN_S:
            return None
        self.last_seen[pid] = now
        row = {"time": datetime.now().isoformat(timespec="seconds"), "id": pid, "name": name,
               "similarity": f"{sim:.3f}"}
        self.rows.append(row)
        self.save()
        return row

    def delete(self, indices):
        for i in sorted(indices, reverse=True):
            del self.rows[i]
        self.save()


# ----------------------------------------------------------------------------- drawing
FONT = load_font(20)
FONT_BIG = load_font(26)


def annotate(frame_bgr, results):
    """results: list of (box, label, known)"""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for (x, y, w, h), label, known in results:
        color = (40, 200, 90) if known else (230, 70, 60)
        d.rectangle([x, y, x + w, y + h], outline=color, width=3)
        tw = d.textlength(label, font=FONT)
        d.rectangle([x, y - 28, x + tw + 12, y], fill=color)
        d.text((x + 6, y - 27), label, font=FONT, fill=(255, 255, 255))
    return img


# ----------------------------------------------------------------------------- camera window
class CameraWindow(tk.Toplevel):
    def __init__(self, app, title, on_frame, on_close=None):
        super().__init__(app.root)
        self.app, self.on_frame, self.on_close_cb = app, on_frame, on_close
        self.title(title)
        self.configure(bg="#111")
        self.view = tk.Label(self, bg="#111")
        self.view.pack(padx=8, pady=8)
        self.status = tk.Label(self, text="", fg="white", bg="#111", font=("Segoe UI", 12))
        self.status.pack(pady=(0, 6))
        ttk.Button(self, text="Kapat", command=self.close).pack(pady=(0, 10))
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Kamera", "Kamera açılamadı.", parent=self)
            self.destroy()
            return
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        self.after(10, self.loop)

    def loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            pil = self.on_frame(frame, self)
            if pil is not None:
                self._photo = ImageTk.PhotoImage(pil)
                self.view.configure(image=self._photo)
        self.after(30, self.loop)

    def close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.on_close_cb:
            self.on_close_cb()
        self.destroy()


# ----------------------------------------------------------------------------- main app
class App:
    def __init__(self, root):
        self.root = root
        root.title("Yüz Tanıma Sistemi")
        root.geometry("560x560")
        root.configure(bg="#f4f6f9")
        self.engine = FaceEngine()
        self.db = Database()
        self.log = SightingLog()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Big.TButton", font=("Segoe UI", 13), padding=12)
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))

        tk.Label(root, text="Yüz Tanıma Sistemi", font=("Segoe UI", 22, "bold"), bg="#f4f6f9",
                 fg="#1f2d3d").pack(pady=(22, 2))
        self.summary = tk.Label(root, font=("Segoe UI", 11), bg="#f4f6f9", fg="#5a6b7b")
        self.summary.pack(pady=(0, 16))

        buttons = [
            ("➕  Kişi Ekle (Kamera)", self.enroll_camera),
            ("🖼  Kişi Ekle (Fotoğraftan)", self.enroll_photo),
            ("🎥  Canlı Yüz Tanıma", self.live_recognition),
            ("🔍  Fotoğrafta Yüz Tanıma", self.recognize_photo),
            ("👥  Kayıtlı Kişiler", self.people_window),
            ("🕒  Görülme Kayıtları", self.log_window),
        ]
        for text, cmd in buttons:
            ttk.Button(root, text=text, style="Big.TButton", command=cmd).pack(fill="x", padx=60, pady=5)
        self.refresh_summary()

    def refresh_summary(self):
        self.summary.config(text=f"{len(self.db.people)} kayıtlı kişi  ·  {len(self.log.rows)} görülme kaydı")

    # ---- enrollment
    def ask_name(self):
        name = simpledialog.askstring("Kişi Ekle", "Kişinin adı soyadı:", parent=self.root)
        return name.strip() if name and name.strip() else None

    def enroll_camera(self):
        name = self.ask_name()
        if not name:
            return
        crops, vecs, last = [], [], [0.0]

        def on_frame(frame, win):
            faces = self.engine.detect(frame)
            results = []
            if len(faces) == 1 and time.time() - last[0] > 0.25 and len(vecs) < SAMPLES_PER_PERSON:
                v, c = self.engine.embed(frame, faces[0])
                vecs.append(v)
                crops.append(c)
                last[0] = time.time()
            for f in faces:
                results.append((tuple(int(t) for t in f[:4]), name, True))
            n = len(vecs)
            if len(faces) == 0:
                win.status.config(text="Yüz görünmüyor - kameraya bakın")
            elif len(faces) > 1:
                win.status.config(text="Kayıt sırasında kadrajda yalnızca bir kişi olmalı")
            else:
                win.status.config(text=f"Kaydediliyor: {n}/{SAMPLES_PER_PERSON}  (başınızı hafifçe sağa-sola çevirin)")
            if n >= SAMPLES_PER_PERSON:
                win.after(10, win.close)
            return annotate(frame, results)

        def finish():
            if len(vecs) >= 5:
                pid = self.db.add_person(name, crops, vecs)
                messagebox.showinfo("Kişi Ekle", f"{name} {pid} numarasıyla kaydedildi ({len(vecs)} örnek).")
            else:
                messagebox.showwarning("Kişi Ekle", "Yeterli yüz örneği alınamadı, kayıt yapılmadı.")
            self.refresh_summary()

        CameraWindow(self, f"Kişi Ekle - {name}", on_frame, finish)

    def enroll_photo(self):
        paths = filedialog.askopenfilenames(title="Kişinin fotoğrafları (tek yüz)",
                                            filetypes=[("Resimler", "*.jpg *.jpeg *.png *.bmp")])
        if not paths:
            return
        name = self.ask_name()
        if not name:
            return
        crops, vecs = [], []
        for p in paths:
            img = imread_u(p)
            if img is None:
                continue
            faces = self.engine.detect(img)
            if faces:
                face = max(faces, key=lambda f: f[2] * f[3])
                v, c = self.engine.embed(img, face)
                vecs.append(v)
                crops.append(c)
        if not vecs:
            messagebox.showwarning("Kişi Ekle", "Fotoğraflarda yüz bulunamadı.")
            return
        pid = self.db.add_person(name, crops, vecs)
        messagebox.showinfo("Kişi Ekle", f"{name} {pid} numarasıyla kaydedildi ({len(vecs)} fotoğraf).")
        self.refresh_summary()

    # ---- recognition
    def label_faces(self, img, log=False):
        results = []
        for f in self.engine.detect(img):
            v, _ = self.engine.embed(img, f)
            pid, sim = self.db.identify(v)
            box = tuple(int(t) for t in f[:4])
            if pid:
                name = self.db.people[pid]["name"]
                results.append((box, f"{name} ({pid})  {sim:.2f}", True))
                if log:
                    row = self.log.add(pid, name, sim)
                    if row:
                        self.root.after(0, self.refresh_summary)
                        self.last_event = sighting_sentence(row)
            else:
                results.append((box, "Bilinmeyen", False))
        return results

    def live_recognition(self):
        if not self.db.people:
            messagebox.showinfo("Canlı Yüz Tanıma", "Önce en az bir kişi ekleyin.")
            return
        self.last_event = "Henüz kayıt yok"

        def on_frame(frame, win):
            results = self.label_faces(frame, log=True)
            known = sum(r[2] for r in results)
            win.status.config(text=f"{len(results)} yüz, {known} tanındı   ·   Son kayıt: {self.last_event}")
            return annotate(frame, results)

        CameraWindow(self, "Canlı Yüz Tanıma", on_frame)

    def recognize_photo(self):
        path = filedialog.askopenfilename(filetypes=[("Resimler", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = imread_u(path)
        if img is None:
            return
        pil = annotate(img, self.label_faces(img))
        pil.thumbnail((1000, 750))
        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win._photo = ImageTk.PhotoImage(pil)
        tk.Label(win, image=win._photo).pack()

    # ---- people
    def people_window(self):
        win = tk.Toplevel(self.root)
        win.title("Kayıtlı Kişiler")
        win.geometry("760x420")
        tree = ttk.Treeview(win, columns=("id", "name", "samples", "created"), show="headings")
        for col, txt, w in (("id", "Numara", 80), ("name", "Ad Soyad", 220), ("samples", "Örnek", 70),
                            ("created", "Kayıt tarihi", 200)):
            tree.heading(col, text=txt)
            tree.column(col, width=w, anchor="w")
        tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        photo = tk.Label(win, text="Fotoğraf", width=24, relief="groove")
        photo.grid(row=0, column=1, sticky="n", padx=8, pady=8)
        bar = tk.Frame(win)
        bar.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        def fill():
            tree.delete(*tree.get_children())
            for pid, p in sorted(self.db.people.items()):
                tree.insert("", "end", iid=pid, values=(pid, p["name"], p["samples"], tr_datetime(p["created"])))
            self.refresh_summary()

        def on_select(_):
            sel = tree.selection()
            img = self.db.photo(sel[0]) if sel else None
            if img is not None:
                pil = Image.fromarray(cv2.cvtColor(cv2.resize(img, (168, 168)), cv2.COLOR_BGR2RGB))
                photo._img = ImageTk.PhotoImage(pil)
                photo.config(image=photo._img, width=168)

        def remove():
            for pid in tree.selection():
                name = self.db.people[pid]["name"]
                if messagebox.askyesno("Kişiyi Sil", f"{name} ({pid}) veritabanından çıkarılsın mı?", parent=win):
                    self.db.remove_person(pid)
            fill()

        def rename():
            sel = tree.selection()
            if not sel:
                return
            new = simpledialog.askstring("Yeniden Adlandır", "Yeni ad:", parent=win,
                                         initialvalue=self.db.people[sel[0]]["name"])
            if new and new.strip():
                self.db.rename(sel[0], new.strip())
                fill()

        tree.bind("<<TreeviewSelect>>", on_select)
        ttk.Button(bar, text="Yeniden Adlandır", command=rename).pack(side="left", padx=4)
        ttk.Button(bar, text="Seçili Kişiyi Sil", command=remove).pack(side="left", padx=4)
        fill()

    # ---- sightings
    def log_window(self):
        win = tk.Toplevel(self.root)
        win.title("Görülme Kayıtları")
        win.geometry("820x460")
        tree = ttk.Treeview(win, columns=("sentence", "sim"), show="headings", selectmode="extended")
        tree.heading("sentence", text="Kayıt")
        tree.heading("sim", text="Benzerlik")
        tree.column("sentence", width=680)
        tree.column("sim", width=90, anchor="center")
        tree.pack(fill="both", expand=True, padx=8, pady=8)
        bar = tk.Frame(win)
        bar.pack(pady=(0, 10))

        def fill():
            tree.delete(*tree.get_children())
            for i, r in reversed(list(enumerate(self.log.rows))):
                tree.insert("", "end", iid=str(i), values=(sighting_sentence(r), r["similarity"]))
            self.refresh_summary()

        def delete_selected():
            sel = [int(i) for i in tree.selection()]
            if sel and messagebox.askyesno("Kayıt Sil", f"{len(sel)} kayıt silinsin mi?", parent=win):
                self.log.delete(sel)
                fill()

        def delete_all():
            if self.log.rows and messagebox.askyesno("Tümünü Sil", "Tüm görülme kayıtları silinsin mi?", parent=win):
                self.log.delete(range(len(self.log.rows)))
                fill()

        def export():
            path = filedialog.asksaveasfilename(defaultextension=".csv", parent=win,
                                                initialfile=f"gorulme_kayitlari_{datetime.now():%Y%m%d}.csv")
            if path:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f, delimiter=";")
                    w.writerow(["Tarih-Saat", "Numara", "Ad Soyad", "Benzerlik", "Açıklama"])
                    for r in self.log.rows:
                        w.writerow([r["time"], r["id"], r["name"], r["similarity"], sighting_sentence(r)])

        ttk.Button(bar, text="Seçili Kayıtları Sil", command=delete_selected).pack(side="left", padx=4)
        ttk.Button(bar, text="Tümünü Sil", command=delete_all).pack(side="left", padx=4)
        ttk.Button(bar, text="Excel/CSV'ye Aktar", command=export).pack(side="left", padx=4)
        fill()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
