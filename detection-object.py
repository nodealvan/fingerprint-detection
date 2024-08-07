import cv2
import numpy as np
from tkinter import Tk, Label, Canvas
from PIL import Image, ImageTk
import mediapipe as mp
import threading

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inisialisasi OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

root = Tk()
root.title("Fingerprint and Face Scanner")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

video_path = "finger.mp4"  # Ganti dengan path video Anda
video_cap = cv2.VideoCapture(video_path)

# Komponen Tkinter
video_label = Label(root)
video_label.grid(row=0, column=0, padx=10, pady=10)

gesture_canvas = Canvas(root, width=640, height=480, bg='white')
gesture_canvas.grid(row=0, column=1, padx=10, pady=10)

message_label = Label(root, text="Mohon Letakkan tangan anda", font=("Arial", 16), fg="#333")
message_label.grid(row=1, column=0, columnspan=2, pady=10)

# Status pemutaran video
video_playing = False
stop_thread = False
video_thread = None
video_imgtk = None

def play_video():
    global video_playing, stop_thread, video_imgtk

    while video_playing and not stop_thread:
        ret, frame = video_cap.read()
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Mulai ulang video
            continue

        frame = cv2.resize(frame, (640, 480))  # Ubah ukuran frame video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_img = Image.fromarray(frame)
        video_imgtk = ImageTk.PhotoImage(image=video_img)

        gesture_canvas.imgtk = video_imgtk
        gesture_canvas.create_image(0, 0, anchor='nw', image=video_imgtk)
        root.update_idletasks()
        root.after(30)

def start_video():
    global video_playing, stop_thread, video_thread

    if video_playing:
        return

    video_playing = True
    stop_thread = False
    if video_thread is None or not video_thread.is_alive():
        video_thread = threading.Thread(target=play_video, daemon=True)
        video_thread.start()

def stop_video():
    global video_playing, stop_thread
    video_playing = False
    stop_thread = True
    gesture_canvas.delete("all")  # Bersihkan canvas

def reset_video():
    global video_cap
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Mulai ulang video dari frame awal

def update_video_feed():
    global video_playing, video_imgtk

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = hands.process(rgb_frame)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if result.multi_hand_landmarks:
        message_label.config(text="Scanner")
        if not video_playing:
            start_video()  # Pastikan video dimulai saat tangan terdeteksi

        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Warna hijau untuk landmark
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Warna hijau untuk koneksi
            )

        # Tampilkan canvas video
        if not gesture_canvas.winfo_ismapped():
            gesture_canvas.grid(row=0, column=1, padx=10, pady=10)
    else:
        message_label.config(text="Mohon Letakkan Sidik Jari")
        if video_playing:
            stop_video()
            reset_video()  # Reset video ketika tangan tidak terdeteksi

        # Sembunyikan canvas video jika tangan tidak terdeteksi
        if gesture_canvas.winfo_ismapped():
            gesture_canvas.grid_forget()

    # Gambar kotak di sekitar wajah yang terdeteksi dan tampilkan teks di atasnya
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(rgb_frame, "Wajah", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    video_img = Image.fromarray(rgb_frame)
    video_imgtk = ImageTk.PhotoImage(image=video_img)
    video_label.imgtk = video_imgtk
    video_label.configure(image=video_imgtk)

    root.after(10, update_video_feed)

# Memulai feed video dan menangani penghentian aplikasi dengan benar
update_video_feed()
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), video_cap.release(), cv2.destroyAllWindows(), root.destroy(), stop_video()))
root.mainloop()
