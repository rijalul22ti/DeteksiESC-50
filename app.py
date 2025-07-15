from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
import os
import numpy as np
import librosa
import uuid
import cv2
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model Keras (.keras)
model = load_model("esc50v1.keras")
le = joblib.load("label_encoder_esc50.pkl")

SAMPLE_RATE = 22050        # jumlah sampel per detik
DURATION = 5               # durasi audio yang ingin dibaca (dalam detik)
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128                # jumlah mel bands untuk spectrogram
# Fungsi bantu untuk ekstrak fitur dari audio
def extract_features_single(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(y)
        y = np.pad(y, (0, padding), 'constant')

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalisasi ke 0–255
    mel_spec_norm = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX)
    mel_spec_norm = mel_spec_norm.astype(np.uint8)
    mel_spec_rgb = cv2.cvtColor(mel_spec_norm, cv2.COLOR_GRAY2RGB)
    mel_spec_resized = cv2.resize(mel_spec_rgb, (224, 224))

    return mel_spec_resized  # shape (224, 224, 3)

def predict_single_audio(file_path, model, label_encoder, input_shape=(224, 224)):
    # Ekstrak Mel-spectrogram
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), 'constant')
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalisasi dan ubah ke RGB
    mel_spec_norm = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mel_rgb = cv2.cvtColor(mel_spec_norm, cv2.COLOR_GRAY2RGB)
    mel_rgb_resized = cv2.resize(mel_rgb, input_shape)

    # Preprocessing seperti data training
    mel_rgb_resized = mel_rgb_resized / 255.0
    mel_rgb_resized = np.expand_dims(mel_rgb_resized, axis=0)  # Tambah batch dimension

    # Prediksi
    prediction = model.predict(mel_rgb_resized)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(prediction)

    return predicted_label, confidence

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")

# Route untuk serve halaman utama
@app.route("/")
def home():
    return render_template("beranda.html")

# Route untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "File audio tidak ditemukan"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        class_labels = [
            "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects",
            "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets",
            "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush",
            "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing",
            "coughing", "footsteps", "laughing", "brushing_teeth", "snoring",
            "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
            "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner",
            "clock_alarm", "clock_tick", "glass_breaking", "helicopter", "chainsaw",
            "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks",
            "hand_saw"
        ]
        test_audio_path = filepath  # ganti dengan path audio kamu
        predicted_label, confidence = extract_features_single(test_audio_path)

        predicted_class_name = predicted_label
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")

        return jsonify({'result': predicted_class_name})

    except Exception as e:
        import traceback
        print("Terjadi error saat prediksi:")
        traceback.print_exc()
        return jsonify({"error": "Terjadi kesalahan saat prediksi", "detail": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
