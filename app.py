# ==========================================
# 🎙️ VOICE DETECTOR: BUKA vs TUTUP
# ==========================================
import io
import numpy as np
import streamlit as st
import librosa
import joblib
import soundfile as sf
from scipy.stats import skew, kurtosis
from streamlit_mic_recorder import mic_recorder
import tempfile
import os

# ===============================
# KONFIGURASI STREAMLIT
# ===============================
st.set_page_config(page_title="Deteksi Suara BUKA/TUTUP", page_icon="🎙️", layout="centered")

st.markdown("""
<h1 style='text-align:center; color:#1f77b4;'>🎙️ Deteksi Kata: <span style='color:#ff4b4b;'>BUKA</span> vs <span style='color:#4caf50;'>TUTUP</span></h1>
<p style='text-align:center; font-size:16px;'>
Klik tombol <b>Mulai Rekam</b> → ucapkan "buka" atau "tutup" (±1–2 detik) → <b>Stop</b> → lihat hasil deteksi.
</p>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL, SCALER, ENCODER
# ===============================
MODEL_PATH = "audio_model.joblib"
SCALER_PATH = "audio_scaler.joblib"
ENCODER_PATH = "audio_encoder.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

# ===============================
# FUNGSI EKSTRAKSI FITUR
# ===============================
def ekstrak_fitur(file_audio):
    try:
        y, sr = librosa.load(file_audio, sr=None)
        y, _ = librosa.effects.trim(y, top_db=20)

        if len(y) < sr * 0.2:
            st.warning("⚠️ Audio terlalu pendek untuk diproses.")
            return None

        # Fitur dasar audio
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        fitur = [
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms),
            np.mean(y), np.std(y), np.var(y),
            skew(y), kurtosis(y),
            np.min(y), np.max(y)
        ]
        return np.array(fitur).reshape(1, -1)

    except Exception as e:
        st.error(f"Gagal ekstraksi fitur: {e}")
        return None

# ===============================
# PENGATURAN TAMBAHAN
# ===============================
with st.expander("⚙️ Pengaturan Rekaman"):
    c1, c2 = st.columns(2)
    with c1:
        sr_target = st.number_input("Target Sample Rate", min_value=8000, max_value=48000, value=16000, step=1000)
    with c2:
        save_rec = st.checkbox("Simpan rekaman ke folder ./recordings", value=False)

# ===============================
# REKAM SUARA
# ===============================
st.subheader("1️⃣ Rekam Suara")
audio = mic_recorder(
    start_prompt="🎤 Mulai Rekam",
    stop_prompt="⏹️ Stop Rekaman",
    just_once=False,
    format="wav"
)

# ===============================
# PROSES AUDIO
# ===============================
if audio and "bytes" in audio and audio["bytes"] is not None:
    st.subheader("2️⃣ Putar & Analisis")
    st.audio(audio["bytes"], format="audio/wav")

    try:
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)

        dur = len(data) / sr
        st.info(f"📊 Sample rate: **{sr} Hz**, Durasi: **{dur:.2f} detik**")

        # Simpan jika diminta
        if save_rec:
            os.makedirs("recordings", exist_ok=True)
            idx = len(os.listdir("recordings")) + 1
            sf.write(f"recordings/rec_{idx:03d}.wav", data, sr)
            st.success(f"💾 Rekaman disimpan: recordings/rec_{idx:03d}.wav")

        # Ekstraksi fitur
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, data, sr)
            fitur = ekstrak_fitur(tmp.name)

        if fitur is not None:
            st.subheader("3️⃣ Prediksi Model")
            fitur_scaled = scaler.transform(fitur)
            pred = model.predict(fitur_scaled)
            label = encoder.inverse_transform(pred)[0].upper()

            if label == "BUKA":
                st.success("✅ HASIL DETEKSI: **BUKA**")
            elif label == "TUTUP":
                st.error("🛑 HASIL DETEKSI: **TUTUP**")
            else:
                st.info(f"🏷️ Prediksi Model: **{label}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(fitur_scaled)[0]
                classes = encoder.inverse_transform(np.arange(len(proba)))
                prob_df = pd.DataFrame({"Kelas": classes, "Probabilitas": proba})
                prob_df["Kelas"] = prob_df["Kelas"].str.upper()
                st.bar_chart(prob_df.set_index("Kelas"))

        else:
            st.warning("⚠️ Gagal memproses fitur dari audio.")

    except Exception as e:
        st.error(f"Gagal membaca audio: {e}")

else:
    st.info("Klik tombol 🎤 **Mulai Rekam**, ucapkan 'buka' atau 'tutup', lalu tekan ⏹️ **Stop** untuk mendeteksi suara.")
