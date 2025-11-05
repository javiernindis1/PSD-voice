# app_record.py
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tsfel
import librosa
import soundfile as sf
import joblib

from streamlit_mic_recorder import mic_recorder

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(page_title="Deteksi Kata BUKA/TUTUP", page_icon="🎙️", layout="centered")
st.title("🎙️ Deteksi Kata: **BUKA** vs **TUTUP**")
st.caption("Klik Rekam → Ucapkan ‘buka’ atau ‘tutup’ (±1–2 detik) → Stop → Prediksi.")

MODEL_PATH = "audio_model.joblib"
FTR_PATH = "feature_names.json"

# =========================
# Util & Loader
# =========================
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not FTR_PATH.exists():
        st.error(
            "Model/fitur belum ditemukan. Jalankan training dulu "
            "(pastikan ada artifacts/model.joblib & artifacts/feature_names.json)."
        )
        st.stop()
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FTR_PATH.read_text(encoding="utf-8"))
    cfg = tsfel.get_features_by_domain("statistical")
    return model, feature_names, cfg

def extract_tsfel_stat_features(y: np.ndarray, sr: int, cfg) -> pd.DataFrame:
    """Ekstrak TSFEL statistical → DataFrame 1 baris; tambah metadata sr & durasi."""
    feats = tsfel.time_series_features_extractor(cfg, y, fs=sr, verbose=0)
    feats.insert(0, "sr", sr)
    feats.insert(1, "duration_sec", len(y) / float(sr) if sr else None)
    return feats

def ensure_feature_alignment(feats_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Pilih numerik & reindex agar sama persis dengan urutan fitur training."""
    feats_df = feats_df.select_dtypes(include=["number"]).copy()
    aligned = feats_df.reindex(columns=feature_names)
    return aligned

def map_to_upper_label(pred, classes=None) -> str:
    """Pastikan output final jadi 'BUKA' atau 'TUTUP' bila memungkinkan."""
    p = str(pred).strip().lower()
    if p in {"buka", "open", "1"}:
        return "BUKA"
    if p in {"tutup", "close", "0"}:
        return "TUTUP"
    # fallback: tampilkan uppercase dari label model
    return str(pred).upper()

# =========================
# Load Model & Fitur
# =========================
model, feature_names, cfg = load_artifacts()

# =========================
# Pengaturan Rekaman
# =========================
with st.expander("Pengaturan Rekaman"):
    c1, c2 = st.columns(2)
    with c1:
        target_sr = st.number_input(
            "Target sample rate (pemrosesan)", min_value=8000, max_value=48000, value=16000, step=1000
        )
    with c2:
        force_mono = st.checkbox("Paksa mono", value=True)
    keep_file = st.checkbox("Simpan hasil rekaman ke ./recordings", value=False)

# =========================
# Rekam Suara (Browser)
# =========================
st.subheader("1) Rekam")
audio = mic_recorder(
    start_prompt="🎤 Mulai Rekam",
    stop_prompt="⏹️ Stop",
    just_once=False,
    format="wav"  # sample rate mengikuti browser/device; akan di-resample setelahnya
)

if audio and "bytes" in audio and audio["bytes"] is not None:
    wav_bytes = audio["bytes"]

    # Putar ulang
    st.subheader("2) Putar Ulang & Info")
    st.audio(wav_bytes, format="audio/wav")

    # Baca WAV dari memori
    try:
        data, sr_loaded = sf.read(io.BytesIO(wav_bytes), dtype="float32")  # (n,) atau (n, ch)
    except Exception as e:
        st.error(f"Gagal membaca audio dari memori: {e}")
        st.stop()

    # Paksa mono jika diminta
    if data.ndim == 2 and force_mono:
        data = data.mean(axis=1)

    # Pastikan ada sample rate; jika tidak, gunakan target_sr
    sr_loaded = int(sr_loaded) if sr_loaded else int(target_sr)

    # Resample ke target_sr jika berbeda
    if sr_loaded != int(target_sr):
        try:
            data = librosa.resample(y=data, orig_sr=sr_loaded, target_sr=int(target_sr), res_type="kaiser_fast")
            sr = int(target_sr)
        except Exception as e:
            st.error(f"Gagal resample audio: {e}")
            st.stop()
    else:
        sr = sr_loaded

    dur = len(data) / float(sr)
    st.info(f"Sample rate: **{sr} Hz** | Durasi: **{dur:.2f} s** | Channel: **{1 if data.ndim==1 else data.shape[1]}**")

    # (Opsional) Simpan rekaman ke file
    if keep_file:
        try:
            rec_dir = Path("./recordings"); rec_dir.mkdir(exist_ok=True)
            idx = len(list(rec_dir.glob("rec_*.wav"))) + 1
            out_path = rec_dir / f"rec_{idx:03d}.wav"
            sf.write(out_path.as_posix(), data, sr)
            st.success(f"Disimpan: {out_path.as_posix()}")
        except Exception as e:
            st.warning(f"Gagal menyimpan rekaman: {e}")

    # =========================
    # Ekstraksi Fitur
    # =========================
    st.subheader("3) Ekstraksi Fitur TSFEL (Statistical)")
    with st.spinner("Ekstraksi fitur..."):
        try:
            feats_df = extract_tsfel_stat_features(data, sr, cfg)
            aligned = ensure_feature_alignment(feats_df, feature_names)
            st.caption(f"Ekstrak {aligned.shape[1]} fitur (sudah disejajarkan dengan model).")
            with st.expander("Lihat 20 kolom fitur pertama"):
                st.dataframe(aligned.iloc[:, :min(20, aligned.shape[1])])
        except Exception as e:
            st.error(f"Gagal ekstraksi fitur: {e}")
            st.stop()

    # =========================
    # Prediksi
    # =========================
    st.subheader("4) Prediksi")
    with st.spinner("Memproses prediksi..."):
        try:
            y_pred = model.predict(aligned)[0]
            label_upper = map_to_upper_label(y_pred, getattr(model, "classes_", None))

            # Tampilkan headline hasil (BUKA/TUTUP)
            if label_upper == "BUKA":
                st.success("✅ HASIL: **BUKA**")
            elif label_upper == "TUTUP":
                st.error("🛑 HASIL: **TUTUP**")
            else:
                st.info(f"🏷️ Prediksi: **{label_upper}**")

            # Probabilitas (jika classifier mendukung)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(aligned)[0]
                classes = getattr(model, "classes_", None)
                if classes is not None and len(classes) == len(proba):
                    prob_df = pd.DataFrame({"class": classes, "prob": proba}).sort_values("prob", ascending=False)
                    prob_df["class"] = prob_df["class"].astype(str).str.upper()
                    st.bar_chart(prob_df.set_index("class"))

                    # Confidence untuk label final (jika ada)
                    try:
                        conf = float(prob_df.loc[prob_df["class"] == label_upper, "prob"].values[0])
                        st.caption(f"Confidence {label_upper}: **{conf:.2%}**")
                    except Exception:
                        pass
        except Exception as e:
            st.error(f"Gagal prediksi: {e}")

else:
    st.info("Klik **🎤 Mulai Rekam**, ucapkan 'buka' atau 'tutup', lalu **⏹️ Stop** untuk memulai pemrosesan.")
