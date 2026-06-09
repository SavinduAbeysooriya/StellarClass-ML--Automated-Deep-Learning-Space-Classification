import os
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="StellarClass-ML",
    page_icon="🌌",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_NAMES = [
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
    "spectral_type_enc",
    "galaxy_pop_enc",
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "u_r",
    "g_i",
    "u_z",
    "redshift_u",
    "redshift_g",
    "redshift_r",
    "u_g_ratio",
    "g_r_ratio",
]

CLASS_INFO = {
    "GALAXY": {
        "emoji": "🌀",
        "colour": "#4A90D9",
        "desc": (
            "A gravitationally bound system of stars, gas, dust, and dark matter. "
            "Galaxies span from dwarf systems to massive ellipticals."
        ),
    },
    "STAR": {
        "emoji": "⭐",
        "colour": "#F5A623",
        "desc": (
            "A luminous sphere of plasma held together by gravity, powered by nuclear fusion."
        ),
    },
    "QSO": {
        "emoji": "✨",
        "colour": "#9B59B6",
        "desc": (
            "A quasi-stellar object, powered by a supermassive black hole at the center of a galaxy."
        ),
    },
}


@st.cache_resource
def load_artifacts():
    artifacts = {}

    def load_pickle(filename):
        path = MODELS_DIR / filename
        if not path.exists():
            return None
        return joblib.load(path)

    artifacts["scaler"] = load_pickle("standard_scaler.pkl")
    artifacts["spectral_enc"] = load_pickle("spectral_encoder.pkl")
    artifacts["galaxy_enc"] = load_pickle("galaxy_encoder.pkl")
    artifacts["target_enc"] = load_pickle("target_encoder.pkl")
    artifacts["rf_model"] = load_pickle("random_forest_model.pkl")

    ann_model = None
    ann_path = MODELS_DIR / "ann_best_model.keras"
    if ann_path.exists():
        try:
            import keras

            ann_model = keras.models.load_model(ann_path)
        except Exception:
            try:
                import tensorflow as tf

                ann_model = tf.keras.models.load_model(ann_path)
            except Exception:
                ann_model = None
    artifacts["ann_model"] = ann_model

    return artifacts


artifacts = load_artifacts()
scaler = artifacts["scaler"]
spectral_enc = artifacts["spectral_enc"]
galaxy_enc = artifacts["galaxy_enc"]
target_enc = artifacts["target_enc"]
rf_model = artifacts["rf_model"]
ann_model = artifacts["ann_model"]


def encode_category(encoder, value):
    if encoder is None:
        raise RuntimeError("Required encoder artifact is missing.")
    return int(encoder.transform([value])[0])


def build_feature_row(alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population):
    spectral_encoded = encode_category(spectral_enc, spectral_type)
    galaxy_encoded = encode_category(galaxy_enc, galaxy_population)

    feature_row = np.array(
        [[
            alpha,
            delta,
            u,
            g,
            r,
            i,
            z,
            redshift,
            spectral_encoded,
            galaxy_encoded,
            u - g,
            g - r,
            r - i,
            i - z,
            u - r,
            g - i,
            u - z,
            redshift * u,
            redshift * g,
            redshift * r,
            u / (g + 1e-8),
            g / (r + 1e-8),
        ]],
        dtype=float,
    )

    if feature_row.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature mismatch: expected {len(FEATURE_NAMES)} columns, got {feature_row.shape[1]}."
        )

    if scaler is None:
        raise RuntimeError("StandardScaler artifact is missing.")

    return scaler.transform(feature_row)


def get_label(encoded_pred):
    if target_enc is None:
        return str(encoded_pred)
    return target_enc.inverse_transform([int(encoded_pred)])[0]


def predict(X):
    results = {}

    if rf_model is not None:
        rf_proba = rf_model.predict_proba(X)[0]
        rf_pred = rf_model.predict(X)[0]
        results["Random Forest"] = {
            "label": get_label(rf_pred),
            "probas": rf_proba,
            "classes": [get_label(c) for c in rf_model.classes_],
        }

    if ann_model is not None:
        ann_proba = ann_model.predict(X, verbose=0)[0]
        ann_pred = int(np.argmax(ann_proba))
        results["ANN (Neural Network)"] = {
            "label": get_label(ann_pred),
            "probas": ann_proba,
            "classes": [get_label(c) for c in range(len(ann_proba))],
        }

    if rf_model is not None and ann_model is not None:
        n_classes = len(ann_proba)
        rf_aligned = np.zeros(n_classes)
        for idx, cls in enumerate(rf_model.classes_):
            rf_aligned[int(cls)] = rf_proba[idx]

        ens_proba = (rf_aligned + ann_proba) / 2
        ens_pred = int(np.argmax(ens_proba))
        results["Ensemble (Average)"] = {
            "label": get_label(ens_pred),
            "probas": ens_proba,
            "classes": [get_label(c) for c in range(n_classes)],
        }

    return results


st.markdown(
    """
<div style='text-align:center; padding: 1.5rem 0 0.5rem'>
    <h1 style='font-size:2.6rem; margin-bottom:0.2rem'>🌌 StellarClass-ML</h1>
    <p style='font-size:1.05rem; color:#9aa0a6'>
        Automated Stellar Classification · Cardiff Metropolitan University · CIS6005
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

with st.sidebar:
    st.header("⚙️ Settings")

    available_models = []
    if rf_model is not None:
        available_models.append("Random Forest")
    if ann_model is not None:
        available_models.append("ANN (Neural Network)")
    if rf_model is not None and ann_model is not None:
        available_models.append("Ensemble (Average)")

    if not available_models:
        st.error("No model artifacts could be loaded. Check the models/ directory.")
        st.stop()

    model_choice = st.radio("Select Model", options=available_models)

    st.divider()
    st.markdown("### 📖 Feature Guide")
    st.markdown(
        """
| Feature | Description |
|---|---|
| `alpha` | Right ascension (degrees) |
| `delta` | Declination (degrees) |
| `u` | Ultraviolet magnitude |
| `g` | Green magnitude |
| `r` | Red magnitude |
| `i` | Near-IR magnitude |
| `z` | Infrared magnitude |
| `redshift` | Cosmological redshift |
| `spectral_type` | Stellar spectral class |
| `galaxy_population` | Galaxy population type |
"""
    )
    st.caption("Models trained on Kaggle Playground Series S6E6")


st.subheader("🔭 Enter Object Parameters")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Position**")
        alpha = st.number_input("Right Ascension (alpha)", 0.0, 360.0, 180.09, format="%.6f")
        delta = st.number_input("Declination (delta)", -90.0, 90.0, 0.0, format="%.6f")

        st.markdown("**🔬 Spectral Properties**")
        spectral_options = list(spectral_enc.classes_) if spectral_enc is not None else ["M"]
        spectral_type = st.selectbox(
            "Spectral Type",
            spectral_options,
            index=spectral_options.index("M") if "M" in spectral_options else 0,
        )
        galaxy_options = list(galaxy_enc.classes_) if galaxy_enc is not None else ["Blue_Cloud"]
        galaxy_population = st.selectbox("Galaxy Population", galaxy_options)

    with col2:
        st.markdown("**🎨 Photometric Magnitudes (SDSS bands)**")
        u = st.number_input("u (ultraviolet)", -5.0, 35.0, 22.07, format="%.4f")
        g = st.number_input("g (green)", -5.0, 35.0, 21.0, format="%.4f")
        r = st.number_input("r (red)", -5.0, 35.0, 20.0, format="%.4f")

    with col3:
        st.markdown("**🎨 (continued)**")
        i = st.number_input("i (near-infrared)", -5.0, 35.0, 19.5, format="%.4f")
        z = st.number_input("z (infrared)", -5.0, 35.0, 19.0, format="%.4f")

        st.markdown("**🌊 Redshift**")
        redshift = st.number_input("Redshift", -0.1, 8.0, 0.45, format="%.6f")

    submitted = st.form_submit_button("🚀 Predict Class", use_container_width=True)


if submitted:
    try:
        with st.spinner("Classifying object..."):
            X = build_feature_row(alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population)
            all_results = predict(X)

        result = all_results.get(model_choice)
        if result is None:
            st.error("Selected model is unavailable.")
            st.stop()

        label = result["label"]
        info = CLASS_INFO.get(label, {})

        st.divider()
        st.subheader("🎯 Prediction Result")

        colour = info.get("colour", "#888888")
        emoji = info.get("emoji", "❓")
        desc = info.get("desc", "")

        st.markdown(
            f"""
<div style='background: linear-gradient(135deg, {colour}22, {colour}11); border: 2px solid {colour}; border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1rem;'>
    <h2 style='color:{colour}; margin:0'>{emoji} {label}</h2>
    <p style='margin:0.5rem 0 0; color:#c8c8c8'>{desc}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("#### Confidence Scores")
        prob_cols = st.columns(len(result["classes"]))
        for col, cls, prob in zip(prob_cols, result["classes"], result["probas"]):
            c_info = CLASS_INFO.get(cls, {})
            with col:
                st.metric(f"{c_info.get('emoji', '')} {cls}", f"{prob * 100:.2f}%")
                st.progress(float(prob))

        if len(all_results) > 1:
            st.divider()
            st.markdown("#### 🔍 Model Comparison")
            comp_cols = st.columns(len(all_results))
            for col, (m_name, m_res) in zip(comp_cols, all_results.items()):
                m_info = CLASS_INFO.get(m_res["label"], {})
                with col:
                    active = "🟢" if m_name == model_choice else "⚪"
                    st.markdown(f"**{active} {m_name}**")
                    st.markdown(f"Prediction: **{m_info.get('emoji', '')} {m_res['label']}**")
                    st.caption(f"Confidence: {max(m_res['probas']) * 100:.1f}%")

        st.divider()
        st.markdown("#### 🎨 Derived Colour Indices")
        ci_cols = st.columns(4)
        indices = [("u-g", u - g), ("g-r", g - r), ("r-i", r - i), ("i-z", i - z)]
        for col, (name, value) in zip(ci_cols, indices):
            with col:
                st.metric(name, f"{value:.3f}")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
