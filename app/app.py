import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StellarClass-ML",
    page_icon="🌌",
    layout="wide",
)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_artifacts():
    scaler       = joblib.load(os.path.join(MODELS_DIR, "standard_scaler.pkl"))
    spectral_enc = joblib.load(os.path.join(MODELS_DIR, "spectral_encoder.pkl"))
    galaxy_enc   = joblib.load(os.path.join(MODELS_DIR, "galaxy_encoder.pkl"))
    target_enc   = joblib.load(os.path.join(MODELS_DIR, "target_encoder.pkl"))

    # Random Forest – may fail with MemoryError on large models
    rf_model = None
    try:
        rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    except (MemoryError, Exception) as e:
        st.warning(f"⚠️ Random Forest could not be loaded ({type(e).__name__}). ANN-only mode.")

    # ANN – try keras 3 standalone first, then tensorflow.keras
    ann_model = None
    try:
        import keras
        ann_model = keras.models.load_model(os.path.join(MODELS_DIR, "ann_best_model.keras"))
    except Exception:
        try:
            import tensorflow as tf
            ann_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "ann_best_model.keras"))
        except Exception:
            pass

    return scaler, spectral_enc, galaxy_enc, target_enc, rf_model, ann_model

scaler, spectral_enc, galaxy_enc, target_enc, rf_model, ann_model = load_artifacts()

# ── Class metadata ──────────────────────────────────────────────────────────────
CLASS_INFO = {
    "GALAXY": {
        "emoji": "🌀",
        "colour": "#4A90D9",
        "desc": "A gravitationally bound system of stars, gas, dust, and dark matter. "
                "Galaxies span from dwarf galaxies with a few million stars to giants "
                "hosting hundreds of trillions.",
    },
    "STAR": {
        "emoji": "⭐",
        "colour": "#F5A623",
        "desc": "A luminous sphere of plasma held together by its own gravity. "
                "Nuclear fusion in the core converts hydrogen into helium, "
                "releasing enormous amounts of energy.",
    },
    "QSO": {
        "emoji": "✨",
        "colour": "#9B59B6",
        "desc": "A Quasi-Stellar Object (quasar) — an extremely luminous active galactic "
                "nucleus powered by a supermassive black hole accreting matter. "
                "Quasars are among the brightest objects in the universe.",
    },
}

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem'>
    <h1 style='font-size:2.6rem; margin-bottom:0.2rem'>🌌 StellarClass-ML</h1>
    <p style='font-size:1.1rem; color:#aaa'>
        Automated Stellar Classification · Cardiff Metropolitan University · CIS6005
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    # Build available model list based on what loaded successfully
    available_models = []
    if rf_model:  available_models.append("Random Forest")
    if ann_model: available_models.append("ANN (Neural Network)")
    if rf_model and ann_model: available_models.append("Ensemble (Average)")
    if not available_models:
        st.error("❌ No models could be loaded. Check the models/ directory.")
        st.stop()

    model_choice = st.radio(
        "Select Model",
        options=available_models,
        help="Choose which trained model to use for prediction.",
    )
    st.divider()
    st.markdown("### 📖 Feature Guide")
    st.markdown("""
| Feature | Description |
|---------|-------------|
| **alpha** | Right ascension (degrees) |
| **delta** | Declination (degrees) |
| **u** | Ultraviolet magnitude |
| **g** | Green magnitude |
| **r** | Red magnitude |
| **i** | Near-IR magnitude |
| **z** | Infrared magnitude |
| **redshift** | Cosmological redshift |
| **spectral_type** | Stellar spectral class |
| **galaxy_population** | Galaxy population type |
""")
    st.divider()
    st.caption("Models trained on Kaggle Playground Series S6E6")

# ── Input form ──────────────────────────────────────────────────────────────────
st.subheader("🔭 Enter Object Parameters")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Position**")
        alpha = st.number_input("Right Ascension (alpha)", 0.0, 360.0, 180.0,
                                format="%.6f", help="0 – 360 degrees")
        delta = st.number_input("Declination (delta)", -90.0, 90.0, 0.0,
                                format="%.6f", help="-90 – +90 degrees")

        st.markdown("**🔬 Spectral Properties**")
        spectral_options = list(spectral_enc.classes_)
        spectral_type = st.selectbox("Spectral Type", spectral_options,
                                     index=spectral_options.index("M") if "M" in spectral_options else 0)
        galaxy_options = list(galaxy_enc.classes_)
        galaxy_population = st.selectbox("Galaxy Population", galaxy_options)

    with col2:
        st.markdown("**🎨 Photometric Magnitudes (SDSS bands)**")
        u = st.number_input("u (ultraviolet)", -5.0, 35.0, 22.0, format="%.4f")
        g = st.number_input("g (green)",       -5.0, 35.0, 21.0, format="%.4f")
        r = st.number_input("r (red)",         -5.0, 35.0, 20.0, format="%.4f")

    with col3:
        st.markdown("**🎨 (continued)**")
        i = st.number_input("i (near-infrared)", -5.0, 35.0, 19.5, format="%.4f")
        z = st.number_input("z (infrared)",      -5.0, 35.0, 19.0, format="%.4f")

        st.markdown("**🌊 Redshift**")
        redshift = st.number_input("Redshift", -0.1, 8.0, 0.5, format="%.6f",
                                   help="Cosmological redshift (z). Galaxies ≈ 0–1, QSOs can exceed 3.")

    submitted = st.form_submit_button("🚀 Predict Class", use_container_width=True)

# ── Prediction logic ─────────────────────────────────────────────────────────────
def preprocess(alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population):
    spectral_encoded = spectral_enc.transform([spectral_type])[0]
    galaxy_encoded   = galaxy_enc.transform([galaxy_population])[0]

    # Recreate the exact 22-feature training matrix
    feature_row = np.array([[
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
    ]], dtype=float)

    num_scaled = scaler.transform(feature_row)

    return num_scaled

def get_class_label(encoded_pred):
    return target_enc.inverse_transform([encoded_pred])[0]

def predict(X):
    results = {}

    if rf_model:
        rf_proba = rf_model.predict_proba(X)[0]
        rf_pred  = rf_model.predict(X)[0]
        results["Random Forest"] = {
            "label":   get_class_label(rf_pred),
            "probas":  rf_proba,
            "classes": [get_class_label(c) for c in rf_model.classes_],
        }

    if ann_model:
        ann_proba = ann_model.predict(X, verbose=0)[0]
        ann_pred  = int(np.argmax(ann_proba))
        results["ANN (Neural Network)"] = {
            "label":   get_class_label(ann_pred),
            "probas":  ann_proba,
            "classes": [get_class_label(c) for c in range(len(ann_proba))],
        }

    if rf_model and ann_model:
        n_classes  = len(ann_proba)
        rf_aligned = np.zeros(n_classes)
        for idx, cls in enumerate(rf_model.classes_):
            rf_aligned[int(cls)] = rf_proba[idx]
        ens_proba = (rf_aligned + ann_proba) / 2
        ens_pred  = int(np.argmax(ens_proba))
        results["Ensemble (Average)"] = {
            "label":   get_class_label(ens_pred),
            "probas":  ens_proba,
            "classes": [get_class_label(c) for c in range(n_classes)],
        }

    return results


if submitted:
    with st.spinner("Classifying object..."):
        X = preprocess(alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population)
        all_results = predict(X)

    result = all_results.get(model_choice) or all_results["Random Forest"]
    label  = result["label"]
    info   = CLASS_INFO.get(label, {})

    st.divider()
    st.subheader("🎯 Prediction Result")

    # Main result card
    colour = info.get("colour", "#888")
    emoji  = info.get("emoji", "❓")
    desc   = info.get("desc", "")

    st.markdown(f"""
<div style='
    background: linear-gradient(135deg, {colour}22, {colour}11);
    border: 2px solid {colour};
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
'>
    <h2 style='color:{colour}; margin:0'>{emoji} {label}</h2>
    <p style='margin:0.5rem 0 0; color:#ccc'>{desc}</p>
</div>
""", unsafe_allow_html=True)

    # Probability bars
    st.markdown("#### Confidence Scores")
    prob_cols = st.columns(len(result["classes"]))
    for col, cls, prob in zip(prob_cols, result["classes"], result["probas"]):
        c_info  = CLASS_INFO.get(cls, {})
        c_color = c_info.get("colour", "#888")
        c_emoji = c_info.get("emoji", "")
        with col:
            st.metric(f"{c_emoji} {cls}", f"{prob*100:.2f}%")
            st.progress(float(prob))

    # All models comparison (if ANN loaded)
    if len(all_results) > 1:
        st.divider()
        st.markdown("#### 🔍 Model Comparison")
        comp_cols = st.columns(len(all_results))
        for col, (m_name, m_res) in zip(comp_cols, all_results.items()):
            m_label = m_res["label"]
            m_info  = CLASS_INFO.get(m_label, {})
            with col:
                active = "🟢" if m_name == model_choice else "⚪"
                st.markdown(f"**{active} {m_name}**")
                st.markdown(f"Prediction: **{m_info.get('emoji','')} {m_label}**")
                top_prob = max(m_res["probas"])
                st.caption(f"Confidence: {top_prob*100:.1f}%")

    # Derived colour indices
    st.divider()
    st.markdown("#### 🎨 Derived Colour Indices")
    ci_cols = st.columns(4)
    indices = [("u−g", u - g), ("g−r", g - r), ("r−i", r - i), ("i−z", i - z)]
    for col, (name, val) in zip(ci_cols, indices):
        with col:
            st.metric(name, f"{val:.3f}")
