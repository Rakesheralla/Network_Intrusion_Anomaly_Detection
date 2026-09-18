import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CyberGuard - Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #07111f 0%,
            #0b1728 50%,
            #101b2d 100%
        );
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    /* Section titles */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    .metric-title {
        font-size: 15px;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Model cards */
    .model-card {
        background: rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 15px;
    }

    .model-name {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Alert box */
    .alert-box {
        background: rgba(255, 80, 80, 0.12);
        border: 1px solid rgba(255, 80, 80, 0.35);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }

    .safe-box {
        background: rgba(80, 200, 120, 0.12);
        border: 1px solid rgba(80, 200, 120, 0.35);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.55;
        margin-top: 50px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

scaler = joblib.load("../Models/scaler.pkl")

isolation_forest = joblib.load(
    "../Models/isolation_forest.pkl"
)

autoencoder = load_model(
    "../Models/autoencoder.keras"
)

feature_columns = joblib.load(
    "../Models/feature_columns.pkl"
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ CyberGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Network Intrusion & Anomaly Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📂 Upload Network Traffic</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a CICIDS2017 network traffic CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # =====================================================
    # READ DATA
    # =====================================================

    data = pd.read_csv(uploaded_file)

    total_flows = len(data)


    st.success(
        "✅ Network traffic data loaded successfully!"
    )


    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Traffic Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Flows",
            f"{total_flows:,}"
        )

    with col2:
        st.metric(
            "Features",
            "78"
        )

    with col3:
        st.metric(
            "ML Models",
            "2"
        )

    with col4:
        st.metric(
            "Detection Type",
            "Anomaly"
        )


    # =====================================================
    # DATA PREVIEW
    # =====================================================

    with st.expander("🔎 Preview Network Traffic"):

        st.dataframe(
            data.head(10),
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # FEATURE CHECK
    # =====================================================

    missing_features = [
        col
        for col in feature_columns
        if col not in data.columns
    ]


    if missing_features:

        st.error(
            f"❌ {len(missing_features)} required features are missing."
        )

        st.stop()


    # =====================================================
    # PREPROCESSING
    # =====================================================

    X_new = data[feature_columns]

    X_new_scaled = scaler.transform(
        X_new
    )


    # =====================================================
    # ISOLATION FOREST
    # =====================================================

    isolation_prediction = (
        isolation_forest.predict(
            X_new_scaled
        )
    )

    isolation_result = np.where(
        isolation_prediction == -1,
        "Anomaly",
        "Normal"
    )


    isolation_normal = np.sum(
        isolation_result == "Normal"
    )

    isolation_anomaly = np.sum(
        isolation_result == "Anomaly"
    )

    isolation_rate = (
        isolation_anomaly / total_flows
    ) * 100


    # =====================================================
    # AUTOENCODER
    # =====================================================

    X_reconstructed = autoencoder.predict(
        X_new_scaled,
        verbose=0
    )


    reconstruction_error = np.mean(
        np.square(
            X_new_scaled - X_reconstructed
        ),
        axis=1
    )


    threshold = 0.2057502607068837


    autoencoder_result = np.where(
        reconstruction_error > threshold,
        "Anomaly",
        "Normal"
    )


    autoencoder_normal = np.sum(
        autoencoder_result == "Normal"
    )

    autoencoder_anomaly = np.sum(
        autoencoder_result == "Anomaly"
    )

    autoencoder_rate = (
        autoencoder_anomaly / total_flows
    ) * 100


    # =====================================================
    # DETECTION RESULTS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🚨 Detection Results</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # ISOLATION FOREST CARD
    # =====================================================

    st.markdown(
        '<div class="model-card">'
        '<div class="model-name">🌲 Isolation Forest</div>'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Normal",
            f"{isolation_normal:,}"
        )

    with col2:
        st.metric(
            "Anomaly",
            f"{isolation_anomaly:,}"
        )

    with col3:
        st.metric(
            "Anomaly Rate",
            f"{isolation_rate:.2f}%"
        )


    # =====================================================
    # AUTOENCODER CARD
    # =====================================================

    st.markdown(
        '<div class="model-card">'
        '<div class="model-name">🧠 Autoencoder</div>'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Normal",
            f"{autoencoder_normal:,}"
        )

    with col2:
        st.metric(
            "Anomaly",
            f"{autoencoder_anomaly:,}"
        )

    with col3:
        st.metric(
            "Anomaly Rate",
            f"{autoencoder_rate:.2f}%"
        )


    # =====================================================
    # CHART
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📈 Model Comparison</div>',
        unsafe_allow_html=True
    )


    comparison_df = pd.DataFrame({
        "Model": [
            "Isolation Forest",
            "Autoencoder"
        ],
        "Normal": [
            isolation_normal,
            autoencoder_normal
        ],
        "Anomaly": [
            isolation_anomaly,
            autoencoder_anomaly
        ]
    })


    chart_data = comparison_df.set_index(
        "Model"
    )


    st.bar_chart(
        chart_data
    )


    # =====================================================
    # CONSENSUS
    # =====================================================

    both_anomaly = np.sum(
        (isolation_result == "Anomaly")
        &
        (autoencoder_result == "Anomaly")
    )


    both_normal = np.sum(
        (isolation_result == "Normal")
        &
        (autoencoder_result == "Normal")
    )


    disagreement = np.sum(
        isolation_result != autoencoder_result
    )


    st.divider()

    st.markdown(
        '<div class="section-title">🤝 Model Consensus</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Both → Anomaly",
            f"{both_anomaly:,}"
        )


    with col2:
        st.metric(
            "Both → Normal",
            f"{both_normal:,}"
        )


    with col3:
        st.metric(
            "Models Disagree",
            f"{disagreement:,}"
        )


    # =====================================================
    # SECURITY STATUS
    # =====================================================

    if both_anomaly > 0:

        st.markdown(
            '<div class="alert-box">'
            '<h3>🚨 Suspicious Network Activity Detected</h3>'
            '<p>Both models identified anomalous traffic.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="safe-box">'
            '<h3>✅ No Consensus Anomalies Detected</h3>'
            '<p>The models did not jointly identify anomalous flows.</p>'
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # FLOW RESULTS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🔍 Flow-Level Predictions</div>',
        unsafe_allow_html=True
    )


    results_df = pd.DataFrame({
        "Flow": range(
            1,
            total_flows + 1
        ),
        "Isolation Forest": isolation_result,
        "Autoencoder": autoencoder_result
    })


    results_df["Consensus"] = np.where(
        results_df["Isolation Forest"]
        ==
        results_df["Autoencoder"],
        results_df["Isolation Forest"],
        "Review"
    )


    st.dataframe(
        results_df.head(100),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'CyberGuard • Network Intrusion Detection • '
    'Machine Learning + Deep Learning'
    '</div>',
    unsafe_allow_html=True
)