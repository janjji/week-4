import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import cv2
import pytesseract
import sys
import re
from PIL import Image
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

# ══════════════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "src" / "models"
SRC_DIR    = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Driver Monitor",
    page_icon="🚗",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════
# MODEL LOADER (cached — loads once per session)
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    M = {}
    try:
        # --- Ratings ---
        M['ratings_xgb']      = joblib.load(MODELS_DIR / 'ratings_xgb.pkl')
        M['ratings_rf']       = joblib.load(MODELS_DIR / 'ratings_rf.pkl')
        with open(MODELS_DIR / 'ratings_feature_cols.json') as f:
            M['ratings_features'] = json.load(f)

        # --- NLP Sentiment ---
        M['nlp_tfidf']        = joblib.load(MODELS_DIR / 'nlp_tfidf.pkl')
        M['nlp_lr']           = joblib.load(MODELS_DIR / 'nlp_sentiment_lr.pkl')
        with open(MODELS_DIR / 'nlp_classes.json') as f:
            M['nlp_classes']  = json.load(f)

        # --- Violations ---
        M['viol_xgb']         = joblib.load(MODELS_DIR / 'violations_xgb_with_flags.pkl')
        M['viol_rf']          = joblib.load(MODELS_DIR / 'violations_rf_with_flags.pkl')
        with open(MODELS_DIR / 'violations_feature_cols_flags.json') as f:
            M['viol_features'] = json.load(f)

        # --- Violations Sensor ---
        M['viol_sensor_rf']   = joblib.load(MODELS_DIR / 'violations_sensor_rf.pkl')
        M['viol_sensor_le']   = joblib.load(MODELS_DIR / 'violations_sensor_le.pkl')
        with open(MODELS_DIR / 'violations_sensor_features.json') as f:
            M['viol_sensor_features'] = json.load(f)

        st.sidebar.success("✅ All models loaded!")

    except Exception as e:
        st.sidebar.error(f"❌ Model load error: {e}")

    return M

M = load_all_models()

# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
def preprocess_text(text):
    """Clean text the same way as training in Module 4."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.title("🚗 Smart Driver Monitoring Dashboard")
st.markdown(
    "AI-powered analytics for **driver ratings**, **violation detection**, "
    "**feedback sentiment**, and **document forgery checks**."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Driver Rating & Violations",
    "💬 Feedback Sentiment",
    "🪪 Document Forgery Check"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DRIVER RATING & VIOLATIONS
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Driver Rating Prediction & Violation Detection")

    # Two uploaders side by side
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🌟 For Driver Ratings**")
        st.caption("Upload: `transportation_logistic_cleaned.csv`")
        rating_csv = st.file_uploader("📂 Upload Ratings CSV", type=["csv"], key="ratings")
    with col_b:
        st.markdown("**⚠️ For Violations**")
        st.caption("Upload: `driver_behavior_cleaned.csv`")
        viol_csv = st.file_uploader("📂 Upload Violations CSV", type=["csv"], key="violations")

    st.divider()

    # ── RATINGS SECTION ───────────────────────────────────────────────
    st.subheader("🌟 Driver Rating Prediction")

    if rating_csv:
        df_rate = pd.read_csv(rating_csv)

        st.subheader("👀 Ratings Data Preview")
        st.dataframe(df_rate.head(), use_container_width=True)
        st.caption(f"Shape: {df_rate.shape[0]} rows × {df_rate.shape[1]} columns")

        try:
            # Feature engineering
            if 'on_time_int' not in df_rate.columns and 'On_Time' in df_rate.columns:
                df_rate['on_time_int'] = df_rate['On_Time'].astype(int)
            if 'maintenance_ratio' not in df_rate.columns:
                df_rate['maintenance_ratio'] = df_rate['Maintenance'] / (df_rate['Fixed Costs'] + 1)
            if 'delivery_per_hour' not in df_rate.columns:
                df_rate['delivery_per_hour'] = df_rate['Delivery_Time'] / (df_rate['Hour'] + 1)
            if 'is_weekend' not in df_rate.columns:
                df_rate['is_weekend'] = df_rate['Day_of_Week'].isin([5, 6]).astype(int)
            if 'is_rush_hour' not in df_rate.columns:
                df_rate['is_rush_hour'] = df_rate['Hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

            # Encode categoricals
            for col in ['Origin', 'Destination', 'Region', 'Weather']:
                if col in df_rate.columns and df_rate[col].dtype.name in ['category', 'object']:
                    df_rate[col] = df_rate[col].astype('category').cat.codes

            # Predict
            rating_features = M['ratings_features']
            missing = [c for c in rating_features if c not in df_rate.columns]

            if missing:
                st.warning(f"⚠️ Still missing columns: {missing}")
            else:
                X_rate = df_rate[rating_features]
                df_rate['predicted_rating'] = M['ratings_xgb'].predict(X_rate).clip(1, 5).round(2)

                st.subheader("📊 Prediction Results")
                display_cols = ['Origin', 'Destination', 'predicted_rating'] \
                    if 'Origin' in df_rate.columns else ['predicted_rating']
                st.dataframe(df_rate[display_cols].head(20), use_container_width=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", len(df_rate))
                col2.metric("Avg Predicted Rating", f"{df_rate['predicted_rating'].mean():.2f} ⭐")
                col3.metric("Low Ratings (<3)", int((df_rate['predicted_rating'] < 3).sum()))

        except Exception as e:
            st.error(f"Rating prediction failed: {e}")

    else:
        st.info("👆 Please upload `transportation_logistic_cleaned.csv` to see rating predictions.")

    st.divider()

    # ── VIOLATIONS SECTION ────────────────────────────────────────────
    st.subheader("⚠️ Violation Detection")

    if viol_csv:
        df_viol = pd.read_csv(viol_csv)

        st.subheader("👀 Violations Data Preview")
        st.dataframe(df_viol.head(), use_container_width=True)
        st.caption(f"Shape: {df_viol.shape[0]} rows × {df_viol.shape[1]} columns")

        try:
            # Feature engineering
            if 'avg_speed'        not in df_viol.columns: df_viol['avg_speed']        = df_viol['speed']
            if 'speed_variance'   not in df_viol.columns: df_viol['speed_variance']   = df_viol.groupby('trip_id')['speed'].transform('std').fillna(0)
            if 'max_accel'        not in df_viol.columns: df_viol['max_accel']        = df_viol['acceleration'].abs()
            if 'hard_brake_count' not in df_viol.columns: df_viol['hard_brake_count'] = (df_viol['brake_usage'] > 5).astype(int)
            if 'stop_count'       not in df_viol.columns: df_viol['stop_count']       = df_viol['stop_events']
            if 'distance_per_min' not in df_viol.columns: df_viol['distance_per_min'] = df_viol['trip_distance'] / (df_viol['trip_duration'] / 60 + 1)
            if 'overspeed_flag'   not in df_viol.columns: df_viol['overspeed_flag']   = (df_viol['speed'] > 80).astype(int)
            if 'hard_brake_flag'  not in df_viol.columns: df_viol['hard_brake_flag']  = (df_viol['brake_usage'] > 5).astype(int)
            if 'high_accel_flag'  not in df_viol.columns: df_viol['high_accel_flag']  = (df_viol['acceleration'].abs() > 3).astype(int)

            # Encode categoricals
            for col in ['weather_conditions', 'road_type', 'traffic_condition']:
                enc_col = col + '_enc'
                if enc_col not in df_viol.columns and col in df_viol.columns:
                    df_viol[enc_col] = df_viol[col].astype('category').cat.codes

            if 'geofencing_violation' in df_viol.columns:
                df_viol['geofencing_violation'] = df_viol['geofencing_violation'].astype(int)

            # Predict
            viol_features = M['viol_features']
            missing_v = [c for c in viol_features if c not in df_viol.columns]

            if missing_v:
                st.warning(f"⚠️ Still missing columns: {missing_v}")
            else:
                X_viol = df_viol[viol_features]
                df_viol['violation_flag']  = M['viol_xgb'].predict(X_viol)
                df_viol['violation_label'] = df_viol['violation_flag'].map(
                    {0: '✅ No Violation', 1: '🚨 Violation'}
                )

                if 'driver_id' in df_viol.columns:
                    viol_summary = df_viol.groupby('driver_id').agg(
                        violation_count=('violation_flag', 'sum'),
                        total_trips=('violation_flag', 'count')
                    ).reset_index()
                    viol_summary['violation_rate'] = (
                        viol_summary['violation_count'] / viol_summary['total_trips'] * 100
                    ).round(1).astype(str) + '%'
                    st.dataframe(viol_summary, use_container_width=True)
                else:
                    st.dataframe(
                        df_viol[['violation_label']].value_counts().reset_index(),
                        use_container_width=True
                    )

                col1, col2 = st.columns(2)
                col1.metric("🚨 Violations Detected", int(df_viol['violation_flag'].sum()))
                col2.metric("✅ Clean Rows",           int((df_viol['violation_flag'] == 0).sum()))

        except Exception as e:
            st.error(f"Violation detection failed: {e}")

    else:
        st.info("👆 Please upload `driver_behavior_cleaned.csv` to see violation predictions.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — FEEDBACK SENTIMENT
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("💬 Passenger Feedback Sentiment Analyzer")
    st.markdown("Analyze whether feedback is **positive**, **neutral**, or **negative**.")

    mode = st.radio(
        "Choose input mode:",
        ["✏️ Type feedback manually", "📂 Upload feedback CSV"],
        horizontal=True
    )

    st.divider()

    # ── MANUAL INPUT ──────────────────────────────────────────────────
    if mode == "✏️ Type feedback manually":
        user_text = st.text_area(
            "Enter passenger feedback:",
            height=150,
            placeholder="e.g. The driver was very rude and drove too fast..."
        )

        with st.expander("⚙️ Optional: Add Driver Behavior Data (improves accuracy)"):
            col1, col2, col3, col4 = st.columns(4)
            speed        = col1.number_input("Speed (km/h)",   value=40.0)
            brake_usage  = col2.number_input("Brake Usage",    value=3.0)
            acceleration = col3.number_input("Acceleration",   value=1.0)
            steering     = col4.number_input("Steering Angle", value=5.0)

        if st.button("🔍 Analyze Sentiment", use_container_width=True):
            if user_text.strip():
                try:
                    cleaned  = preprocess_text(user_text)
                    vec      = M['nlp_tfidf'].transform([cleaned])
                    behavior = csr_matrix([[speed, brake_usage, acceleration, steering]])
                    X_combined = hstack([vec, behavior])

                    pred    = M['nlp_lr'].predict(X_combined)[0]
                    proba   = M['nlp_lr'].predict_proba(X_combined)[0]
                    classes = M['nlp_classes']
                    label   = pred

                    emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(
                        label.lower(), "⚪"
                    )
                    st.markdown(f"### {emoji} Sentiment: **{label.upper()}**")
                    st.markdown(f"**Confidence:** {max(proba):.2%}")

                    proba_df = pd.DataFrame({
                        'Sentiment'  : classes,
                        'Probability': proba.round(3)
                    })
                    st.bar_chart(proba_df.set_index('Sentiment'))

                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")
            else:
                st.warning("⚠️ Please enter some feedback text first.")

    # ── CSV BATCH INPUT ───────────────────────────────────────────────
    else:
        fb_file  = st.file_uploader("📂 Upload feedback.csv", type=["csv"], key="feedback")
        beh_file = st.file_uploader(
            "📂 Upload driver_behavior_cleaned.csv (optional)",
            type=["csv"], key="behavior_nlp"
        )

        if fb_file:
            fb_df = pd.read_csv(fb_file)

            st.subheader("👀 Data Preview")
            st.dataframe(fb_df.head(), use_container_width=True)
            st.caption(f"Shape: {fb_df.shape[0]} rows × {fb_df.shape[1]} columns")

            if st.button("🔍 Run Batch Sentiment Analysis", use_container_width=True):
                try:
                    texts_cleaned = fb_df['feedback_text'].apply(preprocess_text).tolist()
                    X_tfidf       = M['nlp_tfidf'].transform(texts_cleaned)

                    if beh_file:
                        beh_df   = pd.read_csv(beh_file)
                        beh_vals = beh_df[['speed', 'brake_usage', 'acceleration', 'steering_angle']] \
                                         .fillna(0).iloc[:len(fb_df)].values
                    else:
                        beh_vals = np.tile([40.0, 3.0, 1.0, 5.0], (len(fb_df), 1))

                    X_combined = hstack([X_tfidf, csr_matrix(beh_vals)])
                    preds      = M['nlp_lr'].predict(X_combined)
                    fb_df['predicted_sentiment'] = preds

                    show_cols = ['feedback_text', 'predicted_sentiment']
                    if 'sentiment_label' in fb_df.columns:
                        show_cols.insert(1, 'sentiment_label')
                        fb_df['match'] = fb_df['sentiment_label'] == fb_df['predicted_sentiment']
                        show_cols.append('match')

                    st.subheader("📊 Results")
                    st.dataframe(fb_df[show_cols], use_container_width=True)

                    st.subheader("📈 Sentiment Distribution")
                    dist = fb_df['predicted_sentiment'].value_counts().reset_index()
                    dist.columns = ['Sentiment', 'Count']
                    st.bar_chart(dist.set_index('Sentiment'))

                    counts = fb_df['predicted_sentiment'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🟢 Positive", counts.get('positive', 0))
                    col2.metric("🟡 Neutral",  counts.get('neutral',  0))
                    col3.metric("🔴 Negative", counts.get('negative', 0))

                    if 'match' in fb_df.columns:
                        acc = fb_df['match'].mean()
                        st.success(f"✅ Accuracy vs ground truth: **{acc:.2%}**")

                except Exception as e:
                    st.error(f"Batch analysis failed: {e}")

        else:
            st.info("👆 Please upload `feedback.csv` to run batch sentiment analysis.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — DOCUMENT FORGERY CHECK
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🪪 Vehicle Document & License Forgery Check")
    st.markdown(
        "Upload a truck/vehicle image to run **OCR plate extraction**, "
        "**image forensics**, and **CNN forgery detection**."
    )

    img_file = st.file_uploader(
        "📂 Upload Vehicle Image",
        type=["jpg", "jpeg", "png"],
        key="license"
    )

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", width=400)

        if st.button("🔍 Run Forgery Check", use_container_width=True):
            with st.spinner("Analyzing image... this may take a few seconds."):
                try:
                    import tempfile, os
                    from forgery_check import check_vehicle

                    # Save to temp file (check_vehicle needs a file path)
                    suffix = Path(img_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(img_file.getvalue())
                        tmp_path = tmp.name

                    result = check_vehicle(tmp_path)
                    os.unlink(tmp_path)

                    st.divider()

                    # Risk Level Banner
                    risk = result['risk_level']
                    if risk == "HIGH":
                        st.error(f"🚨 RISK LEVEL: **{risk}** — Likely Suspicious!")
                    elif risk == "MEDIUM":
                        st.warning(f"⚠️ RISK LEVEL: **{risk}** — Needs Review")
                    else:
                        st.success(f"✅ RISK LEVEL: **{risk}** — Looks Genuine")

                    st.divider()

                    # Detection Scores
                    st.subheader("📊 Detection Scores")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🤖 CNN Suspicion Score",  f"{result['cnn_score']:.2%}",      help="Higher = more suspicious")
                    col2.metric("🔍 Final Ensemble Score", f"{result['suspicion_score']:.2%}", help="Combined OCR + CNN + Forensics")
                    col3.metric("📄 OCR Confidence",       f"{result['ocr_conf']:.1f}%",       help="Tesseract text confidence")

                    st.markdown("**Overall Suspicion Level:**")
                    st.progress(float(result['suspicion_score']))

                    st.divider()

                    # OCR Results
                    st.subheader("📄 OCR Plate Extraction")
                    col1, col2 = st.columns(2)
                    col1.metric("Plate Text Detected", result['ocr_text'] if result['ocr_text'] else "None found")
                    col2.metric("Plate Region Found",  "✅ Yes" if result['roi_found'] else "❌ No")

                    st.divider()

                    # Image Forensics
                    st.subheader("🔬 Image Forensics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🌫️ Blur Score",    f"{result['blur_score']:.1f}",  help="Lower = blurrier (threshold: 654.3)")
                    col2.metric("📡 Noise Score",   f"{result['noise_score']:.2f}", help="Higher = noisier (threshold: 16.6)")
                    col3.metric("🚩 Forensics Flag", "🚨 Flagged" if result['forensics_flag'] else "✅ Normal")

                    st.divider()

                    # Full Raw Results
                    with st.expander("📋 View Full Raw Results"):
                        st.json(result)

                except FileNotFoundError as e:
                    st.error(f"❌ Model not found: {e}")
                    st.info("Make sure `mobilenetv2_truck_classifier.keras` exists in `outputs/module6/`")
                except ImportError as e:
                    st.error(f"❌ Could not import forgery_check.py: {e}")
                    st.info("Make sure `src/forgery_check.py` exists and SRC_DIR is in sys.path")
                except Exception as e:
                    st.error(f"❌ Forgery check failed: {e}")

    else:
        st.info("👆 Upload a vehicle or truck image to run the forgery detection pipeline.")
