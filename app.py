import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- Config ----------
st.set_page_config(page_title="Aurora – Ames Housing", layout="wide")
MODELS_DIR = Path("models")
ASSETS_DIR = Path("assets")
MODEL_PATH = MODELS_DIR / "house_price_pipeline.joblib"
METRICS_JSON = ASSETS_DIR / "metrics.json"
PLOT_CORR = ASSETS_DIR / "top_corr_numeric.png"
PLOT_AVP_TRAIN = ASSETS_DIR / "actual_vs_pred_train.png"
PLOT_AVP_TEST  = ASSETS_DIR / "actual_vs_pred_test.png"
PLOT_FI        = ASSETS_DIR / "feature_importance_top30.png"

# ---------- Helpers ----------
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_metrics():
    import json
    if METRICS_JSON.exists():
        with open(METRICS_JSON, "r") as f:
            return json.load(f)
    return {}

def show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing: `{path.as_posix()}`")

def predict_df(pipe, df: pd.DataFrame) -> pd.DataFrame:
    preds = pipe.predict(df)
    out = df.copy()
    out["Predicted_SalePrice"] = preds
    return out

# ---------- App UI ----------
st.title("Aurora 🌅 – Ames Housing Price App")

tabs = st.tabs(["📘 Summary", "📊 Correlations", "🏡 Predict", "🔎 Hypotheses", "⚙️ Tech"])

with tabs[0]:  # Summary
    st.subheader("Project Summary")
    st.markdown("""
**Business Requirements**
1) Identify features most correlated with **SalePrice**  
2) Predict **SalePrice** for 4 inherited houses and any new house  
**Success Metric:** R² ≥ 0.75 (train & test)
    """)
    metrics = load_metrics()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² (Train)", f"{metrics.get('r2_train','N/A')}")
    with c2:
        st.metric("R² (Test)",  f"{metrics.get('r2_test','N/A')}")
    st.divider()
    st.write("Actual vs Predicted (Train/Test)")
    show_image_if_exists(PLOT_AVP_TRAIN, "Actual vs Predicted — Train")
    show_image_if_exists(PLOT_AVP_TEST,  "Actual vs Predicted — Test")

with tabs[1]:  # Correlations
    st.subheader("Correlations / EDA")
    show_image_if_exists(PLOT_CORR, "Top numeric correlations with SalePrice")
    st.divider()
    st.subheader("Feature Importance (Global)")
    show_image_if_exists(PLOT_FI, "Top 30 Feature Importances")

with tabs[2]:  # Predict
    st.subheader("Predict House Price")
    pipe = load_model()
    if pipe is None:
        st.warning("Model not found. Please add `models/house_price_pipeline.joblib`.")
    else:
        mode = st.radio("Input mode:", ["Form (single)", "CSV (batch)"], horizontal=True)

        if mode == "Form (single)":
            # NOTE: These fields must exist in your training pipeline schema.
            c1, c2, c3 = st.columns(3)
            with c1:
                lot_area = st.number_input("LotArea", min_value=200, value=8450, step=50)
                neighborhood = st.text_input("Neighborhood", "CollgCr")
            with c2:
                overall_qual = st.slider("OverallQual", 1, 10, 7)
                year_built = st.number_input("YearBuilt", 1800, 2024, 2003)
            with c3:
                gr_liv_area = st.number_input("GrLivArea", 200, 6000, 1710, step=10)

            X = pd.DataFrame([{
                "LotArea": lot_area,
                "OverallQual": overall_qual,
                "YearBuilt": year_built,
                "GrLivArea": gr_liv_area,
                "Neighborhood": neighborhood
            }])

            if st.button("🚀 Predict", use_container_width=True):
                res = predict_df(pipe, X)
                st.success(f"Predicted SalePrice: ${res['Predicted_SalePrice'].iloc[0]:,.0f}")
                st.dataframe(res, use_container_width=True)

        else:
            file = st.file_uploader("Upload CSV matching training schema", type=["csv"])
            if file:
                df = pd.read_csv(file)
                res = predict_df(pipe, df)
                st.dataframe(res, use_container_width=True)
                st.download_button(
                    "⬇️ Download predictions",
                    res.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

with tabs[3]:  # Hypotheses
    st.subheader("Hypotheses & Validation")
    st.markdown("""
- **H1:** OverallQual has a strong positive relationship with SalePrice — validate via correlation & importances.  
- **H2:** GrLivArea is among the top numeric drivers — validate via correlation & importances.  
- **H3:** Newer YearBuilt increases SalePrice — validate via correlation distribution & partial importances.  
    """)

with tabs[4]:  # Tech
    st.subheader("Technical Details")
    st.write("• Model: scikit-learn `Pipeline` with preprocessing + RandomForestRegressor")
    st.write("• Artifacts loaded from `assets/` and `models/`")
    st.code("models/house_price_pipeline.joblib", language="bash")
    st.code("assets/metrics.json", language="bash")
