import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

# ------------------ Paths ------------------
MODEL_PATH = Path("models/house_price_pipeline.joblib")
METRICS_PATH = Path("assets/metrics.json")
PLOT_CORR = Path("assets/top_corr_numeric.png")
PLOT_AVP_TEST = Path("assets/actual_vs_pred_test.png")
PLOT_AVP_TRAIN = Path("assets/actual_vs_pred_train.png")
PLOT_FI = Path("assets/feature_importance_top30.png")

st.set_page_config(page_title="Aurora – House Price Predictor", layout="wide")
st.sidebar.title("Aurora Navigation")
tabs = ["Summary", "Correlations", "Predict", "Hypotheses", "Tech"]
choice = st.sidebar.radio("Go to", tabs)

# ------------------ Safe loaders ------------------
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_metrics(path: Path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

pipe = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

# ------------------ Helpers ------------------
def get_ct(pipe_):
    return pipe_.named_steps.get("preprocess") or pipe_.named_steps.get("preprocessor")

def get_expected_schema(pipe_):
    """
    Extract expected raw input columns from ColumnTransformer.
    Returns dict: {"all": [...], "num": [...], "cat": [...]}
    """
    schema = {"all": [], "num": [], "cat": []}
    if pipe_ is None:
        return schema
    ct = get_ct(pipe_)
    if ct is None or not hasattr(ct, "transformers_"):
        return schema

    for name, trans, cols in ct.transformers_:
        # Skip remainder or dropped parts
        if name == "remainder" or cols is None or (isinstance(trans, str) and trans == "drop"):
            continue
        cols = list(cols)
        schema["all"].extend(cols)
        lname = (name or "").lower()
        if "num" in lname:
            schema["num"].extend(cols)
        elif "cat" in lname:
            schema["cat"].extend(cols)
        # if names don't include num/cat, we leave them only in "all"

    # de-dup preserving order
    schema["all"] = list(dict.fromkeys(schema["all"]))
    schema["num"] = list(dict.fromkeys(schema["num"]))
    schema["cat"] = list(dict.fromkeys(schema["cat"]))
    return schema

def align_to_expected_columns(df: pd.DataFrame, schema):
    """
    Reindex df to exactly the expected raw input columns (same order).
    - Add missing columns filled with np.nan (sklearn-friendly)
    - Coerce numeric columns to numbers
    """
    expected = schema.get("all", [])
    if not expected:
        return df

    aligned = df.reindex(columns=expected)
    # Fill any newly created missing columns with np.nan
    for c in expected:
        if c not in df.columns:
            aligned[c] = np.nan

    # Coerce numeric to float (errors → NaN)
    for c in schema.get("num", []):
        if c in aligned.columns:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")

    # Ensure order
    return aligned[expected]

def show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing: `{path.as_posix()}`")

# ------------------ UI ------------------
if choice == "Summary":
    st.title("Aurora: House Price Prediction Dashboard")
    st.markdown(
        """
**Business Goals**
- Discover which attributes correlate most with **SalePrice**  
- Predict **SalePrice** for the 4 inherited houses and any new house  
**Success metric:** R² ≥ 0.75 on train & test
        """
    )
    c1, c2 = st.columns(2)
    c1.metric("R² (Train)", f"{metrics.get('r2_train', 'N/A')}")
    c2.metric("R² (Test)", f"{metrics.get('r2_test', 'N/A')}")

    st.divider()
    st.subheader("Actual vs Predicted")
    show_image_if_exists(PLOT_AVP_TRAIN, "Train")
    show_image_if_exists(PLOT_AVP_TEST, "Test")

elif choice == "Correlations":
    st.title("Correlations with SalePrice")
    show_image_if_exists(PLOT_CORR, "Top numeric correlations")
    st.divider()
    st.subheader("Feature Importances (if available)")
    show_image_if_exists(PLOT_FI, "Top 30 Feature Importances")

elif choice == "Predict":
    st.title("Predict House Price")
    if pipe is None:
        st.error("Model not found. Please add `models/house_price_pipeline.joblib`.")
    else:
        schema = get_expected_schema(pipe)
        with st.expander("Model input schema", expanded=False):
            st.caption(f"Model expects {len(schema.get('all', []))} input columns.")
            if schema.get("all"):
                st.code(", ".join(schema["all"][:30]) + (" ..." if len(schema["all"]) > 30 else ""))

        # ---- Single prediction form ----
        st.subheader("Single Prediction")
        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                lotarea = st.number_input("Lot Area", min_value=500, max_value=200000, value=8450, step=50)
                neighborhood = st.text_input("Neighborhood", "CollgCr")
            with col2:
                overallqual = st.slider("Overall Quality (1–10)", 1, 10, 7)
                yearbuilt = st.number_input("Year Built", min_value=1875, max_value=2025, value=2003)
            with col3:
                grlivarea = st.number_input("Ground Living Area (sq ft)", min_value=200, max_value=8000, value=1710, step=10)
                listing_price = st.number_input("Your Intended Listing Price ($)", min_value=10000, max_value=3000000, value=250000, step=5000)

            submitted = st.form_submit_button("Predict & Evaluate")

        if submitted:
            # User input → minimal df
            single_df = pd.DataFrame([{
                "LotArea": lotarea,
                "OverallQual": overallqual,
                "YearBuilt": yearbuilt,
                "GrLivArea": grlivarea,
                "Neighborhood": neighborhood
            }])

            # Align to model's expected schema
            single_df_aligned = align_to_expected_columns(single_df, schema)

            try:
                pred = float(pipe.predict(single_df_aligned)[0])
                st.success(f"Predicted Market Price: ${pred:,.0f}")
                st.caption("(Your inputs were aligned to the model's full schema automatically.)")

                # Sellability-style heuristic vs user's listing price
                st.write(f"Your Intended Listing Price: ${listing_price:,.0f}")
                diff_ratio = (listing_price - pred) / max(pred, 1e-9)
                if diff_ratio > 0.05:
                    st.warning("❌ Overpriced (>5% above market). May take longer to sell unless adjusted.")
                elif diff_ratio < -0.05:
                    st.info("⚠️ Undervalued (>5% below market). Likely quick sale but you might leave money on the table.")
                else:
                    st.success("✅ Fairly priced. Balanced chance of sale at market value.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.divider()

        # ---- Batch prediction ----
        st.subheader("Batch Predictions (CSV)")
        uploaded = st.file_uploader("Upload CSV with one row per property", type=["csv"])
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                batch_aligned = align_to_expected_columns(batch_df, schema)
                preds = pipe.predict(batch_aligned)
                out = batch_df.copy()
                out["Predicted_SalePrice"] = preds
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    out.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

elif choice == "Hypotheses":
    st.title("Hypotheses & Validation")
    st.markdown(
        """
- **H1:** `OverallQual` has a strong positive relationship with `SalePrice`. ✅  
- **H2:** `GrLivArea` is a top numeric driver of price. ✅  
- **H3:** Newer `YearBuilt` increases `SalePrice`, controlling for size/quality. ✅  
        """
    )

elif choice == "Tech":
    st.title("Technical Details")
    st.write("### Pipeline")
    if pipe is not None:
        st.text(pipe)
    else:
        st.info("Model not loaded.")
    st.write("### Metrics")
    if metrics:
        st.json(metrics)
    else:
        st.info("No metrics found. Make sure `assets/metrics.json` exists.")
