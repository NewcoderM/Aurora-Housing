import streamlit as st

st.set_page_config(page_title="Aurora – Ames Housing", layout="wide")
st.title("Aurora 🌅 – Ames Housing Price App")

tabs = st.tabs(["📘 Summary", "📊 Correlations", "🏡 Predict", "🔎 Hypotheses", "⚙️ Tech"])

with tabs[0]:
    st.subheader("Project Summary")
    st.markdown("""
**Business Requirements**
1) Identify features most correlated with **SalePrice**  
2) Predict **SalePrice** for 4 inherited houses and any new house  
**Success Metric:** R² ≥ 0.75 (train & test)
""")

with tabs[1]:
    st.subheader("Correlations / EDA")
    st.info("Correlation plots will appear here after notebook outputs are saved into `assets/`.")

with tabs[2]:
    st.subheader("Predict")
    st.warning("Model not found yet. After training, place `models/house_price_pipeline.joblib` here.")

with tabs[3]:
    st.subheader("Hypotheses")
    st.write("- H1: OverallQual ↑ ⇒ SalePrice ↑\n- H2: GrLivArea drives price\n- H3: Newer YearBuilt ⇒ higher price")

with tabs[4]:
    st.subheader("Technical / Performance")
    st.info("Pipeline steps & metrics will be shown here.")
