import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------- CONFIG (MUST BE FIRST) ----------------
st.set_page_config(
    page_title="ChurnIQ",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/final_churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ---------------- CSS ----------------
st.markdown("""
<style>
.hero {
    padding: 30px 0 20px 0;
    text-align: center;
}
.hero-title {
    font-size: 64px;
    font-weight: 800;
    background: linear-gradient(90deg, #8B5CF6, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 20px;
    color: #9CA3AF;
    margin-top: 12px;
}
.card {
    background: linear-gradient(145deg, #14141F, #0F0F18);
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 0 40px rgba(139,92,246,0.18);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
  <div class="hero-title">ChurnIQ</div>
  <div class="hero-sub">
    AI-powered behavioral intelligence to predict customer churn<br>
    <b>before it happens</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔮 Single Customer", "📂 Batch Customers"])

# ======================================================
# 🔮 TAB 1 — SINGLE CUSTOMER
# ======================================================
with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧩 Customer Signals")

        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 70)

        c1, c2 = st.columns(2)
        with c1:
            is_month_to_month = st.toggle("Month-to-month contract")
            fiber_risk_flag = st.toggle("Fiber internet")
        with c2:
            support_gap = st.toggle("No tech support")
            manual_payment_flag = st.toggle("Manual payment")

        service_complexity_score = st.slider("Service complexity", 0, 6, 2)

        predict = st.button("✨ Analyze churn risk", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if predict:
            input_df = pd.DataFrame(0, index=[0], columns=feature_names)
            input_df.loc[0, "tenure"] = tenure
            input_df.loc[0, "MonthlyCharges"] = monthly_charges
            input_df.loc[0, "is_month_to_month"] = int(is_month_to_month)
            input_df.loc[0, "fiber_risk_flag"] = int(fiber_risk_flag)
            input_df.loc[0, "support_gap"] = int(support_gap)
            input_df.loc[0, "manual_payment_flag"] = int(manual_payment_flag)
            input_df.loc[0, "service_complexity_score"] = service_complexity_score

            prob = model.predict_proba(input_df)[0][1]

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔮 Churn Risk Intelligence")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)

            if prob >= 0.4:
                st.error("⚠️ High churn risk")
            else:
                st.success("✅ Customer appears stable")

            st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# 📂 TAB 2 — BATCH CUSTOMERS
# ======================================================
with tab2:
    st.subheader("📂 Batch Churn Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV with engineered customer features",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        X_batch = df[feature_names]
        df["churn_probability"] = model.predict_proba(X_batch)[:, 1]

        def risk_bucket(p):
            if p < 0.4:
                return "Low Risk"
            elif p < 0.7:
                return "Medium Risk"
            else:
                return "High Risk"

        df["risk_bucket"] = df["churn_probability"].apply(risk_bucket)

        st.success("✅ Batch predictions completed")
        st.dataframe(df.head(20))

        risk_counts = df["risk_bucket"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Customers"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Customers",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )
