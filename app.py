import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
FEATURES = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "MonthlyCharges",
    "Contract_Risk",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "PaymentMethod_Automatic",
]

INTERNET_MAP = {
    "No Internet": 0,
    "DSL": 1,
    "Fiber Optic": 2,
}

CONTRACT_MAP = {
    # Based on churn-rate ordering in your final CSV:
    # 0 lowest churn, 2 highest churn
    "Two year (Low risk)": 0,
    "One year (Medium risk)": 1,
    "Month-to-month (High risk)": 2,
}

YESNO_MAP = {"No": 0, "Yes": 1}

PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Automatic (Bank transfer / Credit card)",
]


def build_input_row(
    SeniorCitizen: int,
    Partner: int,
    Dependents: int,
    tenure: int,
    InternetService: int,
    OnlineSecurity: int,
    TechSupport: int,
    PaperlessBilling: int,
    MonthlyCharges: float,
    Contract_Risk: int,
    payment_method: str,
) -> pd.DataFrame:
    row = {f: 0 for f in FEATURES}

    row["SeniorCitizen"] = SeniorCitizen
    row["Partner"] = Partner
    row["Dependents"] = Dependents
    row["tenure"] = tenure
    row["InternetService"] = InternetService
    row["OnlineSecurity"] = OnlineSecurity
    row["TechSupport"] = TechSupport
    row["PaperlessBilling"] = PaperlessBilling
    row["MonthlyCharges"] = MonthlyCharges
    row["Contract_Risk"] = Contract_Risk

    # one-hot payment method (your engineered format)
    if payment_method == "Electronic check":
        row["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        row["PaymentMethod_Mailed check"] = 1
    elif payment_method == "Automatic (Bank transfer / Credit card)":
        row["PaymentMethod_Automatic"] = 1

    return pd.DataFrame([row], columns=FEATURES)


def soft_vote_probability(xgb_model, rf_model, ensemble_cfg, X: pd.DataFrame) -> float:
    w_xgb = float(ensemble_cfg["weights"]["xgb"])
    w_rf = float(ensemble_cfg["weights"]["rf"])
    p_xgb = float(xgb_model.predict_proba(X)[:, 1][0])
    p_rf = float(rf_model.predict_proba(X)[:, 1][0])
    return w_xgb * p_xgb + w_rf * p_rf


def predict_label(prob: float, threshold: float) -> int:
    return int(prob >= threshold)


def friendly_reasoning_from_contribs(contribs: pd.Series, top_k: int = 3) -> dict:
    """
    contribs: signed contributions (positive => pushes toward churn, negative => pushes away)
    Returns a dict with top positive/negative reasons and a plain-English paragraph.
    """
    contribs = contribs.sort_values()

    top_decrease = contribs.head(top_k)          # most negative
    top_increase = contribs.tail(top_k)[::-1]    # most positive

    def feature_to_friendly_name(f: str) -> str:
        # Make names nicer for normal users
        nice = {
            "SeniorCitizen": "Senior citizen",
            "Partner": "Has a partner",
            "Dependents": "Has dependents",
            "tenure": "Tenure (months with company)",
            "InternetService": "Internet service type",
            "OnlineSecurity": "Online security",
            "TechSupport": "Tech support",
            "PaperlessBilling": "Paperless billing",
            "MonthlyCharges": "Monthly charges",
            "Contract_Risk": "Contract risk level",
            "PaymentMethod_Electronic check": "Payment method: Electronic check",
            "PaymentMethod_Mailed check": "Payment method: Mailed check",
            "PaymentMethod_Automatic": "Payment method: Automatic",
        }
        return nice.get(f, f)

    inc_list = [(feature_to_friendly_name(i), float(top_increase[i])) for i in top_increase.index]
    dec_list = [(feature_to_friendly_name(i), float(top_decrease[i])) for i in top_decrease.index]

    # Build a plain-English explanation
    def sentence_from_feature(fname: str, val: float) -> str:
        # Keep it simple; we don’t expose SHAP jargon
        if val > 0:
            return f"**{fname}** is increasing churn risk."
        else:
            return f"**{fname}** is reducing churn risk."

    expl_lines = []
    expl_lines.append("Here are the strongest drivers behind this prediction:")
    expl_lines.append("")
    expl_lines.append("**Increasing churn risk:**")
    for n, v in inc_list:
        expl_lines.append(f"- {sentence_from_feature(n, v)}")
    expl_lines.append("")
    expl_lines.append("**Reducing churn risk:**")
    for n, v in dec_list:
        expl_lines.append(f"- {sentence_from_feature(n, v)}")

    return {
        "top_increase": inc_list,
        "top_decrease": dec_list,
        "explanation_md": "\n".join(expl_lines),
    }


@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load("xgb_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    ensemble_cfg = joblib.load("ensemble_config.pkl")

    # background data for SHAP (use your final engineered CSV)
    df = pd.read_csv("Telco-Customer-Churn-Final.csv")
    X_bg = df.drop(columns=["Churn"], errors="ignore")[FEATURES].copy()

    # Use a small sample to speed SHAP
    if len(X_bg) > 500:
        X_bg_sample = X_bg.sample(500, random_state=42)
    else:
        X_bg_sample = X_bg

    # TreeExplainers are fast & good for XGB/RF
    expl_xgb = shap.TreeExplainer(xgb_model, X_bg_sample)
    expl_rf = shap.TreeExplainer(rf_model, X_bg_sample)

    return xgb_model, rf_model, ensemble_cfg, X_bg_sample, expl_xgb, expl_rf


def compute_weighted_shap_contribs(expl_xgb, expl_rf, ensemble_cfg, X_one: pd.DataFrame) -> pd.Series:
    """
    We compute SHAP values for each model separately, then weight them
    similarly to the ensemble and return a single contribution per feature.
    """
    w_xgb = float(ensemble_cfg["weights"]["xgb"])
    w_rf = float(ensemble_cfg["weights"]["rf"])

    # SHAP output shape differs across versions:
    # sometimes returns list [class0, class1], sometimes returns array.
    def shap_for_class1(explainer, X: pd.DataFrame) -> np.ndarray:
        """
        Returns a 1D array of length n_features for the positive class (class 1).
        Handles SHAP outputs across versions/models:
        - list of [class0, class1] arrays
        - (n_samples, n_features)
        - (n_samples, n_features, n_classes)
        - (n_features, n_classes)  (some edge cases)
        """
        sv = explainer.shap_values(X)

        # Case 1: SHAP returns list per class
        if isinstance(sv, list):
            arr = np.asarray(sv[1])  # class 1
        else:
            arr = np.asarray(sv)

        # Now normalize to (n_features,)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            return arr[0, :, 1]
        if arr.ndim == 2:
            # could be (n_samples, n_features) OR (n_features, n_classes)
            if arr.shape[0] == 1 and arr.shape[1] == X.shape[1]:
                return arr[0]  # (1, n_features)
            if arr.shape[0] == X.shape[1] and arr.shape[1] == 2:
                return arr[:, 1]  # (n_features, 2)
            # fallback: assume first row is the sample
            return arr[0]
        if arr.ndim == 1:
            return arr

        raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


    shap_xgb = shap_for_class1(expl_xgb, X_one)
    shap_rf = shap_for_class1(expl_rf, X_one)

    weighted = w_xgb * np.array(shap_xgb) + w_rf * np.array(shap_rf)
    return pd.Series(weighted, index=X_one.columns).sort_values()


st.title("📉 Customer Churn Predictor")
st.write("Enter customer information to estimate churn probability and see easy-to-understand reasons.")


xgb_model, rf_model, ensemble_cfg, X_bg_sample, expl_xgb, expl_rf = load_artifacts()
threshold = float(ensemble_cfg.get("threshold", 0.5))


st.divider()
st.subheader("Customer Details")

SeniorCitizen = st.selectbox("Senior Citizen?", ["No", "Yes"], index=0)
Partner = st.selectbox("Has Partner?", ["No", "Yes"], index=0)
Dependents = st.selectbox("Has Dependents?", ["No", "Yes"], index=0)

tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

InternetService = st.selectbox("Internet Service", list(INTERNET_MAP.keys()), index=2)
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"], index=0)
TechSupport = st.selectbox("Tech Support", ["No", "Yes"], index=0)

PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)

MonthlyCharges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0
)

Contract_Risk = st.selectbox("Contract Type", list(CONTRACT_MAP.keys()), index=2)
payment_method = st.selectbox("Payment Method", PAYMENT_OPTIONS, index=2)

predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

st.divider()
st.subheader("Prediction")

if predict_btn:
    X_one = build_input_row(
        SeniorCitizen=YESNO_MAP[SeniorCitizen],
        Partner=YESNO_MAP[Partner],
        Dependents=YESNO_MAP[Dependents],
        tenure=int(tenure),
        InternetService=INTERNET_MAP[InternetService],
        OnlineSecurity=YESNO_MAP[OnlineSecurity],
        TechSupport=YESNO_MAP[TechSupport],
        PaperlessBilling=YESNO_MAP[PaperlessBilling],
        MonthlyCharges=float(MonthlyCharges),
        Contract_Risk=CONTRACT_MAP[Contract_Risk],
        payment_method=payment_method,
    )

    y_prob = soft_vote_probability(xgb_model, rf_model, ensemble_cfg, X_one)
    pred = int(y_prob >= threshold)
    confidence = float(np.maximum(y_prob, 1 - y_prob))

    if pred == 1:
        st.error("⚠️ **Customer WILL churn**")
    else:
        st.success("✅ **Customer will NOT churn**")

    st.metric("Model confidence", f"{confidence*100:.2f}%")
    st.caption("Confidence reflects how sure the model is about this outcome.")

    st.divider()
    st.subheader("Why this prediction? (Simple Explanation)")

    contribs = compute_weighted_shap_contribs(
        expl_xgb, expl_rf, ensemble_cfg, X_one
    )
    reasoning = friendly_reasoning_from_contribs(contribs, top_k=3)
    st.markdown(reasoning["explanation_md"])

    st.subheader("Main factors (visual)")
    topN = 8
    top = contribs.reindex(
        contribs.abs().sort_values(ascending=False).head(topN).index
    )
    fig, ax = plt.subplots(figsize=(5, 3))  # width, height in inches
    ax.barh(top.index, top.values)
    ax.set_xlabel("Impact on churn risk")
    ax.set_ylabel("Feature")

    st.pyplot(fig, use_container_width=False)


    with st.expander("Show input data used for prediction"):
        st.dataframe(X_one)
else:
    st.info("Fill the form and click **Predict Churn**.")
