import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error

# =====================
# KONFIGURASI HALAMAN
# =====================
st.set_page_config(
    page_title="Customer Classification & Regression App",
    layout="centered"
)

st.title("📊 Customer Classification & Regression App")
st.markdown(
    "Aplikasi ini menggunakan **Ensemble Method (Random Forest)** "
    "untuk melakukan **klasifikasi subscription** dan "
    "**regresi churn risk pelanggan**."
)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_customers_cleaned (1).csv")

df = load_data()

features = ["age", "income", "credit_score", "total_spent"]

# =====================
# INPUT USER
# =====================
st.header("🧑 Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Umur", 18, 80, 30)
    income = st.number_input("Pendapatan", 0, 100_000_000, 5_000_000)

with col2:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    total_spent = st.number_input("Total Pengeluaran", 0, 50_000_000, 1_000_000)

input_df = pd.DataFrame([{
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "total_spent": total_spent
}])

# =====================
# PROSES MODEL
# =====================
st.header("⚙️ Proses Model")

if st.button("🚀 Jalankan Model"):
    with st.spinner("Melatih model, mohon tunggu..."):

        X = df[features]

        # ===== KLASIFIKASI =====
        y_class = df["subscription"]
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(
            n_estimators=10,
            random_state=42
        )
        clf.fit(Xc_train, yc_train)

        # ===== REGRESI =====
        y_reg = df["churn_risk"]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

        reg = RandomForestRegressor(
            n_estimators=10,
            random_state=42
        )
        reg.fit(Xr_train, yr_train)

    st.success("Model berhasil dijalankan!")

    # =====================
    # HASIL PREDIKSI
    # =====================
    st.subheader("📌 Hasil Prediksi")

    # --- Klasifikasi ---
    status = clf.predict(input_df)[0]
    prob = clf.predict_proba(input_df)[0][1]

    if status == 1:
        st.success(f"Status Subscription: **BERLANGGANAN** (Probabilitas: {prob:.2f})")
    else:
        st.warning(f"Status Subscription: **TIDAK BERLANGGANAN** (Probabilitas: {prob:.2f})")

    # --- Regresi ---
    pred_churn = reg.predict(input_df)[0]

    # Interpretasi tingkat risiko
    if pred_churn < 0.34:
        risk_level = "🟢 Rendah"
        risk_desc = "Pelanggan memiliki risiko churn yang rendah."
    elif pred_churn < 0.67:
        risk_level = "🟡 Sedang"
        risk_desc = "Pelanggan memiliki risiko churn sedang dan perlu perhatian."
    else:
        risk_level = "🔴 Tinggi"
        risk_desc = "Pelanggan memiliki risiko churn tinggi dan perlu tindakan segera."

    st.info(f"Prediksi Churn Risk: **{pred_churn:.2f}**")
    st.write("Tingkat Risiko Churn:", risk_level)
    st.write(risk_desc)

    # =====================
    # EVALUASI MODEL
    # =====================
    st.header("📈 Evaluasi Model")

    # --- Confusion Matrix ---
    cm = confusion_matrix(yc_test, clf.predict(Xc_test))
    fig1, ax1 = plt.subplots()
    ax1.imshow(cm)
    ax1.set_title("Confusion Matrix - Subscription")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    st.write("Accuracy (Klasifikasi):", accuracy_score(yc_test, clf.predict(Xc_test)))

    # --- Evaluasi Regresi ---
    y_pred_r = reg.predict(Xr_test)
    st.write("MAE (Regresi Churn Risk):", mean_absolute_error(yr_test, y_pred_r))

    # --- Visualisasi Regresi ---
    st.subheader("📉 Visualisasi Regresi Churn Risk")

    fig2, ax2 = plt.subplots()
    ax2.scatter(yr_test, y_pred_r, alpha=0.6)
    ax2.plot(
        [yr_test.min(), yr_test.max()],
        [yr_test.min(), yr_test.max()],
        linestyle="--"
    )
    ax2.set_xlabel("Actual Churn Risk")
    ax2.set_ylabel("Predicted Churn Risk")
    ax2.set_title("Actual vs Predicted Churn Risk")
    st.pyplot(fig2)

# =====================
# FOOTER
# =====================
st.markdown("---")
st.caption(
    "Customer ML App | Ensemble Method (Random Forest) | Streamlit Cloud"
)
