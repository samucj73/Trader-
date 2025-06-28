import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import joblib

# Inicialização Firebase
@st.cache_resource
def init_firebase():
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Carrega o modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_duzia.pkl")

def classificar_duzia(n):
    if n == 0: return 0
    if 1 <= n <= 12: return 1
    if 13 <= n <= 24: return 2
    return 3

def prever_duzia(modelo, ultimos):
    x = np.array(ultimos[-5:]).reshape(1, -1)
    return modelo.predict(x)[0]

st.title("🎰 Previsão da Dúzia na Roleta")

db = init_firebase()
model = load_model()

# Busca últimos resultados do Firestore
docs = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
resultados = [d.to_dict() for d in docs]
resultados.sort(key=lambda r: r["timestamp"])

nums = [r["number"] for r in resultados]
st.write("Últimos números:", nums)

if len(nums) >= 5:
    pred = prever_duzia(model, nums)
    st.subheader(f"🔮 Próxima dúzia prevista: **{pred}**")

    real = classificar_duzia(nums[-1])
    if pred == real:
        st.success(f"✅ Acertou! O número {nums[-1]} pertence à dúzia {real}.")
    else:
        st.error(f"❌ Errou! O número {nums[-1]} pertence à dúzia {real}.")

    # Simula acertos anteriores
    acertos = sum(
        prever_duzia(model, nums[i-5:i]) == classificar_duzia(nums[i])
        for i in range(5, len(nums))
    )
    st.info(f"📊 Acertos simulados: {acertos} de {len(nums)-5}")
else:
    st.warning("Aguardando mais dados para previsão.")
