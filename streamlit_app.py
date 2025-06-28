import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import joblib
import json

@st.cache_resource
def init_firebase():
    key_secret = st.secrets["firebase_key"]
    if isinstance(key_secret, str):
        cred_dict = json.loads(key_secret)
    else:
        cred_dict = key_secret
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    return firestore.client()

@st.cache_resource
def load_model():
    return joblib.load("modelo_duzia.pkl")

def classificar_duzia(n):
    if n == 0:
        return 0
    elif 1 <= n <= 12:
        return 1
    elif 13 <= n <= 24:
        return 2
    else:
        return 3

def prever_duzia(modelo, ultimos_numeros):
    entrada = np.array(ultimos_numeros[-5:]).reshape(1, -1)
    return modelo.predict(entrada)[0]

st.set_page_config(page_title="Previsão de Dúzia", layout="centered")
st.title("🎰 Previsão de Dúzia na Roleta")

db = init_firebase()
modelo = load_model()

docs = db.collection("resultados") \
         .order_by("timestamp", direction=firestore.Query.DESCENDING) \
         .limit(20) \
         .stream()
resultados = [doc.to_dict() for doc in docs]
resultados.sort(key=lambda x: x["timestamp"])
numeros = [r["number"] for r in resultados]

st.write("🎯 Últimos números:", numeros)

if len(numeros) >= 5:
    previsao = prever_duzia(modelo, numeros)
    st.subheader(f"🔮 Próxima dúzia prevista: **{previsao}**")

    ultimo = numeros[-1]
    real = classificar_duzia(ultimo)
    if previsao == real:
        st.success(f"✅ Acertou! O número {ultimo} é da dúzia {real}.")
    else:
        st.error(f"❌ Errou! O número {ultimo} é da dúzia {real}.")

    acertos = sum(
        prever_duzia(modelo, numeros[i-5:i]) == classificar_duzia(numeros[i])
        for i in range(5, len(numeros))
    )
    st.info(f"📊 Acertos simulados: {acertos} de {len(numeros) - 5} previsões")
else:
    st.warning("🔄 Aguardando mais resultados para gerar previsões.")
