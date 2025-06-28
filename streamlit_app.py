import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import joblib
import json

# Inicializa Firebase usando secrets do Streamlit
@st.cache_resource
def init_firebase():
    cred_dict = json.loads(st.secrets["firebase_key"])
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Carrega modelo de IA
@st.cache_resource
def load_model():
    return joblib.load("modelo_duzia.pkl")

# Classifica número em sua dúzia
def classificar_duzia(n):
    if n == 0:
        return 0
    elif 1 <= n <= 12:
        return 1
    elif 13 <= n <= 24:
        return 2
    else:
        return 3

# Realiza a previsão com os últimos 5 números
def prever_duzia(modelo, ultimos_numeros):
    entrada = np.array(ultimos_numeros[-5:]).reshape(1, -1)
    return modelo.predict(entrada)[0]

# Interface do app
st.set_page_config(page_title="Previsão de Dúzia", layout="centered")
st.title("🎰 Previsão de Dúzia na Roleta")

# Inicializa Firebase e carrega modelo
db = init_firebase()
modelo = load_model()

# Obtém os últimos resultados do Firestore
docs = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
resultados = [doc.to_dict() for doc in docs]
resultados.sort(key=lambda x: x["timestamp"])

# Extrai apenas os números
numeros = [r["number"] for r in resultados]
st.write("🎯 Últimos números:", numeros)

# Faz a previsão com base nos últimos 5 números
if len(numeros) >= 5:
    previsao = prever_duzia(modelo, numeros)
    st.subheader(f"🔮 Próxima dúzia prevista: **{previsao}**")

    # Compara com o último número real
    ultimo_num = numeros[-1]
    duzia_real = classificar_duzia(ultimo_num)

    if previsao == duzia_real:
        st.success(f"✅ Acertou! O número {ultimo_num} pertence à dúzia {duzia_real}.")
    else:
        st.error(f"❌ Errou! O número {ultimo_num} pertence à dúzia {duzia_real}.")

    # Simula acertos nas previsões anteriores
    acertos = sum(
        prever_duzia(modelo, numeros[i-5:i]) == classificar_duzia(numeros[i])
        for i in range(5, len(numeros))
    )
    st.info(f"📊 Acertos simulados: **{acertos} de {len(numeros) - 5} previsões**")

else:
    st.warning("🔄 Aguardando mais resultados para gerar previsões.")
