import streamlit as st
import json
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Previsão Roleta IA", layout="centered")

# === Inicializa Firebase ===
@st.cache_resource
def init_firebase():
    key_json_str = st.secrets["firebase_key_json"]
    if isinstance(key_json_str, str):
        key_data = json.loads(key_json_str)
    else:
        key_data = key_json_str

    # Corrige quebra de linha da chave privada
    if "\\n" in key_data["private_key"]:
        key_data["private_key"] = key_data["private_key"].replace("\\n", "\n")

    cred = credentials.Certificate(key_data)
    firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

# === Carrega modelo IA ===
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_duzia.pkl")

modelo = carregar_modelo()

# === Função para processar histórico ===
def extrair_features(entrada):
    ultimos_numeros = [item["number"] for item in entrada[-6:]]
    return np.array(ultimos_numeros).reshape(1, -1)

# === Interface Streamlit ===
st.title("🎯 Previsão de Dúzia - Roleta IA")

# Obtém histórico do Firestore
docs = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(6).stream()
historico = [doc.to_dict() for doc in docs]

if len(historico) < 6:
    st.warning("⚠️ Aguardando mais resultados para prever...")
else:
    entrada = list(reversed(historico))  # ordem cronológica
    features = extrair_features(entrada)

    previsao = modelo.predict(features)[0]
    st.subheader(f"🎲 Próxima Dúzia Prevista: **{previsao}**")

    # Último número sorteado
    ultimo = entrada[-1]["number"]
    if 1 <= ultimo <= 12:
        duzia_real = 1
    elif 13 <= ultimo <= 24:
        duzia_real = 2
    else:
        duzia_real = 3

    if duzia_real == previsao:
        st.success(f"✅ Acertou! Último número foi {ultimo} (dúzia {duzia_real})")
    else:
        st.error(f"❌ Errou. Último número foi {ultimo} (dúzia {duzia_real})")

# Footer
st.markdown("---")
st.caption("Desenvolvido com ❤️ por Sam Rock • IA com Streamlit + Firebase")
