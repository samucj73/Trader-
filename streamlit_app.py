import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
import joblib
import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingClassifier

# === Firebase - usando chave embutida ===
@st.cache_resource
def init_firebase():
    firebase_key_json = {
        "type": "service_account",
        "project_id": "roleta-ia-duzia",
        "private_key_id": "2c717cb6feb5a9678d1d9a23d5349f55c7a6bb32",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCYOcX/+QLEzwNP\nNVTG5De91sBB1bLuzL9wkoYC6dRnFdyQsfyoQVUMr1G5tBYXrjlE50dbHxku4PHT\n7WQHELihcDQVhujNqWkhBfCsvFm1ZEPsof9lxYssycnJyg3Hd1vf0JF7N39tVmY4\nJRAWXbjDmlkFui1qsZ22Ugeo3OkbSPr6BHYdcTRm6MjeDDXGoaddITzIybcy1W6P\nsMv259kvO9UtmfzEHi+GqViBmbDCNHlwnNEqm9CBF+NZtAT4nJEB0LqpdJ6tElcn\n0upQilVxPYwmOS10KZwG+vspoNRbx/2N52OvwWMTcxbcjn7zhxGRFmEN0Equ08N3\n2o8DqWzXAgMBAAECggEAJmDyfY2/V/VFRVQR0q85KoCZKkABg9lVLsGSTeu8JF1L\npaNqKtWF5tPQy/wOUdwYDjotMFkP1VTbQ24neOyLUBBk4USJL0jV+4Bu5G38RBJK\nWb+iD3uVki4x1NE/VhnOrQf9lc4xL1TIIsGdY7YvXgltBlehGbQ8KfpWeglwL/N7\n2KUCPLXb1W72uoP9r3irD7y5tJYVdqzCyi3XfrHvJLMlKXSEOjP5G1qr7ZxAJOq5\ngM1408Te2mJ1MWYNn5rSX0m6+/DYQE/OYVaYXtyRxQgXbt4ixzqDJ/03OhNoAbpf\ne1ZXhaPpGmvQ1CGaWSJDd6cIDQaBERsOkWyK4v5Z6QKBgQDKcqhOMni+TKFBuLQ7\nzqXGUDmuoQ14OaLUWN6agG7AJeRt88Khs5aUTYaqkZQwZDOjncHIELT3yS+RSAEg\nx8QePvKWnoC/IXOUr0FZ0vKpjiPQ6E44qAvmpZ0uqxOqJbgz6XBwKc+czjI9De+R\nb4RWEhH51VZAkodHOibVvg/N3QKBgQDAfiyvfaWEEsSmvsvRkeA69/3UAsVe9LN4\neFlZOfbMrKeNs9+abAwKWnctSky2r1yotrg+Jbpe057jPveW5hULPz+XTe8VLoDE\n1tjmuHwBct4LBNCzPQtH3TdTMNdoofPxR4l71fZBkCaCBX4HDZl1uuBmmZjiv1nc\nfroORtX8QwKBgQCYvRq4LKImSSBcumLYwJcX4Q0z8HR+IVX1Sbtg68cjFzOZtRBB\n+YGBEGCqrb0VKXRAXFSIgfpW/CX0QVQAjFctzqYt9xYBndZa9kKi52GHhSMGiU4C\ntt6LvKWzQQVMGLs6B0R6i0EE+Bi1MZ9upak9WtLPICK8AxEnLvt/xa6czQKBgE24\nZBhzNcoOveHJdYfnS07j5FOezswZJwGELictlS1spLY9IxI5f98KScY6kqDMSzA/\nnkJRf19cOHHucY821NZWsjlIGTlHLmzLhoYZhNAc7fQq/IzyH8TjV7w6Iy82/MS1\nqpgaerRcHIAw8YKthgGX85TZfXZH5mN1s2+iQDH7AoGAMDqQNLjNEMXmfpAAsXkz\nQC4zCReFzLEDZkE9qdhxAYhJ9orU5Ono+hnRtEYgDT76sP0YfEZG/rCD7GgB2qDm\nhCGwxylHuadzX3A6v73dl9slEWcmVte9sZsoehTWbzCKocNgx68u7p9RJKKPSe6i\nUYB/RXw/SvJNPGNoz7juw9A=\n-----END PRIVATE KEY-----\n",
        "client_email": "firebase-adminsdk-fbsvc@roleta-ia-duzia.iam.gserviceaccount.com",
        "client_id": "100348573633888104841",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40roleta-ia-duzia.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }

    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key_json)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# === Carregar modelo ===
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_duzia.pkl")

# === App principal ===
st.title("Previsão de Dúzia - Roleta IA")

db = init_firebase()
modelo = carregar_modelo()

# Puxar últimos resultados do Firestore
colecao = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20)
docs = colecao.stream()

historico = []
for doc in docs:
    data = doc.to_dict()
    historico.append(data["number"])

if len(historico) < 10:
    st.warning("Ainda não há dados suficientes.")
else:
    entrada = pd.DataFrame([historico[:10]], columns=[f"n{i}" for i in range(10)])
    predicao = modelo.predict(entrada)[0]
    st.success(f"🔮 Previsão da próxima dúzia: **{predicao}**")

    # Exibir últimos números
    st.markdown("### Últimos números:")
    st.write(historico)
