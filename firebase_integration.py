import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Caminho do seu arquivo de credenciais Firebase (baixe pelo console do Firebase)
FIREBASE_CREDENTIAL_PATH = "firebase_credentials.json"

# Nome da coleção onde será salvo o histórico
FIREBASE_COLLECTION = "historico_roleta"

# Inicializa Firebase se ainda não estiver inicializado
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIAL_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

def salvar_resultado_firebase(resultado: dict):
    """
    Salva um novo resultado no Firestore, usando o timestamp como ID.
    """
    try:
        timestamp = resultado.get("timestamp") or datetime.utcnow().isoformat()
        doc_id = timestamp.replace(":", "-").replace(".", "-")
        db.collection(FIREBASE_COLLECTION).document(doc_id).set(resultado)
        print(f"[FIREBASE] Resultado salvo: {resultado}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar no Firebase: {e}")

def carregar_historico_firebase() -> list:
    """
    Retorna todos os resultados da coleção do Firestore, ordenados por timestamp.
    """
    try:
        docs = db.collection(FIREBASE_COLLECTION).order_by("timestamp").stream()
        historico = []
        for doc in docs:
            data = doc.to_dict()
            if "number" in data:
                historico.append(data)
        print(f"[FIREBASE] {len(historico)} registros carregados.")
        return historico
    except Exception as e:
        print(f"[ERRO] Falha ao carregar do Firebase: {e}")
        return []
