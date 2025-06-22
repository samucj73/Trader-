import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Caminho para as credenciais do Firebase (renomeie se necessário)
FIREBASE_CREDENTIAL_PATH = "firebase-adminsdk-fbsvc-2c717cb6fe.json"

# Nome da coleção do Firestore onde os dados serão salvos
FIREBASE_COLLECTION = "historico_roleta"

# Inicializa Firebase apenas uma vez
firebase_db = None
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIAL_PATH)
        firebase_admin.initialize_app(cred)
        firebase_db = firestore.client()
        print("[FIREBASE] Conectado com sucesso ao Firestore.")
    except Exception as e:
        print(f"[ERRO] Falha ao conectar com o Firebase: {e}")

def salvar_resultado_firebase(resultado: dict):
    """
    Salva um novo resultado no Firestore usando o timestamp como ID do documento.
    """
    try:
        if not firebase_db:
            raise RuntimeError("Firebase não inicializado.")

        timestamp = resultado.get("timestamp") or datetime.utcnow().isoformat()
        doc_id = timestamp.replace(":", "-").replace(".", "-").replace(" ", "_")
        doc_ref = firebase_db.collection(FIREBASE_COLLECTION).document(doc_id)

        if not doc_ref.get().exists:
            doc_ref.set(resultado)
            print(f"[FIREBASE] Resultado salvo com sucesso: {resultado}")
        else:
            print(f"[FIREBASE] Resultado já existe: {timestamp}")

    except Exception as e:
        print(f"[ERRO] Falha ao salvar no Firebase: {e}")

def carregar_historico_firebase() -> list:
    """
    Carrega todos os registros da coleção, ordenados por timestamp.
    """
    try:
        if not firebase_db:
            raise RuntimeError("Firebase não inicializado.")

        docs = firebase_db.collection(FIREBASE_COLLECTION).order_by("timestamp").stream()
        historico = []
        for doc in docs:
            data = doc.to_dict()
            if "number" in data:
                historico.append(data)

        print(f"[FIREBASE] {len(historico)} registros carregados do Firestore.")
        return historico

    except Exception as e:
        print(f"[ERRO] Falha ao carregar histórico do Firebase: {e}")
        return []
