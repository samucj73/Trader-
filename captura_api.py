import requests
import logging
from fastapi import APIRouter, HTTPException
import os
import json

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# === Firebase Setup ===
FIREBASE_CRED_PATH = "firebase-adminsdk-fbsvc-2c717cb6fe.json"
FIREBASE_COLLECTION = "resultados_duzia"
firebase_db = None

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        firebase_db = firestore.client()
        print("[FIREBASE] Firebase conectado com sucesso.")
except Exception as e:
    print(f"[ERRO] Erro ao conectar ao Firebase: {e}")

API_URL = "https://mute-grass-cc9b.samu-rcj.workers.dev/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ARQUIVO_LOCAL = "historico_coluna_duzia.json"

router = APIRouter()

# === Captura do número da API externa ===
def fetch_latest_result():
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        game_data = data.get("data", {})
        result = game_data.get("result", {})
        outcome = result.get("outcome", {})
        lucky_list = result.get("luckyNumbersList", [])

        number = outcome.get("number")
        color = outcome.get("color", "-")
        timestamp = game_data.get("startedAt")
        lucky_numbers = [item["number"] for item in lucky_list]

        print(f"[API] Número capturado: {number} | Timestamp: {timestamp}")

        return {
            "number": number,
            "color": color,
            "timestamp": timestamp,
            "lucky_numbers": lucky_numbers
        }
    except Exception as e:
        logging.error(f"[ERRO] Erro ao buscar resultado da API: {e}")
        return None

# === Salvar no Firebase ou JSON local ===
def salvar_resultado_em_arquivo(novo_resultado, caminho=ARQUIVO_LOCAL):
    try:
        # === Se Firebase ativo, salvar lá ===
        if firebase_db:
            doc_id = novo_resultado["timestamp"]
            doc_ref = firebase_db.collection(FIREBASE_COLLECTION).document(doc_id)
            if not doc_ref.get().exists:
                doc_ref.set(novo_resultado)
                print("[FIREBASE] Novo resultado salvo no Firestore.")
                return {"status": "salvo_firebase", "resultado": novo_resultado}
            else:
                print(f"[INFO] Resultado já existe no Firebase: {doc_id}")
                return {"status": "duplicado_firebase", "timestamp": doc_id}

        # === Fallback: salvar em arquivo local ===
        dados_existentes = []
        if os.path.exists(caminho):
            with open(caminho, "r") as f:
                try:
                    dados_existentes = json.load(f)
                except json.JSONDecodeError:
                    logging.warning("[AVISO] JSON vazio ou corrompido, recriando.")
                    dados_existentes = []

        timestamps_existentes = {item.get("timestamp") for item in dados_existentes}

        if novo_resultado.get("timestamp") not in timestamps_existentes:
            dados_existentes.append(novo_resultado)
            dados_existentes.sort(key=lambda x: x["timestamp"])
            with open(caminho, "w") as f:
                json.dump(dados_existentes, f, indent=2)
            print("[LOCAL] Resultado salvo em JSON.")
            return {"status": "salvo_local", "resultado": novo_resultado}
        else:
            print("[INFO] Resultado repetido no JSON local.")
            return {"status": "duplicado_local", "timestamp": novo_resultado["timestamp"]}

    except Exception as e:
        logging.error(f"[ERRO] Falha ao salvar resultado: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao salvar resultado")

# === Rota de captura manual ===
@router.get("/capturar-resultado")
def capturar_resultado():
    resultado = fetch_latest_result()
    if resultado is None:
        raise HTTPException(status_code=500, detail="Erro ao buscar resultado da API externa")
    resposta = salvar_resultado_em_arquivo(resultado)
    return resposta
