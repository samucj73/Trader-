import requests
import logging
from fastapi import APIRouter, HTTPException
import os
import json

from firebase_integration import salvar_resultado_firebase  # 🔥 Integração Firebase

API_URL = "https://mute-grass-cc9b.samu-rcj.workers.dev/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
ARQUIVO_RESULTADOS = "historico_coluna_duzia.json"

router = APIRouter()

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

def salvar_resultado_em_arquivo(novo_resultado, caminho=ARQUIVO_RESULTADOS):
    try:
        dados_existentes = []

        print(f"[DEBUG] Salvando em: {os.path.abspath(caminho)}")

        if os.path.exists(caminho):
            with open(caminho, "r") as f:
                try:
                    dados_existentes = json.load(f)
                except json.JSONDecodeError:
                    logging.warning("[AVISO] Arquivo JSON vazio ou corrompido. Recriando...")
                    dados_existentes = []

        timestamps_existentes = {item.get("timestamp") for item in dados_existentes}

        if novo_resultado.get("timestamp") not in timestamps_existentes:
            dados_existentes.append(novo_resultado)
            dados_existentes.sort(key=lambda x: x["timestamp"])

            with open(caminho, "w") as f:
                json.dump(dados_existentes, f, indent=2)

            print(f"[OK] Novo resultado salvo localmente.")

            # 🔥 Salvar também no Firestore
            salvar_resultado_firebase(novo_resultado)

            return {"status": "novo resultado salvo", "resultado": novo_resultado}
        else:
            print(f"[INFO] Resultado repetido: {novo_resultado['timestamp']}")
            return {"status": "resultado já existe", "timestamp": novo_resultado['timestamp']}

    except Exception as e:
        logging.error(f"[ERRO] Falha ao salvar resultado: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro ao salvar resultado")

@router.get("/capturar-resultado")
def capturar_resultado():
    resultado = fetch_latest_result()
    if resultado is None:
        raise HTTPException(status_code=500, detail="Erro ao buscar resultado da API externa")
    resposta = salvar_resultado_em_arquivo(resultado)
    return resposta
