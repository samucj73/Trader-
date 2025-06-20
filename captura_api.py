import requests
import logging
from fastapi import APIRouter, HTTPException
import os
import json

API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
ARQUIVO_RESULTADOS = "historico_resultados.json"

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

        return {
            "number": number,
            "color": color,
            "timestamp": timestamp,
            "lucky_numbers": lucky_numbers
        }
    except Exception as e:
        logging.error(f"Erro ao buscar resultado da API: {e}")
        return None

def salvar_resultado_em_arquivo(novo_resultado, caminho=ARQUIVO_RESULTADOS):
    dados_existentes = []

    if os.path.exists(caminho):
        with open(caminho, "r") as f:
            try:
                dados_existentes = json.load(f)
            except json.JSONDecodeError:
                logging.warning("Arquivo JSON vazio ou corrompido. Recriando arquivo.")
                dados_existentes = []

    timestamps_existentes = {item['timestamp'] for item in dados_existentes}

    if novo_resultado['timestamp'] not in timestamps_existentes:
        dados_existentes.append(novo_resultado)
        dados_existentes.sort(key=lambda x: x['timestamp'])
        with open(caminho, "w") as f:
            json.dump(dados_existentes, f, indent=2)
        return {"status": "novo resultado salvo", "resultado": novo_resultado}
    else:
        return {"status": "resultado já existe", "timestamp": novo_resultado['timestamp']}

@router.get("/capturar-resultado")
def capturar_resultado():
    resultado = fetch_latest_result()
    if resultado is None:
        raise HTTPException(status_code=500, detail="Erro ao buscar resultado da API externa")
    resposta = salvar_resultado_em_arquivo(resultado)
    return resposta
