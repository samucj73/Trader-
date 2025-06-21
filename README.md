# Backend da Previsão de Dúzia - IA Roleta

Este projeto expõe a previsão da dúzia da sua IA de roleta via API para integração com PWA ou outros sistemas.

## Como usar no Render.com

1. Crie uma conta gratuita em https://render.com
2. Clique em "New Web Service"
3. Conecte seu GitHub e selecione este repositório
4. Use as seguintes configurações:

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn backend_previsao_duzia:app --host=0.0.0.0 --port=$PORT`
- Runtime: Python 3.10+

5. Após deploy, acesse `https://SEUAPP.onrender.com/previsao-duzia` para obter a previsão da dúzia da IA.

O arquivo `historico_coluna_duzia.json` precisa estar presente no repositório (com alguns dados).

## Endpoint

- `GET /previsao-duzia` → retorna a previsão mais provável de dúzia

## Exemplo de resposta

```json
{ "duzia_prevista": 2 }
```