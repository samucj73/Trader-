import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Subindo servidor na porta {port}...")
    uvicorn.run("backend_previsao_duzia:app", host="0.0.0.0", port=port)
