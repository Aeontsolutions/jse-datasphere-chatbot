import os
import sys
from pathlib import Path

# Ensure the internal `app` package (fastapi_app/app) is discoverable even when
# this file is executed from outside the `fastapi_app` directory.
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))           # adds fastapi_app/
sys.path.insert(0, str(BASE_DIR / "app"))  # adds fastapi_app/app/
from app.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
