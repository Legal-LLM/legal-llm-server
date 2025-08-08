from app.api import create_app

# Uvicorn entrypoint expects a global named `app`
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
