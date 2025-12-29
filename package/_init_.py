from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

class PlayReq(BaseModel):
    path: str

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/play")
def play(req: PlayReq):
    os.startfile(req.path)  # 用 Windows 預設播放器/瀏覽器開啟
    return {"ok": True, "playing": req.path}
