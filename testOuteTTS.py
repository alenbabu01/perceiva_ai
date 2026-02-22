from fastapi import FastAPI,Body
from kokoro import KPipeline
import soundfile as sf
import numpy as np

app = FastAPI()

pipeline = KPipeline(lang_code="a")


@app.post("/tts")
def tts(text: str = Body(..., embed = True)):
    generator = pipeline(text, voice="af_heart")

    chunks = []
    for _, _, audio in generator:
        chunks.append(audio)

    if not chunks:
        return {"ok": False, "error": "No audio generated"}

    full_audio = np.concatenate(chunks)
    out_path = "kokoro_output.wav"
    sf.write(out_path, full_audio, 24000)
    return {"ok": True, "path": out_path}