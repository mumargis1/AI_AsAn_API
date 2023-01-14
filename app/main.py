import pathlib
from fastapi import FastAPI
from typing import Optional

from . import (ml, config)

app = FastAPI()
settings = config.Settings()

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent /"models"
SMS_SPAM_MODEL_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_DIR / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-metadata.json"

AI_MODEL = None

@app.on_event("startup")
def on_startup():
    global AI_MODEL
    AI_MODEL = ml.AIModel(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH
    )

@app.get("/")
def read_index(q:Optional[str] = None):
    global AI_MODEL
    query = q or "HELO WORLD"
    prediction = AI_MODEL.predict_text(query)
    return {"query": query,
            "results":prediction,
            "db_client_id":settings.db_client_id}

# @app.on_event("startup")
# def on_startup():
#     global AI_MODEL, AI_TOKNIZER, MODEL_METADATA, labels_legend_inverted
#     if MODEL_PATH.exists():
#         AI_MODEL = load_model(MODEL_PATH)
#     if TOKENIZER_PATH.exists():
#         t_json = TOKENIZER_PATH.read_text()
#         AI_TOKNIZER = tokenizer_from_json(t_json)
#     if METADATA_PATH.exists():
#         MODEL_METADATA = json.loads(METADATA_PATH.read_text())
#         labels_legend_inverted = MODEL_METADATA['labels_legend_inverted']

# def predict(query):
#     '''sequences
#     pad_sequences
#     model.predict
#     convert to labels'''
#     # global AI_MODEL

#     sequences = AI_TOKNIZER.texts_to_sequences([query])
#     maxlen = MODEL_METADATA.get('max_sequence') or 280
#     x_input = pad_sequences(sequences, maxlen=maxlen)
#     print(x_input.shape)
#     preds_array = AI_MODEL.predict(x_input)
#     preds = preds_array[0]
#     top_idx_val = np.argmax(preds)
#     print(top_idx_val)
#     top_pred = {"label": labels_legend_inverted[str(top_idx_val)],
#             "confidence": float(preds[top_idx_val])}
#     labeled_preds = [{"label": labels_legend_inverted[str(i)],
#                 "confidence": float(v)} for i, v in enumerate(list(preds))]
#     return json.loads(json.dumps({"top": top_pred, "predictions":labeled_preds},
#                                 cls=NumpyEncoder))

