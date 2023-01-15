import pathlib
from fastapi import FastAPI
from typing import Optional
from fastapi.responses import StreamingResponse
from cassandra.cqlengine.management import sync_table
from cassandra.query import SimpleStatement

from . import (
    ml,
    config,
    models,
    db)

app = FastAPI()
settings = config.Settings()

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent /"models"
SMS_SPAM_MODEL_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_DIR / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_DIR / "spam-classifer-metadata.json"

AI_MODEL = None
DB_SESSION = None
SMSInference = models.SMSInferece


@app.on_event("startup")
def on_startup():
    global AI_MODEL, DB_SESSION
    AI_MODEL = ml.AIModel(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH
    )
    DB_SESSION = db.get_session()
    print("DB_SESSION:", DB_SESSION)
    sync_table(SMSInference)

@app.get("/")
def read_index(q:Optional[str] = None):
    global AI_MODEL
    query = q or "HELO WORLD"
    prediction = AI_MODEL.predict_text(query)
    top = prediction.get('top')
    data = {"query":query, **top}
    print(data)
    obj = SMSInference.objects.create(**data)
    return obj
    # return {"query":query, "results":prediction}

@app.get("/inferences") #/?q=this is awsome
def list_inferences():
    q = SMSInference.objects.all()
    print(q)
    return list(q)

@app.get("/inferences/{my_uuid}") #/?qthis is awsome
def read_index(my_uuid):
    obj = SMSInference.objects.get(uuid=my_uuid)
    return obj

def fetch_rows(
    stmt:SimpleStatement,
    fetch_size:int=25,
    session=None):
    stmt.fetch_size = fetch_size
    result_set = session.execute(stmt)
    has_pages = result_set.has_more_pages
    yield "uuid, label, confidence, query\n"
    while has_pages:
        for row in result_set.current_rows:
            yield f"{row['uuid']},{row['label']}, {row['confidence']}, {row['qurey']}\n"
        has_pages = result_set.has_more_pages
        result_set = session.execute(stmt, paging_state=result_set.paging_state)

@app.get("/dataset")
def export_inferences():
    global DB_SESSION
    cql_query = "SELECT * FROM spam_inferences.smsinferece LIMIT 10000"
    # rows = DB_SESSION.execute(cql_query)
    statement = SimpleStatement(cql_query)
    return StreamingResponse(fetch_rows(statement, 25, DB_SESSION))

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

