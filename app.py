import email
from email.policy import default
from pathlib import Path
import tempfile
from zipfile import ZipFile

from flask import Flask, request, abort, jsonify
import joblib
import pandas as pd


ALLOWED_EXTENSIONS = {'.eml'}
ALLOWED_ARCHIVES = {'.zip'}


TEXT_MODEL = joblib.load('text_model.joblib')
OBJECT_MODEL = joblib.load('object_model.joblib')


app = Flask(__name__)


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.post("/api/predict/text")
def api_predict_text():
    input = request.json
    if 'instances' not in input:
        abort(400, description="Request must include an array with key "
                               "'instances'")
    try:
        email_df = pd.DataFrame(input['instances'])[["subject", "body"]]
    except (ValueError, KeyError):
        abort(400, description="Key 'instances' in request must be an array "
                               "of JSON objects with keys 'subject' and "
                               "'body'")
    response_df = pd.DataFrame(
        {"prediction": TEXT_MODEL.predict(email_df).astype(bool)})
    if input.setdefault('return_prob', True):
        response_df['spam_probability'] = \
            TEXT_MODEL.predict_proba(email_df)[:,1]
    if input.setdefault('return_inputs', False):
        response_df['subject_head'] = (email_df['subject'].str.slice(stop=100))
        response_df['body_head'] = email_df['body'].str.slice(stop=100)
    return response_df.to_dict(orient='records')


@app.post("/api/predict/file")
def api_predict_file():
    try:
        file = request.files['file']
    except KeyError:
        abort(400, description="A file named 'file' was not found in the "
                               "request")
    if file and Path(file.filename).suffix in ALLOWED_EXTENSIONS:
        email_obj = email.message_from_binary_file(file, policy=default)
    else:
        abort(400, description="'file' must be one of the following file "
                                f"types: {', '.join(ALLOWED_EXTENSIONS)}")
    response_df = pd.DataFrame({
            "prediction": OBJECT_MODEL.predict([email_obj,]).astype(bool),
            "spam_probability": OBJECT_MODEL.predict_proba([email_obj,])[:,1]
        })
    return response_df.to_dict(orient='records')


@app.post("/api/predict/archive")
def api_predict_archive():
    try:
        archive = request.files['archive']
    except KeyError:
        abort(400, description="A file named 'archive' was not found in "
                               "the request")
    email_objs = {}
    if archive and Path(archive.filename).suffix in ALLOWED_ARCHIVES:
        zip_ref = ZipFile(archive)
    else:
        abort(400, description="'archive' must be one of the following file "
                                f"types: {', '.join(ALLOWED_ARCHIVES)}")
    with tempfile.TemporaryDirectory() as td:
        zip_ref.extractall(td)
        for file in Path(td).iterdir():
            if file.suffix not in ALLOWED_EXTENSIONS:
                abort(400, description="Only email files of the following "
                                       "types are allowed: "
                                       f"{', '.join(ALLOWED_EXTENSIONS)}")
            email_objs[file.name] = email.message_from_bytes(
                file.read_bytes(),policy=default)
    email_obj_series = pd.Series(email_objs)
    response_df = pd.DataFrame({
            "file": email_obj_series.index,
            "prediction": OBJECT_MODEL.predict(email_obj_series).astype(bool),
            "spam_probability": OBJECT_MODEL.predict_proba(email_obj_series)[:,1]
        })
    return response_df.to_dict(orient='records')
