import requests


TEXT_MODEL_URL = "https://github.com/zcline91/spam_filter/releases/" \
                 "download/v1.2.1/text_model.joblib"
OBJECT_MODEL_URL = "https://github.com/zcline91/spam_filter/releases/" \
                   "download/v1.2.1/object_model.joblib"

TEXT_MODEL_FILENAME = "text_model.joblib"
OBJECT_MODEL_FILENAME = "object_model.joblib"

r = requests.get(TEXT_MODEL_URL)
with open(TEXT_MODEL_FILENAME, "wb") as file:
    file.write(r.content)

r = requests.get(OBJECT_MODEL_URL)
with open(OBJECT_MODEL_FILENAME, "wb") as file:
    file.write(r.content)
