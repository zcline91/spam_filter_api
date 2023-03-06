## Usage

The API site is currently hosted at https://spamfilterapi-kepp3lctya-uc.a.run.app with the following endpoints:

### `/api/predict/text`

- Accepts a POST request with a JSON object of the form

```jsonc
{
    "instances": [{"subject": <str>, "body": <str>},]
    "return_prob": <boolean>, // default: true
    "return_inputs": <boolean> // default: false
}
```

`instances` represents an array of emails for the model to create predictions from. If `return_prob` is true (the default), in addition to a boolean prediction of whether or not each instance is spam, the response will contain a probability between 0 and 1 for each instance. If `return_inputs` is true (not default), the response will also contain the head (first 100 characters) of each instance's subject and head.

### `/api/predict/file`

- Accepts a POST request with a single .eml file named "file"
- The response will contain a prediction and probability

### `api/predict/archive`

- Accepts a POST request with a single .zip file of .eml files, named "archive"
- The response will contain a prediction and probability for each .eml file in the archive.


### Output

Regardless of the endpoint accessed, the reponse will be a JSON array of the form:

```jsonc
[
    {
        "prediction": <boolean>,
        "spam_probability": <number>, // optional for text endpoint
        "file": <str>, // only included in archive endpoint
        "subject_head": <str>, // optional, only available from text endpoint
        "body_head": <str>, //optional, only available from text endpoint
    },
]
```

For `api/predict/text`, the array will be in the same order as the `instances` array from the request.
