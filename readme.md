# Classifier and API for predictions

This package contains two scripts that produce a text classification model based on 
scikit-learn and then use that model to power a lightweight web-api that gives
predictions for the classification of text strings.

## Classifier

The first part of the package is `generate_model.py` which takes as input a directory
containing text files. Each text file consists of a series of lines, each line should
be a string with a value in the category. The name of the category is taken from the
filename (excluding `.txt`). The text files can be gzipped, in which case they should
end with `.txt.gz`

All the text files from the directory are imported into a script which then trains a
text classifier, and saves the model to `model.pkl.gz` (by default). If the filename
for the model (given by the `--output-file` flag) ends in `.gz` then a gzipped version
of the model is created (smaller for transport).

## Server

The second part takes the gzipped model from the classifier stage and loads it into
a simple web API where queries can be passed to it. To run the server, the command
is:

```
python server.py model.pkl.gz
```

where `model.pkl.gz` is the model created in the first part. This will run a server
accessible on `http://localhost:8080/` where you can produce predictions. The server
has two endpoints:

- `/predict_proba?q=XXXXXX` gives a json object containing the predicted
  probabilities for each classification category for the string `XXXXXXX`.
  You can do multiple predictions at once by given multiple `q=` parameters,
  eg `/predict_proba?q=XXXXXX&q=YYYYYYY`.

- `/predict?q=XXXXXX` gives the most likely category for the string
  `XXXXXXX`. If just one `q=` is given then the server just returns the string
  of the most likely one. If more than one `q=` is provided then a JSON object 
  with name:class pairs is returned instead.


# Setup with dokku

1. `dokku apps:create org-type`
2. `dokku domains:enable org-type`
3. `dokku domains:add org-type orgtype.findthatcharity.uk`
4. `dokku config:set --no-restart org-type DOKKU_LETSENCRYPT_EMAIL=your@email.tld`
5. `dokku config:set org-type PREDICT_MODEL=model.pkl.gz`