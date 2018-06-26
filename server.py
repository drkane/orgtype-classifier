from __future__ import print_function
import os
import argparse
import pickle
import gzip
from collections import OrderedDict

import bottle

app = bottle.default_app()

def fetch_model(location):
    if location.endswith(".gz"):
        a = gzip.open(location, 'rb')
    else:
        a = open(location, 'rb')
    with a:
        return pickle.load(a)

# get model location from environment variable if set
if os.environ.get("PREDICT_MODEL"):
    app.config["predict_model"] = fetch_model(os.environ.get("PREDICT_MODEL"))


@app.route('/predict_proba')
def get_prediction_proba():
    if not app.config.get("predict_model"):
        return bottle.abort(501, "Prediction model not accessible.")
        
    search = bottle.request.params.getlist("q")
    if not search:
        return {}

    model = app.config.get("predict_model")
    
    results = {}
    for i, v in enumerate(model.predict_proba(search)):
        r = zip(model.classes_.tolist(), v.tolist())
        r = sorted(r, key=lambda x: x[1], reverse=True)
        results[search[i]] = OrderedDict(r)
    return results

@app.route('/predict')
def get_prediction():
    if not app.config.get("predict_model"):
        return bottle.abort(501, "Prediction model not accessible.")
        
    search = bottle.request.params.getlist("q")
    if not search:
        return {}

    model = app.config.get("predict_model")
    
    results = {}
    for i, v in enumerate(model.predict(search)):
        results[search[i]] = v
    if len(results)==1:
        return results[search[0]]
    return results


def main():

    parser = argparse.ArgumentParser(description='API wrapper for a sklearn prediction API')

    # prediction model
    parser.add_argument('predict_model', help='file location of prediction model in pickle format')

    # server options
    parser.add_argument('-host', '--host', default="localhost", help='host for the server')
    parser.add_argument('-p', '--port', default=8080, help='port for the server')
    parser.add_argument('--debug', action='store_true', dest="debug", help='Debug mode (autoreloads the server)')
    parser.add_argument('--server', default="auto", help='Server backend to use (see http://bottlepy.org/docs/dev/deployment.html#switching-the-server-backend)')

    args = parser.parse_args()

    if args.predict_model:
        app.config["predict_model"] = fetch_model(args.predict_model)

    bottle.debug(args.debug)

    bottle.run(app, server=args.server, host=args.host, port=args.port, reloader=args.debug)

if __name__ == '__main__':
    main()
