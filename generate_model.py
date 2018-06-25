import os
import pickle
import argparse
import gzip

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

NAME_FIELD='name'
CLASS_FIELD='class'

# import all the data files
def import_data(input_dir, file_encoding='utf8', ):
    file_data = []
    for f in os.listdir(input_dir):
        if f.endswith(".txt.gz") or f.endswith(".txt"):
            file_d = pd.read_csv(os.path.join(input_dir, f), encoding=file_encoding, header=None)
            file_d.columns = [NAME_FIELD]
            file_d.loc[:, CLASS_FIELD] = f.replace(".txt", '').replace('.gz', '')
            file_data.append(file_d)
            print("{:,.0f} rows imported from {}".format(len(file_d), os.path.join(input_dir, f)))
    population = pd.concat(file_data, sort=False)
    print("{:,.0f} rows imported in total".format(len(population)))
    return population

# create a sample
def get_sample(population, max_sample=10000):
    sample = []
    for i in population[CLASS_FIELD].dropna().unique():
        this_pop = population[population[CLASS_FIELD]==i]
        if len(this_pop) > max_sample:
            sample.append(this_pop.sample(max_sample))
        else:
            sample.append(this_pop)
    sample = pd.concat(sample)
    return sample

# set up the pipeline
def get_pipeline(use_gridsearch=False):
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                        ('tfidf', TfidfTransformer(use_idf=False)),
                        ('clf', MultinomialNB(alpha=0.01)),
    ])

    # set up the parameters to test
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1,3), (2,3)],
                'tfidf__use_idf': (True, False),
                'clf__alpha': (1e-2, 1e-3),
                }
    # best: {'clf__alpha': 0.01, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 2)}

    if use_gridsearch:
        # use gridsearch to produce the optimum model
        model = GridSearchCV(text_clf, parameters, n_jobs=1)
    else:
        model = text_clf

    return model

# fit the model to a sample of the data
def fit_model(model, sample):
    model.fit(sample[NAME_FIELD], sample[CLASS_FIELD])

# get a table with the predicted and actual results
def print_fit_results(model, population):
    results = pd.DataFrame({
        "prediction": model.predict(population[NAME_FIELD]),
        "actual": population[CLASS_FIELD]
    })

    # print the metrics for the model's predictive power
    print("Classification metrics:")
    print(metrics.classification_report(
        results["actual"].fillna("Unknown"), 
        results["prediction"]
        ))
    print()

    # print the parameters found using the model
    if hasattr(model, 'best_params_'):
        print("Optimum parameters:")
        print(model.best_params_)
        print()

    return results

# save the model to a gzipped pickle file
def save_model(model, output_file):
    if output_file.endswith(".gz"):
        a = gzip.open(output_file, 'rb')
    else:
        a = open(output_file, 'rb')
    with a:
        pickle.dump(model, a, protocol=2)
        


def main():

    parser = argparse.ArgumentParser(description='Generate a classification model with sklearn')

    # files to use
    parser.add_argument('--input-dir', default='model_inputs', help='Location of directory containing .txt files to help classification. The files should contain a list of strings to be classified, while the filename gives the classification category.')
    parser.add_argument('--output-file', default='model.pkl.gz', help='Location of pickle file containing the model that will be saved.')
    parser.add_argument('--file-encoding', default='utf8', help='Encoding of text files.')

    # model options
    parser.add_argument('--max-sample', default=10000, type=int, help='Maximum number of rows to sample from a particular class')
    parser.add_argument('--gridsearch', action='store_true', dest="use_gridsearch", help='If given the classifier will try to find the optimum solution from a range')

    args = parser.parse_args()

    
    print("Importing data...")
    population = import_data(args.input_dir, args.file_encoding)
    print()
    print("{} value counts:".format(CLASS_FIELD))
    print(population[CLASS_FIELD].value_counts())
    print()
    exit()

    print("Creating sample with maximum {:,.0f} rows per class...".format(args.max_sample))
    sample = get_sample(population, args.max_sample)
    print("Sample created with {:,.0f} rows".format(len(sample)))
    print()
    print("{} value counts (sample):".format(CLASS_FIELD))
    print(sample[CLASS_FIELD].value_counts())
    print()

    print("Set up the classification pipeline")
    if args.use_gridsearch:
        print("Finding optimum model to use...")
    model = get_pipeline(args.use_gridsearch)

    print("Fitting to model...")
    fit_model(model, sample)

    print_fit_results(model, population)

    print("Saving to '{}'".format(args.output_file))
    save_model(model, args.output_file)

if __name__ == '__main__':
    main()