import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for test_word in test_set.get_all_Xlengths().keys():
        test = test_set.get_all_Xlengths()[test_word]
        temp_dict = {}
        for word in models.keys():
            try:
                score = models[word].score(test[0], test[1])
            except:
                score = float('-inf')

            temp_dict[word] = score
        probabilities.append(temp_dict)


    guesses = [max(probabilities[p].keys(), key=lambda k: probabilities[p][k]) for p in range(len(probabilities))]

    # TODO implement the recognizer
    return probabilities, guesses

