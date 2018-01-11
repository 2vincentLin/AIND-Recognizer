import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    definition of variables

    logL: the score given by hmm.score
    p: number of free variable
    N: number of observation, in this case, number of frames
    n: number of state (n_component)
    d: number of features

    explanation

    number of free variable has 4 terms,
        1. starting probability: this is number of states (n_component, let it be n)
        2. transition probability: this is n by n matrix
        3. mean array: this is n by d array
        4. covariance array: this is n by d array

        for 1., the probability sum to 1, so number of free variables is n - 1
        for 2., each column sum to 1, so number of free variables is n*(n-1)
        for 3. and 4., both are n*d

        so p should be n*(n-1)+n-1+2*n*d

    from A Model Selection Criterion for Classification: Application to HMM Topology Optimization, we know that BIC we
    use here is the real BIC time -2, so unlike other selectors, here we want the minimum BIC.


    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
            except:
                break

            if logL is not None:
                d = self.X.shape[1]
                n = n_components
                N = self.X.shape[0]
                p = n * (n - 1) + 2 * n * d + n - 1
                bic = -2 * logL + p * np.log(N)

                if best_score is None or best_score > bic:
                    best_score = bic
                    best_model = model
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    M is number of all words

    the idea of DIC is we train a hmm model with a word, then we get the hmm score of all words,
    we then calculate score(this word) - average(sum(score(all other words))).
    repeat this with number of n_component, then we choose model with highest DIC.

    '''

    def test(self):
        print('all words', self.hwords.keys())
        print('len is ', len(self.hwords))
        print('len of keys is ', len(self.hwords.keys()))
        print([word for word in self.hwords if word != self.this_word])
        print(len(([word for word in self.hwords if word != self.this_word])))

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_dic = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
            except:
                break

            if model is not None:
                log_list = [model.score(self.hwords[word][0], self.hwords[word][1]) for word in self.hwords if
                            word != self.this_word]
                dic = logL - np.mean(log_list)
                if best_dic is None or best_dic < dic:
                    best_model = model
                    best_dic = dic
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    the parameter we need to tune is n_components.
    here we utilize combine_sequences, this function will return [X, length],
    all you need is feed it [index, X]

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = None
        n_splits = min(len(self.sequences), 3)  # there are some cases can't be divided into 3 folds.



        for n_components in range(self.min_n_components, self.max_n_components + 1):

            log_scores = []

            if n_splits == 1:
                return self.base_model(n_components)

            else:

                split_method = KFold(random_state=self.random_state, n_splits=n_splits)
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train = combine_sequences(cv_train_idx, self.sequences)
                    test = combine_sequences(cv_test_idx, self.sequences)
                    try:
                        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train[0], train[1])
                        logL = model.score(test[0], test[1])
                        log_scores.append(logL)

                    except:
                        # sometimes, 1st fold success, 2nd fails, we need to set log_scores to None
                        # to avoid being calculated in next part.
                        # print('fail, word = {}, component = {}, score= {}'.format(self.this_word, n_components, scores))
                        log_scores = None
                        break
                if log_scores is not None:
                    if best_model is None or best_score < np.mean(log_scores):
                        best_model = model
                        best_score = np.mean(log_scores)

                # print('success, word = {}, component = {}, avg score= {}'.format(self.this_word, n_components,
                #                                                                 np.mean(scores)))

        # print(best_model, best_score)
        return best_model
