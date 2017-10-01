import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import operator

class featureSelector(object):
    
    def __init__(self, model, scorer, cv, prcnt=0.8, to_keep=2, tol=0.05, mode='reg', verbose=False):

        self.mode = mode
        self.model = model
        self.verbose = verbose
        self.scorer = scorer
        self.prcnt = prcnt
        self.to_keep = to_keep
        self.tol = tol
        self.feature_sets = []
        self.best = None
        self.cv = cv

    def rank_pertrub(self, x, y):
        impt = np.zeros(x.shape[1])
        for i, col in enumerate(x.columns):
            hold = x[col]
            x[col] = x[col].sample(frac=1).values
            if self.mode == 'reg':
                score = self.scorer(y, self.model.predict(x))
            if self.mode == 'class':                
                score = self.scorer(y, self.model.predict_proba(x))
            impt[i] = score
            x[col] = hold
        impt /= np.sum(impt)
        return impt

    def rank_stat(self, x, y):
        impt = np.zeros(x.shape[1])
        for i, col in enumerate(x.columns):
            c = np.corrcoef(x[col].values, y.ravel())[1,0]
            impt[i] = abs(c)
        impt /= np.sum(impt)
        return impt
        
    def fit(self, train, y_train):
        self.features = list(train.columns)
        self.scores = []
        trsh = int((self.to_keep / self.prcnt) * (1-self.prcnt) + self.to_keep)
        if trsh <= self.to_keep:
            trsh += 1
        while True:
            if len(self.features) >= trsh:
                score, importances, impt_pertrub, impt_stat, fold = 0, 0, 0, 0, 1
                for train_index, test_index in self.cv.split(train, y_train):
                    self.model.fit(train[self.features].loc[train_index], y_train[train_index])
                    if self.mode == 'reg':
                        score += self.scorer(y_train[test_index], self.model.predict(train[self.features].loc[test_index]))
                    elif self.mode == 'class':
                        score += self.scorer(y_train[test_index], self.model.predict_proba(train[self.features].loc[test_index]))
                    try:
                        importances_std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
                        importances_std /= np.sum(importances_std)
                        importances_ = ((self.model.feature_importances_/np.sum(self.model.feature_importances_))*importances_std)
                    except: 
                        importances_ = abs(self.model.coef_[0])
                    importances += importances_ / np.sum(importances_)
                    impt_pertrub += self.rank_pertrub(train[self.features].loc[train_index], y_train[train_index])
                    impt_stat += self.rank_stat(train[self.features].loc[train_index], y_train[train_index])
                    fold += 1
                importances /= fold
                impt_pertrub /= fold                
                score /= fold
                d = np.std(impt_pertrub)
                importances += impt_pertrub * d + impt_stat * (1 - d)
                importances /= np.sum(importances)
                self.scores.append(score)
                if self.verbose:
                    print(len(self.features), ' : ', score)
                sort_importances = sorted([(j,k) for j, k in zip(self.features, importances)], key=operator.itemgetter(1), reverse=True)
                self.feature_sets.append(sort_importances)
                self.features = [i[0] for i in sort_importances[:int(len(sort_importances) * self.prcnt)]]
            else:
                break
        min_ = min(self.scores)
        for score in self.scores[::-1]:
            if score != min_:
                if score < min_ or abs(score - min_) <= self.tol:
                    self.best = self.scores.index(score)
                    break
            else:
                self.best = self.scores.index(min_)

    def transform(self, df):
        return df[[i[0] for i in self.feature_sets[self.best]]]