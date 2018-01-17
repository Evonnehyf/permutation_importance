# Hybrid feature importance for feature selection

This simple algorithm combines input perturbation and ML model's weight ranks:  

importance = weight_rank + perturbation_rank * std(perturbation_rank)  

Can be used with linear / tree-based models from sklearn.  
Compatible with the sklearn pipeline.  

Check out the code for more info.  

## Installation
```
pip install featureselection
```


## Usage

```
from featureSelection import featureSelector
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

selector = featureSelector(model=RandomForestRegressor(n_estimators=100), scorer=mean_squared_error, 
                           cv=KFold(n_splits=5), prcnt=0.8, to_keep=1, tol=0.01, mode='reg', verbose=True)
                           
selector.fit(train, y_train.values)
train = selector.transform(train)
```

Algorithm recursively drops 1.0-**prcnt** of features, modifies feature importance according to the formula above and stops when the number of remaining features == **to_keep**.  
Transform method chooses feature set with minimum amount of features, which satisfies following condition:

score - min(scores) <= **tol**

## Dependencies  
* python 3.6
* numpy 1.12.1
* pandas 0.20.1
