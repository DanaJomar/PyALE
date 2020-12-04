import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

path_to_fixtures = os.path.dirname(__file__)

np.random.seed(876)
x1 = np.random.uniform(low=0, high=1, size=200)
x2 = np.random.uniform(low=0, high=1, size=200)
x3 = np.random.uniform(low=0, high=1, size=200)
x4 = np.random.choice(range(10), 200)
x5 = np.random.choice(["A", "B", "C"], 200)
y = x1 + 2 * x2 ** 2 + np.log(x4 + 1) + np.random.normal(size=200)
X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})


def onehot_encode(feat, groups=X.x5.unique()):
    ohe = OneHotEncoder().fit(feat)
    col_names = ohe.categories_[0]
    feat_coded = pd.DataFrame(ohe.transform(feat).toarray())
    feat_coded.columns = col_names
    missing_feat = [x for x in groups if x not in col_names]
    if missing_feat:
        feat_coded[missing_feat] = 0
    return feat_coded


X_cleaned = pd.concat(
    [X.drop("x5", inplace=False, axis=1), onehot_encode(X[["x5"]])], axis=1
)
model = RandomForestRegressor(random_state=123)
model.fit(X_cleaned, y)

print("Writting X ...... ")
with open(os.path.join(path_to_fixtures, "X.pickle"), "wb") as X_pickle:
    X.to_pickle(X_pickle)
print("Writting X_cleaned ...... ")
with open(os.path.join(path_to_fixtures, "X_cleaned.pickle"), "wb") as X_pickle:
    X_cleaned.to_pickle(X_pickle)
print("Writting y ...... ")
with open(os.path.join(path_to_fixtures, "y.npy"), "wb") as y_npy:
    np.save(y_npy, y)
print("Writting the model ...... ")
with open(os.path.join(path_to_fixtures, "model.pickle"), "wb") as model_pickle:
    pickle.dump(model, model_pickle)

print("Saving metadata ...... ")
with open(
    os.path.join(path_to_fixtures, "fixtures_metadata.txt"), "w"
) as metadata_file:
    metadata_file.writelines(
        [
            "Time created: {}\n".format(pd.Timestamp.now()),
            "sklearn: {}\n".format(model.__getstate__()["_sklearn_version"]),
            "pandas: {}\n".format(pd.__version__),
            "numpy: {}\n".format(np.__version__),
        ]
    )
