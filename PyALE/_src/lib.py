import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import t


def cmds(D, k=2):
    """Classical multidimensional scaling

    Theory and code references:
    https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/

    Arguments:
    D -- A squared matrix-like object (array, DataFrame, ....), usually a distance matrix
    """

    n = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise Exception("The matrix D should be squared")
    if k > (n - 1):
        raise Exception("k should be an integer <= D.shape[0] - 1")

    # (1) Set up the squared proximity matrix
    D_double = np.square(D)
    # (2) Apply double centering: using the centering matrix
    # centering matrix
    center_mat = np.eye(n) - np.ones((n, n)) / n
    # apply the centering
    B = -(1 / 2) * center_mat.dot(D_double).dot(center_mat)
    # (3) Determine the m largest eigenvalues
    # (where m is the number of dimensions desired for the output)
    # extract the eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(B)
    # sort descending
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    # (4) Now, X=eigenvecs.dot(eigen_sqrt_diag),
    # where eigen_sqrt_diag = diag(sqrt(eigenvals))
    eigen_sqrt_diag = np.diag(np.sqrt(eigenvals[0:k]))
    ret = eigenvecs[:, 0:k].dot(eigen_sqrt_diag)
    return ret


def order_groups(X, feature):
    """Assign an order to the values of a categorical feature.

    The function returns an order to the unique values in X[feature] according to
    their similarity based on the other features.
    The distance between two categories is the sum over the distances of each feature.

    Arguments:
    X -- A pandas DataFrame containing all the features to considering in the ordering
    (including the categorical feature to be ordered).
    feature -- String, the name of the column holding the categorical feature to be ordered.
    """

    features = X.columns
    # groups = X[feature].cat.categories.values
    groups = X[feature].unique()
    D_cumu = pd.DataFrame(0, index=groups, columns=groups)
    K = len(groups)
    for j in set(features) - set([feature]):
        D = pd.DataFrame(index=groups, columns=groups)
        # discrete/factor feature j
        # e.g. j = 'color'
        if (X[j].dtypes.name == "category") | (
            (len(X[j].unique()) <= 10) & ("float" not in X[j].dtypes.name)
        ):
            # counts and proportions of each value in j in each group in 'feature'
            cross_counts = pd.crosstab(X[feature], X[j])
            cross_props = cross_counts.div(np.sum(cross_counts, axis=1), axis=0)
            for i in range(K):
                group = groups[i]
                D_values = abs(cross_props - cross_props.loc[group]).sum(axis=1) / 2
                D.loc[group, :] = D_values
                D[group] = D_values
        else:
            # continuous feature j
            # e.g. j = 'length'
            # extract the 1/100 quantiles of the feature j
            seq = np.arange(0, 1, 1 / 100)
            q_X_j = X[j].quantile(seq).to_list()
            # get the ecdf (empiricial cumulative distribution function)
            # compute the function from the data points in each group
            X_ecdf = X.groupby(feature)[j].agg(ECDF)
            # apply each of the functions on the quantiles
            # i.e. for each quantile value get the probability that j will take
            # a value less than or equal to this value.
            q_ecdf = X_ecdf.apply(lambda x: x(q_X_j))
            for i in range(K):
                group = groups[i]
                D_values = q_ecdf.apply(lambda x: max(abs(x - q_ecdf[group])))
                D.loc[group, :] = D_values
                D[group] = D_values
        D_cumu = D_cumu + D
    # reduce the dimension of the cumulative distance matrix to 1
    D1D = cmds(D_cumu, 1).flatten()
    # order groups based on the values
    order_idx = D1D.argsort()
    groups_ordered = D_cumu.index[D1D.argsort()]
    return pd.Series(range(K), index=groups_ordered)


def quantile_ied(x_vec, q):
    """
    Inverse of empirical distribution function (quantile R type 1).

    More details in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    https://en.wikipedia.org/wiki/Quantile

    Arguments:
    x_vec -- A pandas series containing the values to compute the quantile for
    q -- An array of probabilities (values between 0 and 1)
    """

    x_vec = x_vec.sort_values()
    n = len(x_vec) - 1
    m = 0
    j = (n * q + m).astype(int)  # location of the value
    g = n * q + m - j

    gamma = (g != 0).astype(int)
    quant_res = (1 - gamma) * x_vec.shift(1, fill_value=0).iloc[j] + gamma * x_vec.iloc[
        j
    ]
    quant_res.index = q
    # add min at quantile zero and max at quantile one (if needed)
    if 0 in q:
        quant_res.loc[0] = x_vec.min()
    if 1 in q:
        quant_res.loc[1] = x_vec.max()
    return quant_res


def CI_estimate(x_vec, C=0.95):
    """Estimate the size of the confidence interval of a data sample.

    The confidence interval of the given data sample (x_vec) is
    [mean(x_vec) - returned value, mean(x_vec) + returned value].
    """
    alpha = 1 - C
    n = len(x_vec)
    stand_err = x_vec.std() / np.sqrt(n)
    critical_val = 1 - (alpha / 2)
    z_star = stand_err * t.ppf(critical_val, n - 1)
    return z_star
