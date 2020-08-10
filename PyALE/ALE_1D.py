import numpy as np
import pandas as pd

def aleplot_1D_continuous(X, model, feature, grid_size = 40):
    quantiles = np.append(0, np.arange(1/grid_size, 1+1/grid_size, 1/grid_size))
    # feat_cut, bins = pd.qcut(X[feature], quantiles, duplicates='drop', retbins=True, precision=1)
    bins = np.round([min(X[feature])] + X[feature].quantile(quantiles).to_list(), 1)
    bins = np.unique(bins)
    feat_cut = pd.cut(X[feature], bins, include_lowest=True)
    
    bin_codes = feat_cut.cat.codes
    bin_codes_unique = np.unique(bin_codes)
    
    X1 = X.copy()
    X2 = X.copy()
    X1[feature] = [bins[i] for i in bin_codes]
    X2[feature] = [bins[i+1] for i in bin_codes]
    y_1 = model.predict(X1)
    y_2 = model.predict(X2)
    
    res_df = pd.DataFrame({'x':X1[feature], 'Delta':y_2 - y_1})
    res_df = res_df.groupby(['x']).Delta.agg(['size', 'mean'])
    res_df['eff'] = res_df['mean'].cumsum()
    res_df = res_df.assign(eff = lambda df: df['eff'].shift(1, fill_value=0) - ((df['eff'] + df['eff'].shift(1, fill_value=0))/2 * df['size']).sum()/df['size'].sum())
    return(res_df['eff'])


def aleplot_1D_discrete(X, model, feature):
    groups = X[feature].unique()
    groups.sort()
    groups_codes = [x for x in range(len(groups))]
    
    groups_counts = X.groupby(feature).size()
    groups_props = groups_counts/sum(groups_counts)
    
    K = len(groups)
    
    # create copies of the dataframe
    X_plus = X.copy()
    X_neg = X.copy()
    # all groups except last one
    ind_plus = X[feature] < groups[K-1]
    # all groups except first one
    ind_neg = X[feature] > groups[0]
    # replace once with one level up
    X_plus.loc[ind_plus, feature] = groups[X.loc[ind_plus, feature] + 1]
    # replace once with one level down
    X_neg.loc[ind_neg, feature] = groups[X.loc[ind_neg, feature] - 1]
    # predict with original and with the replaced values
    y_hat = model.predict(X)
    y_hat_plus = model.predict(X_plus[ind_plus])
    y_hat_neg = model.predict(X_neg[ind_neg])
    # compute prediction difference
    Delta_plus = y_hat_plus - y_hat[ind_plus]
    Delta_neg = y_hat[ind_neg] - y_hat_neg
    
    # compute the mean of the difference per group
    res_df = pd.concat([pd.DataFrame({'Delta':Delta_plus, 'x':X.loc[ind_plus, feature] + 1}), 
                        pd.DataFrame({'Delta':Delta_neg, 'x':X.loc[ind_neg, feature]})])
    res_df = res_df.groupby(['x']).mean()
    res_df['eff'] = res_df['Delta'].cumsum()
    res_df.loc[0] = 0
    res_df = res_df.sort_index()
    res_df['eff'] = res_df['eff'] - sum(res_df['eff']*groups_props)
    return(res_df)
