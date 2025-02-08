# takes dataframe, the columns to be noised, the mean of the Gaussian noise, and its variance as input
# doesn't output anything; modifies in-place
import numpy as np

def injectGaussianNoise(df, columnNames, mu, sigma):
    for col in columnNames:
        # print(np.random.normal(mu, sigma, df[col].shape))
        df[col] = df[col] + np.random.normal(mu, sigma, df[col].shape)
    return