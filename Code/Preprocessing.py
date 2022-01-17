import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import QuantileTransformer

class Preprocess:
    def __init__(self, data_input_path, resample_method):
        self.df = pd.read_csv(data_input_path)
        self.resample_method = resample_method
        self.resample_data()
        self.transform_data()
        self.drop_correlated()

    def resample_data(self):
        if self.resample_method.lower() == 'random' or 'rand' or 'r':
            over_sampler = RandomOverSampler(sampling_strategy="minority")
        elif self.resample_method.lower() == 'smote' or 'smotek' or 's':
            over_sampler = SMOTE(sampling_strategy="minority")
        X = self.df.iloc[:, :30]
        Y = self.df.iloc[:, 30]

        X_oversampled, Y_oversampled = over_sampler.fit_resample(X, Y)
        counter = 0
        o_df = pd.DataFrame()
        for i in list(self.df.columns)[:30]:
            o_df[i] = X_oversampled[i]
            counter += 1
            
        o_df["Class"] = Y_oversampled
        self.df = o_df.copy()

    def transform_data(self):
        cols = ["Time", "V1", "V3", "V4", "V7", "V10", "V11", "V12", "V14", "V16", "V17", "V18", "V19", "V24"]
        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')
        for i in cols:
            self.df[i] = qt.fit_transform(np.array(self.df[i]).reshape(-1,1))

    def drop_correlated(self):
        self.df.drop(columns=["V5", "V14", "V17", "V22"], inplace = True)

# preprocess = Preprocess('path/to/data', 'ranom / smote')