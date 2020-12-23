

class Cleaner(object):

    def __init__(self, window=96, std_mult=1.5):

        self.window = window
        self.std_mult = std_mult


    def clean(self, df):

        info = df.describe()
        columns = df.columns

        df = self._compute_mean(df)
        self._clean(df, columns, info)
        self._interpolate(df, columns)
        self._drop_mean_columns(df)

        df = df.dropna()

        return df


    def _compute_mean(self, df):
        for column in df.columns:
            column_mean_name = "Mean {0}".format(column)
            df[column_mean_name] = df[ column ].rolling(window=self.window).mean()
        return df
    

    def _clean(self, df, columns, info):

        for column in columns:
            #print(column)

            std = info[column]['std']
            mean_colum = "Mean {0}".format(column)

            for _, row in df.iterrows():
                
                mean = row[mean_colum]
                val = row[column]

                if abs(val - mean) > self.std_mult * std:
                    row[column] = mean
    

    def _interpolate(self, df, columns):
        for column in columns:
            df[ column ] = df[ column ].interpolate(method='linear')
    

    def _drop_mean_columns(self, df):

        columns = df.columns
        for column in columns:
            if "Mean" in column:
                df.drop([column], axis=1, inplace=True)