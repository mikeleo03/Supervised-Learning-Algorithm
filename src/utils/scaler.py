import numpy as np

class Scaler:
    '''
    Kelas yuntuk melakukan normalisasi terhadap data
    '''
    def __init__(self):
        '''
            Konstruktor nilai min_vals dan max_vals
        '''
        self.min_vals = None
        self.max_vals = None

    def fit(self, X):
        '''
            Mencari nilai minumum dan maksimum
            params :
            X = komponen yang akan dinormalisasi
        '''
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)

    def transform(self, X):
        '''
            Melakukan normalisasi terhadap nilai minimum dan maksimum
            params :
            X = komponen yang akan dinormalisasi
        '''
        X_normalized = (X - self.min_vals) / (self.max_vals - self.min_vals)
        return X_normalized

    def fit_transform(self, X):
        '''
            Menggabungkan proses fit dan transform
            params :
            X = komponen yang akan dinormalisasi
        '''
        self.fit(X)
        return self.transform(X)