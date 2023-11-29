import numpy as np
import math
from sklearn.preprocessing import normalize

class NaiveBayes:
    '''
    Kelas yang mengimplementasikan Algoritma Naive Bayes from scratch
    '''
    def __init__(self):
        '''
            Inisialisasi variabel-variabel yang diperlukan
        '''
        self.prior_probs = {}
    
    def fit(self, x, y):
        '''
            Melakukan assignment kolom pada dataset
            params:
            x = dataframe selain kolom target
            y = kolom target
        '''
        self.x = np.array(x)
        self.y = np.array(y)
        self.process_per_label()
        self.prior_probs_init()

    def gaussianDist(self, x, mean, var) :
        '''
            Menghitung probabilitas nilai x berdasarkan fungsi distribusi normal/Gaussian
            params:
            x    = nilai data yang ingin dicek probabilitasnya
            mean = rata-rata nilai kolom yang bersangkutan
            var  = variansi nilai kolom yang bersangkutan
        '''
        return 1/(math.sqrt(var * 2 * math.pi)) * math.exp(-0.5 * math.pow((x-mean),2)/var)
    
    def process_per_label(self):
        '''
            Melakukan pemisahan data per label
        '''
        # Kelompokkan data berdasarkan label
        self.data_per_label = {}

        for label in (np.unique(self.y)) :
            self.data_per_label[label] = []

        for i in range(len(self.y)) :
            self.data_per_label[self.y[i]].append(self.x[i])

        for label in self.data_per_label.keys() :
            self.data_per_label[label] = np.array(self.data_per_label[label])
    
    def prior_probs_init(self):
        '''
            Melakukan kalkulasi prior probability untuk tiap label
        '''
        # Inisialisasi jumlah nilai tiap kolom untuk tiap label
        label, counts = np.unique(self.y, return_counts=True)
        self.prior_probs = dict(zip(label, counts * 1.0/sum(counts)))

    def predict(self, x, var_smoothing):
        '''
            Melakukan prediksi berdasarkan Algoritma Naive Bayes
            params:
            x             = sampel data
            var_smoothing = koefisien untuk menentukan nilai yang ditambahkan ke tiap variance (variance smoothing)
        '''
        x = np.array(x)
        y_pred = []
    
        # Proses kalkulasi utama
        for sample in x :
            maxLabel = 0
            maxProb = 0    
            epsilon = var_smoothing * np.var(self.x, axis=0).max() # Nilai yang akan ditambahkan ke tiap variance
            for label in np.unique(self.y) :
                currProb = 1
                for i in range(len(sample)) :
                    # Hitung probabilitas menggunakan fungsi distribusi Gaussian
                    var = np.var(self.data_per_label[label][:,i]) + epsilon
                    mean = np.mean(self.data_per_label[label][:,i])
                    currProb *= self.gaussianDist(sample[i], mean, var)
                
                # Kalikan dengan prior probability label yang bersangkutan
                currProb *= self.prior_probs[label]
                # Update label dengan probabilitas maksimum
                if (currProb > maxProb) :
                    maxLabel = label
                    maxProb = currProb
            y_pred.append(maxLabel)
        # Kembalikan nilai prediksi
        return y_pred