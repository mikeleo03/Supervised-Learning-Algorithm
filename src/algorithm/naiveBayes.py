import numpy as np
import math
from sklearn.preprocessing import normalize

class NaiveBayes:
    '''
    Kelas yang mengimplementasikan Algoritma Naive Bayes from scratch
    '''
    def __init__(self):
        self.class_cond_probs = {}
        self.prior_probs = {}
        self.data_kinds = []
    
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

    def process_per_label(self):
        '''
            Melakukan pemrosesan data untuk tiap label pada setiap kolom
            -
        '''
        # Kelompokkan data berdasarkan label
        self.data_per_label = {}

        for label in (np.unique(self.y)) :
            self.data_per_label[label] = []

        for i in range(len(self.y)) :
            self.data_per_label[self.y[i]].append(self.x[i])

        for label in self.data_per_label.keys() :
            self.data_per_label[label] = np.array(self.data_per_label[label])

        boundaries = {}
        for label in self.data_per_label.keys() :
            currMax = np.max(self.data_per_label[label], axis=0)
            currMin = np.min(self.data_per_label[label], axis=0)
            currRanges = currMax-currMin
            boundaries[label] = []
            for i in range(len(currRanges)) :
                if (len(np.unique(self.data_per_label[label][:,i])) > 2) :
                    # Data kontinu, gunakan gaussian naive bayes
                    self.data_kinds.append("continous")
                    currProbs = {}
                else :
                    # Data boolean, lakukan perbandingan jumlah saja
                    self.data_kinds.append("bool")
                    currProbs = {}
                    for j in np.unique(self.data_per_label[label][:,i]) :
                        currProbs[j] = np.count_nonzero(self.data_per_label[label][:,i] == j)/len(self.data_per_label[label][:,i])
                
                boundaries[label].append(currProbs)
        
        self.class_cond_probs = boundaries
    
    def kernel_function(self, x) :
        '''
        Fungsi kernel untuk melakukan kernel density estimation
        '''

        # Gaussian
        # return (1/math.sqrt(2*math.pi)) * math.exp(-0.5*x**2)
    
        # Logistic
        return 1/(math.exp(x) + 2 + math.exp(-x))

        # Sigmoid
        # return 2/math.pi * (1/(math.exp(x) + math.exp(-x)))
    
        # Silverman
        # return 0.5 * math.exp(-abs(x)/math.sqrt(2)) * math.sin(abs(x)/math.sqrt(2) + math.pi/4)

    def KDE(self, x, data, bandwidth) :
        '''
        Menentukan probabilitas nilai x berdasarkan data menggunakan kernel density estimation
        '''
        currSum = 0
        for val in data :
            currSum += self.kernel_function((x-val)/bandwidth)
        currSum /= len(data)*bandwidth
        return currSum
    
    def prior_probs_init(self):
        '''
            Melakukan kalkulasi class conditional probabilities untuk tiap label
            -
        '''
        # Inisialisasi jumlah nilai tiap kolom untuk tiap label
        label, counts = np.unique(self.y, return_counts=True)
        self.prior_probs = dict(zip(label, counts/sum(counts)))

    def predict(self, x):
        '''
            Melakukan prediksi berdasarkan Algoritma Naive Bayes
            params:
            x = sampel data
        '''
        x = np.array(x)
        y_pred = []
    
        # Proses kalkulasi utama
        for sample in x :
            maxLabel = 0
            maxProb = 0    
            for label in np.unique(self.y) :
                currProb = 1
                for i in range(len(sample)) :
                    if (self.data_kinds[i] == "continous") :
                        # Cari nilai bandwidth paling optimal menggunakan Silverman's rule of thumb
                        std = np.std(self.data_per_label[label][:,i])
                        iqr = np.percentile(self.data_per_label[label][:,i],75)-np.percentile(self.data_per_label[label][:,i],25)
                        n = len(self.data_per_label[label][:,i])
                        bandwidth = 0.9 * min(std, iqr/1.35) * n**(-1/5)

                        # Kalikan dengan probabilitas sebelumnya
                        currProb *= self.KDE(sample[i], self.data_per_label[label][:,i], bandwidth)
                    else :
                        currProb *= self.class_cond_probs[label][i][sample[i]]
                currProb *= self.prior_probs[label]
                if (currProb > maxProb) :
                    maxLabel = label
                    maxProb = currProb
            y_pred.append(maxLabel)
        # Kembalikan nilai prediksi
        return y_pred