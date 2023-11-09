import numpy as np

class KNNAlgorithm:
    '''
    Kelas yang mengimplementasikan Algoritma KNN from scratch
    '''
    def __init__(self, k):
        '''
            Inisialisasi nilai k yang menyatakan jumlah neighbor yang dpilih
            params :
            k = jumlah neighbor
        '''
        self.k = k
    
    def fit(self, x, y):
        '''
            Melakukan assignment kolom pada dataset
            params:
            x = dataframe selain kolom target
            y = kolom target
        '''
        self.x = np.array(x)
        self.y = np.array(y)

    def euclidean_dist(self, x1, x2):
        '''
            Melakukan kalkulasi nilai jarak euclidean
            params:
            x1 = nilai titik pertama
            x2 = nilai titik kedua
        '''
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, x):
        '''
            Melakukan prediksi berdasarkan Algoritma KNN
            params:
            x = sampel data
        '''
        x = np.array(x)
        y_pred = []
        
        # Proses kalkulasi utama
        for sample in x:            # Untuk setiap sampel data
            distances = []          # Untuk menyimpan nilai jarak
            
            # Lakukan kalkulasi jarak antara sampel dengan setiap data training
            for train_sample, train_label in zip(self.x, self.y):
                distance = self.euclidean_dist(sample, train_sample)
                distances.append((train_sample, train_label, distance))     # Simpan pada senarai

            distances.sort(key=lambda x: x[2])      # Urutkan berdasarkan jarak
            neighbors = distances[:self.k]          # Ambil k tetangga terdekat

            # Ambil nilai label dari setiap tetangga tersebut
            labels = [neighbor[1] for neighbor in neighbors]
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Pilih yang memiliki frekuensi paling tinggi
            y_pred.append(unique_labels[np.argmax(counts)])

        # Kembalikan nilai prediksi
        return y_pred
        