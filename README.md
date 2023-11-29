# Tugas Besar 2 IF3170 - Intelegensi Buatan
<h2 align="center">
    Supervised-Learning Algorithm<br/>
</h2>
<hr>

> Disusun untuk memenuhi Tugas Besar 2 - Supervised learning Algorithm | IF3170 Intelegensia Buatan tahun 2023/2024 

## Table of Contents
1. [General Info](#general-information)
2. [Creator Info](#creator-information)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Structure](#structure)

<a name="general-information"></a>

## General Information
Pada tugas besar ini, Kami melakukan implementasi algoritma pembelajaran mesin KNN dan Naive-Bayes (sesuai dengan cakupan materi kuliah IF3170 - Intelegensia Buatan). Data yang digunakan pada implementasi ini sama seperti data [tugas kecil 2](https://drive.google.com/file/d/14kZUHH39P9-U2W8KDJt1i2X1wVJ_45bf/view?usp=drive_link). Kami melakukan proses pelatihan model menggunakan data latih yang terdapat pada pranala tersebut, kemudian dilakukan validasi hasil dengan menggunakan data validasi untuk mendapatkan _insight_ seberapa baik model melakukan generalisasi.Tahap selanjutnya adalah melakukan perbandingan hasil implementasi algoritma KNN dan Naive-Bayes kelompok Kami dengan algoritma milik pustaka eksternal _scikit-learn_. Parameter perbandingan yang digunakan, antara lain: _precision_, _recall_, _F1-score_, _support_, _accuracy_, _macro avg_, dan _weighted avg_.
<a name="creator-information"></a>

## Creator Information

| Nama                        | NIM      | E-Mail                      |
| --------------------------- | -------- | --------------------------- |
| Michael Leon Putra Widhi    | 13521108 | 13521108@std.stei.itb.ac.id |
| Muhammad Zaki Amanullah     | 13521146 | 13521146@std.stei.itb.ac.id |
| Mohammad Rifqi Farhansyah   | 13521166 | 13521166@std.stei.itb.ac.id |
| Nathan Tenka                | 13521172 | 13521172@std.stei.itb.ac.id |

<a name="features"></a>

## Features
1. Implementasi algoritma KNN dan Naive-Bayes
2. Perbandingan hasil implementasi algoritma KNN dan Naive-Bayes dengan pustaka eksternal _scikit-learn_
3. Penyimpanan dan load _model_
4. Submisi kaggle

<a name="technologies-used"></a>

## Technologies Used
- python
- numpy
- pandas
- matplotlib
- scikit-learn

<a name="structure"></a>

## Structure
```bash
│   README.md
│
├───data
│       data_train.csv
│       data_validation.csv
│       full_data.csv
│       test.csv
│
├───result
│       predictions-knn.csv
│       predictions-naive-bayes.csv
│
└───src
    │   knn.ipynb
    │   naive.ipynb
    │
    ├───algorithm
    │       knn.py
    │       naiveBayes.py
    │       weightedKnn.py
    │
    ├───models
    │       knn_model.pkl
    │       naive_bayes_model.pkl
    │
    └───utils
            scaler.py
```