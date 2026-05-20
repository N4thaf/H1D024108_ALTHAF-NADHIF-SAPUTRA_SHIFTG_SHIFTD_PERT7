Langkah Pembuatan Program

Import library yang dibutuhkan: TensorFlow.
Load dataset dari file iris.data menggunakan Pandas, lalu pisahkan menjadi fitur X (4 kolom pertama) dan label y (kolom terakhir).
Konversi label spesies dari string menjadi numerik (0, 1, 2) menggunakan LabelEncoder dari Scikit-learn.
Bagi dataset menjadi data training dan testing dengan rasio 80:20 menggunakan train_test_split.
Bangun model Sequential dengan 1 input layer dan 4 Dense layer: tiga hidden layer (1000, 500, 300 neuron) dengan aktivasi ReLU, dan output layer dengan 3 neuron dan aktivasi Softmax.
Kompilasi model menggunakan optimizer Adam, loss function sparse_categorical_crossentropy, dan metrik accuracy.
Latih model selama 50 epoch dengan batch size 32, sambil memvalidasi performa di setiap epoch menggunakan data testing.
Evaluasi model, tampilkan confusion matrix, lalu prediksi data baru.
