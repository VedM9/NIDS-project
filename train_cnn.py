"""
train_cnn.py

Trains a 1D CNN on the same NIDS data as the Random Forest baseline, to
test whether a deep learning approach improves on minority-class detection.

This script deliberately runs THREE experiments so the comparison is a
controlled one -- each model is built fresh (untrained weights) before
its own training run:

  1. Unweighted training        -- treats every misclassification equally
  2. Balanced class weighting   -- weight = N / (K * n_class), uncapped
  3. Square-root class weighting -- the square root of the balanced weight,
                                    a softer correction that grows much
                                    more slowly for the rarest classes

See README for the results.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from data_prep import load_and_prepare_data, data_dir_arg
from evaluation import summarize


def build_model(input_len, num_classes):
    """Returns a freshly-initialized, untrained CNN with the same
    architecture every time -- call this again before each training run
    so experiments don't accidentally build on a previous run's weights."""
    keras.utils.set_random_seed(42)
    model = keras.Sequential([
        keras.Input(shape=(input_len, 1)),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate(X_train_cnn, y_train, X_test_cnn, y_test, le, name, class_weight=None):
    model = build_model(X_train_cnn.shape[1], len(le.classes_))  # fresh model
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(
        X_train_cnn, y_train,
        validation_split=0.1, epochs=20, batch_size=512,
        class_weight=class_weight, callbacks=[early_stop], verbose=2,
    )
    y_pred = model.predict(X_test_cnn, batch_size=4096, verbose=0).argmax(axis=1)
    summarize(y_test, y_pred, le, name)


def main():
    args = data_dir_arg("Train and evaluate the 1D CNN experiments").parse_args()
    X_train, X_test, y_train, y_test, le = load_and_prepare_data(args.data_dir)

    # Neural nets need scaled input (unlike Random Forest) since large-
    # magnitude features would otherwise dominate purely by scale.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Conv1D expects (samples, features, channels)
    X_train_cnn = X_train_scaled.reshape(*X_train_scaled.shape, 1)
    X_test_cnn = X_test_scaled.reshape(*X_test_scaled.shape, 1)
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    # 'balanced' weights classes inversely proportional to their frequency,
    # so mistakes on rare classes (e.g. Web Attack, ~1,500 samples) count
    # far more toward the loss than mistakes on BENIGN (~1.8M samples).
    classes = np.unique(y_train)
    balanced = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    train_and_evaluate(X_train_cnn, y_train, X_test_cnn, y_test, le, "CNN unweighted")
    train_and_evaluate(X_train_cnn, y_train, X_test_cnn, y_test, le, "CNN balanced weights",
                       class_weight=dict(zip(classes, balanced)))
    train_and_evaluate(X_train_cnn, y_train, X_test_cnn, y_test, le, "CNN sqrt weights",
                       class_weight=dict(zip(classes, np.sqrt(balanced))))


if __name__ == "__main__":
    main()
