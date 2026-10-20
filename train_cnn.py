"""
train_cnn.py

Trains a 1D CNN on the same NIDS data as the Random Forest baseline, to
test whether a deep learning approach improves on minority-class detection.

This script deliberately runs TWO experiments so the comparison is a
controlled one -- each model is built fresh (untrained weights) before
its own training run:

  1. Unweighted training        -- treats every misclassification equally
  2. Class-weighted training    -- penalizes mistakes on rare classes more

Result (see README for full tables): the unweighted CNN under-detected
minority classes (e.g. Bot recall 0.39, Web Attack recall 0.14) compared
to the Random Forest baseline. Balanced class weighting fixed recall on
those classes but overcorrected, tanking precision (e.g. Bot precision
dropped to 0.03) -- the model began flagging far too much traffic as
those rare classes. The Random Forest baseline remained the best overall
model for this dataset.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from data_prep import load_and_prepare_data

DATA_DIR = r"C:\Users\mved0\Downloads\MachineLearningCSV\MachineLearningCVE"


def build_model(input_len, num_classes):
    """Returns a freshly-initialized, untrained CNN with the same
    architecture every time -- call this again before each training run
    so experiments don't accidentally build on a previous run's weights."""
    model = keras.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                       input_shape=(input_len, 1)),
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


def evaluate(model, X_test_cnn, y_test, le, label):
    y_pred = model.predict(X_test_cnn).argmax(axis=1)
    print(f"\n--- {label} ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))


def main():
    X_train, X_test, y_train, y_test, le = load_and_prepare_data(DATA_DIR)
    num_classes = len(le.classes_)

    # Neural nets need scaled input (unlike Random Forest) since large-
    # magnitude features would otherwise dominate purely by scale.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Conv1D expects (samples, features, channels)
    X_train_cnn = X_train_scaled.reshape(*X_train_scaled.shape, 1)
    X_test_cnn = X_test_scaled.reshape(*X_test_scaled.shape, 1)

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # --- Experiment 1: unweighted ---
    model = build_model(X_train_cnn.shape[1], num_classes)
    model.fit(
        X_train_cnn, y_train,
        validation_split=0.1, epochs=20, batch_size=512,
        callbacks=[early_stop]
    )
    evaluate(model, X_test_cnn, y_test, le, "CNN (unweighted)")

    # --- Experiment 2: class-weighted ---
    # 'balanced' weights classes inversely proportional to their frequency,
    # so mistakes on rare classes (e.g. Web Attack, ~1,500 samples) count
    # far more toward the loss than mistakes on BENIGN (~2.3M samples).
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    model_weighted = build_model(X_train_cnn.shape[1], num_classes)  # fresh model
    model_weighted.fit(
        X_train_cnn, y_train,
        validation_split=0.1, epochs=20, batch_size=512,
        class_weight=class_weight_dict,
        callbacks=[early_stop]
    )
    evaluate(model_weighted, X_test_cnn, y_test, le, "CNN (class-weighted)")


if __name__ == "__main__":
    main()
