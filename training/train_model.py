import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_mlp(input_dim, num_classes):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/landmarks.csv')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--model-out', default='model/sign_model.h5')
    ap.add_argument('--labels-out', default='model/sign_labels.json')
    args = ap.parse_args()

    df = pd.read_csv(args.csv, header=None)
    X = df.iloc[:, :-1].values.astype('float32')
    y = df.iloc[:, -1].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_enc, num_classes)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_enc)

    model = build_mlp(X.shape[1], num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cbs = [
        callbacks.ModelCheckpoint(args.model_out, monitor='val_accuracy', save_best_only=True, mode='max'),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ]

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=32, callbacks=cbs)
    labels = list(le.classes_)
    with open(args.labels_out, 'w') as f:
        json.dump(labels, f, indent=2)
    print('Training complete. Model saved to', args.model_out)

if __name__ == '__main__':
    main()
