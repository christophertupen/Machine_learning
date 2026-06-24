"""
Train dan bandingkan 3 model image classification PlantVillage:
Custom CNN, EfficientNetB0, dan EfficientNetB1.
"""

import json
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    print(f"\nERROR: Dependency belum terinstall: {error.name}")
    print("Jalankan: python -m pip install -r requirements.txt\n")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError:
    print("\nERROR: Install TensorFlow terlebih dahulu.")
    print("Jalankan: python -m pip install -r requirements.txt\n")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
except ModuleNotFoundError as error:
    print(f"\nERROR: Dependency belum terinstall: {error.name}")
    print("Jalankan: python -m pip install -r requirements.txt\n")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
METADATA_FILE = BASE_DIR / "outputs" / "normalization_all" / "normalized_dataset_metadata.csv"
LABEL_MAPPING_FILE = BASE_DIR / "outputs" / "normalization_all" / "label_mapping.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "three_models_comparison_v1"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.1
SOURCE_FILTER = "color"


# ============================================================================
# DATASET HANDLING
# ============================================================================

def load_metadata():
    """Load metadata dan label mapping hasil normalize_dataset_all.py."""
    print("\nloading metadata...")

    if not METADATA_FILE.exists() or not LABEL_MAPPING_FILE.exists():
        print("\nERROR: File metadata normalisasi tidak ditemukan.")
        print("Jalankan normalize_dataset_all.py terlebih dahulu.\n")
        sys.exit(1)

    df = pd.read_csv(METADATA_FILE)
    label_mapping = pd.read_csv(LABEL_MAPPING_FILE)

    required_columns = {"image_path", "label", "label_id", "source", "filename"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"\nERROR: Kolom metadata kurang: {sorted(missing_columns)}")
        sys.exit(1)

    if df.empty:
        print("\nERROR: Total gambar 0.")
        sys.exit(1)

    return df, label_mapping


def filter_dataset(df):
    """Filter source dan sampling stratified jika SAMPLE_FRACTION < 1.0."""
    print("\nfiltering dataset...")

    df = df.copy()
    if SOURCE_FILTER == "all":
        print("WARNING: Menggunakan color + grayscale + segmented bisa menyebabkan data leakage jika gambar yang sama muncul dalam versi berbeda.")
    elif SOURCE_FILTER in {"color", "grayscale", "segmented"}:
        df = df[df["source"] == SOURCE_FILTER].copy()
    else:
        print(f"\nERROR: SOURCE_FILTER tidak valid: {SOURCE_FILTER}")
        print('Gunakan "color", "grayscale", "segmented", atau "all".')
        sys.exit(1)

    if df.empty:
        print("\nERROR: Total gambar 0 setelah filter.")
        sys.exit(1)

    if SAMPLE_FRACTION < 1.0:
        sampled_indices = []
        for _, group in df.groupby("label_id"):
            sample_size = max(1, int(round(len(group) * SAMPLE_FRACTION)))
            sampled_indices.extend(
                group.sample(n=sample_size, random_state=RANDOM_STATE).index.tolist()
            )
        df = df.loc[sampled_indices].sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Total data setelah filter: {len(df)}")
    print(f"Jumlah kelas: {df['label_id'].nunique()}")
    print("Jumlah data per source:")
    print(df["source"].value_counts().sort_index())

    return df.reset_index(drop=True)


def split_dataset(df):
    """Split dataset menjadi train 80%, validation 10%, test 10%."""
    print("\nsplitting dataset...")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_id"],
    )

    print(f"Jumlah data train: {len(train_df)}")
    print(f"Jumlah data validation: {len(val_df)}")
    print(f"Jumlah data test: {len(test_df)}")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_label_info(label_mapping, df):
    """Buat mapping label_id ke nama label sesuai data yang dipakai."""
    label_mapping = label_mapping.copy()
    if "label_name" not in label_mapping.columns:
        label_mapping = label_mapping.rename(columns={"label": "label_name"})

    id_to_label = dict(zip(label_mapping["label_id"], label_mapping["label_name"]))
    used_label_ids = sorted(df["label_id"].unique())
    class_names = [id_to_label[label_id] for label_id in used_label_ids]

    return id_to_label, class_names


# ============================================================================
# PREPROCESSING PIPELINES
# ============================================================================

def load_image_for_cnn(image_path, label_id):
    """Load gambar dan normalisasi 0-1 untuk Custom CNN."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # Normalisasi untuk Custom CNN: pixel 0-255 menjadi 0-1
    image = image / 255.0

    return image, label_id


def load_image_for_efficientnet(image_path, label_id):
    """Load gambar dan gunakan preprocessing bawaan EfficientNet."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # Preprocessing bawaan EfficientNet
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image, label_id


def create_tf_dataset(df, model_type, shuffle=False):
    """Buat tf.data pipeline sesuai tipe model."""
    print(f"creating dataset pipeline for {model_type}...")

    image_paths = df["image_path"].astype(str).values
    labels = df["label_id"].values

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if model_type == "cnn":
        loader = load_image_for_cnn
    elif model_type == "efficientnet":
        loader = load_image_for_efficientnet
    else:
        raise ValueError(f"model_type tidak valid: {model_type}")

    dataset = dataset.map(loader, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=RANDOM_STATE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def compute_class_weights(train_df):
    """Hitung class weight untuk dataset imbalanced."""
    classes = np.sort(train_df["label_id"].unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label_id"].values,
    )
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


# ============================================================================
# MODELS
# ============================================================================

def build_custom_cnn(num_classes):
    """Build model Custom CNN."""
    model = keras.Sequential([
        keras.Input(shape=IMAGE_SIZE + (3,)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnetb0(num_classes):
    """Build EfficientNetB0 transfer learning."""
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = keras.Sequential([
        keras.Input(shape=IMAGE_SIZE + (3,)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnetb1(num_classes):
    """Build EfficientNetB1 transfer learning."""
    base_model = tf.keras.applications.EfficientNetB1(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = keras.Sequential([
        keras.Input(shape=IMAGE_SIZE + (3,)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_single_model(model, model_name, train_ds, val_ds, class_weight):
    """Train satu model dan simpan history serta model final."""
    print(f"\ntraining {model_name}...")

    model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / f"{model_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    training_time_minutes = (time.time() - start_time) / 60.0

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(model_output_dir / "training_history.csv", index=False)

    model.save(MODEL_DIR / f"{model_name}_final.keras")

    return history, training_time_minutes


def evaluate_single_model(model, model_name, test_ds, test_df, id_to_label, training_time_minutes):
    """Evaluasi satu model pada test set dan simpan hasilnya."""
    print(f"evaluating {model_name}...")

    model_output_dir = OUTPUT_DIR / model_name
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_df["label_id"].values

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(model_output_dir / "classification_report.csv")

    final_model_file = MODEL_DIR / f"{model_name}_final.keras"
    model_file_size_mb = final_model_file.stat().st_size / (1024 * 1024)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "training_time_minutes": float(training_time_minutes),
        "model_parameters": int(model.count_params()),
        "model_file_size_mb": float(model_file_size_mb),
    }
    with open(model_output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    sample_predictions = test_df[["image_path", "label", "label_id", "source", "filename"]].copy()
    sample_predictions["predicted_label_id"] = y_pred
    sample_predictions["predicted_label"] = [id_to_label.get(int(label_id), str(label_id)) for label_id in y_pred]
    sample_predictions["confidence"] = np.max(probabilities, axis=1)
    sample_predictions.to_csv(model_output_dir / "sample_predictions.csv", index=False)

    return metrics, y_true, y_pred


def save_training_plot(history, model_name):
    """Simpan plot accuracy dan loss training."""
    model_output_dir = OUTPUT_DIR / model_name
    history_data = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history_data.get("accuracy", []), label="train_accuracy")
    axes[0].plot(history_data.get("val_accuracy", []), label="val_accuracy")
    axes[0].set_title(f"{model_name} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history_data.get("loss", []), label="train_loss")
    axes[1].plot(history_data.get("val_loss", []), label="val_loss")
    axes[1].set_title(f"{model_name} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(model_output_dir / "training_plot.png", dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, model_name, class_names):
    """Simpan confusion matrix sebagai gambar matplotlib."""
    model_output_dir = OUTPUT_DIR / model_name
    matrix = confusion_matrix(y_true, y_pred)

    fig_size = max(10, len(class_names) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(model_output_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def save_model_explanation():
    """Simpan file penjelasan model dan hyperparameter untuk presentasi."""
    print("saving presentation explanation...")

    explanation = f"""# Model Explanation

## 1. Buat Model Baru

Project ini membuat 3 model image classification: Custom CNN, EfficientNetB0, dan EfficientNetB1.
Ketiganya termasuk supervised learning, multiclass image classification, dan CNN-based model.
Dataset yang digunakan adalah PlantVillage dengan 38 kelas penyakit/tanaman.

## 2. Hyperparameter

| Hyperparameter | Nilai |
|---|---|
| Image size | {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} |
| Batch size | {BATCH_SIZE} |
| Epochs | {EPOCHS} |
| Optimizer | Adam |
| Learning rate | {LEARNING_RATE} |
| Loss function | sparse_categorical_crossentropy |
| Activation output | softmax |
| Dropout | Custom CNN 0.5, EfficientNet 0.3 |
| Class weight | Ya, balanced berdasarkan data train |
| Sample fraction | {SAMPLE_FRACTION} |
| Train/val/test split | 80% / 10% / 10% |

## 3. Cara Kerja / Algoritma Model

Custom CNN menerima gambar, convolution mengambil fitur visual, pooling mengecilkan dimensi, dense layer melakukan klasifikasi, dan softmax menghasilkan probabilitas kelas.

EfficientNetB0 menggunakan model pretrained ImageNet sebagai feature extractor. Base model dibekukan, lalu classifier baru ditambahkan untuk klasifikasi PlantVillage.

EfficientNetB1 mirip EfficientNetB0, tetapi kapasitas model lebih besar sehingga dapat menangkap fitur visual yang lebih kompleks.

Perbedaan utama: Custom CNN belajar fitur dari awal, sedangkan EfficientNet memakai transfer learning dari ImageNet. Perbandingan 3 model membantu melihat trade-off antara akurasi, waktu training, dan ukuran model.

## 4. Kesimpulan Sementara

- Model terbaik berdasarkan test accuracy: isi setelah training selesai.
- Model terbaik berdasarkan macro F1-score: isi setelah training selesai.
- Model tercepat berdasarkan training time: isi setelah training selesai.
- Model paling ringan berdasarkan model size: isi setelah training selesai.
"""

    hyperparameters = f"""# Hyperparameters

- Image size: ukuran gambar input yang dipakai model, yaitu {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} pixel.
- Batch size: jumlah gambar yang diproses model dalam satu langkah training.
- Epoch: satu putaran penuh model melihat seluruh data train.
- Learning rate: besar langkah optimizer saat memperbarui bobot model.
- Optimizer: algoritma untuk memperbaiki bobot model; script ini memakai Adam.
- Dropout: teknik mematikan sebagian neuron saat training agar model tidak mudah overfitting.
- Loss function: fungsi yang mengukur kesalahan prediksi; script ini memakai sparse_categorical_crossentropy.
- Activation softmax: mengubah output akhir menjadi probabilitas untuk setiap kelas.
- Class weight: bobot tambahan untuk kelas yang datanya lebih sedikit agar training lebih seimbang.
- Sample fraction: persentase data yang dipakai untuk percobaan awal.
- Train/validation/test split: pembagian data untuk belajar, validasi saat training, dan evaluasi akhir.
"""

    with open(OUTPUT_DIR / "MODEL_EXPLANATION.md", "w", encoding="utf-8") as file:
        file.write(explanation)

    with open(OUTPUT_DIR / "HYPERPARAMETERS.md", "w", encoding="utf-8") as file:
        file.write(hyperparameters)


# ============================================================================
# COMPARISON
# ============================================================================

def compare_all_models(results):
    """Simpan tabel dan diagram perbandingan semua model."""
    print("\ncreating comparison report...")

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plot_specs = [
        ("test_accuracy", "Test Accuracy", "comparison_accuracy.png"),
        ("f1_macro", "Macro F1-score", "comparison_f1_score.png"),
        ("training_time_minutes", "Training Time (minutes)", "comparison_training_time.png"),
        ("model_file_size_mb", "Model Size (MB)", "comparison_model_size.png"),
    ]

    for column, title, filename in plot_specs:
        plt.figure(figsize=(8, 5))
        plt.bar(comparison_df["model_name"], comparison_df[column], color=["#2f80ed", "#27ae60", "#f2994a"])
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(title)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close()

    return comparison_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Jalankan training dan comparison 3 model."""
    print("=" * 75)
    print("TRAIN 3 MODELS COMPARISON - CUSTOM CNN, EFFICIENTNETB0, EFFICIENTNETB1")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, label_mapping = load_metadata()
    df = filter_dataset(df)
    train_df, val_df, test_df = split_dataset(df)
    id_to_label, class_names = create_label_info(label_mapping, df)
    num_classes = len(class_names)
    class_weight = compute_class_weights(train_df)

    model_configs = [
        {
            "model_name": "custom_cnn",
            "display_name": "Custom CNN",
            "model_type": "cnn",
            "builder": build_custom_cnn,
            "preprocessing_method": "image / 255.0",
            "notes": "Custom CNN trained from scratch",
        },
        {
            "model_name": "efficientnetb0",
            "display_name": "EfficientNetB0",
            "model_type": "efficientnet",
            "builder": build_efficientnetb0,
            "preprocessing_method": "tf.keras.applications.efficientnet.preprocess_input",
            "notes": "EfficientNetB0 transfer learning, frozen ImageNet base",
        },
        {
            "model_name": "efficientnetb1",
            "display_name": "EfficientNetB1",
            "model_type": "efficientnet",
            "builder": build_efficientnetb1,
            "preprocessing_method": "tf.keras.applications.efficientnet.preprocess_input",
            "notes": "EfficientNetB1 transfer learning, frozen ImageNet base",
        },
    ]

    comparison_results = []

    for config in model_configs:
        model_name = config["model_name"]
        display_name = config["display_name"]

        try:
            print(f"\ntraining {display_name}")
            train_ds = create_tf_dataset(train_df, config["model_type"], shuffle=True)
            val_ds = create_tf_dataset(val_df, config["model_type"], shuffle=False)
            test_ds = create_tf_dataset(test_df, config["model_type"], shuffle=False)

            model = config["builder"](num_classes)
            history, training_time = train_single_model(
                model,
                model_name,
                train_ds,
                val_ds,
                class_weight,
            )
            save_training_plot(history, model_name)

            print(f"evaluating {display_name}")
            metrics, y_true, y_pred = evaluate_single_model(
                model,
                model_name,
                test_ds,
                test_df,
                id_to_label,
                training_time,
            )
            save_confusion_matrix(y_true, y_pred, model_name, class_names)

            comparison_results.append({
                "model_name": display_name,
                "model_type": config["model_type"],
                "test_accuracy": metrics["test_accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "training_time_minutes": metrics["training_time_minutes"],
                "model_parameters": metrics["model_parameters"],
                "model_file_size_mb": metrics["model_file_size_mb"],
                "preprocessing_method": config["preprocessing_method"],
                "notes": config["notes"],
            })

        except Exception as error:
            print(f"\nERROR: Training model {display_name} gagal.")
            print(f"Detail error: {error}")
            raise

    comparison_df = compare_all_models(comparison_results)
    save_model_explanation()

    print("\ncompleted")
    print("=" * 75)
    print("RINGKASAN PERBANDINGAN MODEL")
    print("=" * 75)
    print(comparison_df[[
        "model_name",
        "test_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "training_time_minutes",
        "model_file_size_mb",
    ]])
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Model folder: {MODEL_DIR}")


if __name__ == "__main__":
    main()
