"""
Preprocessing dan normalisasi seluruh dataset PlantVillage untuk Custom CNN.

Script ini tidak menyimpan semua gambar hasil normalisasi ke file baru.
Normalisasi dilakukan di pipeline TensorFlow dan diverifikasi dengan statistik pixel.
"""

import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as error:
    print(f"\nERROR: Dependency belum terinstall: {error.name}")
    print("Jalankan perintah berikut di terminal VS Code:")
    print("  python -m pip install -r requirements.txt\n")
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    print(f"\nERROR: Dependency belum terinstall: {error.name}")
    print("Jalankan perintah berikut di terminal VS Code:")
    print("  python -m pip install -r requirements.txt\n")
    sys.exit(1)

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("\nERROR: TensorFlow belum terinstall.")
    print("Jalankan perintah berikut di terminal VS Code:")
    print("  python -m pip install -r requirements.txt\n")
    sys.exit(1)

try:
    from sklearn.preprocessing import LabelEncoder
except ModuleNotFoundError:
    print("\nERROR: scikit-learn belum terinstall.")
    print("Jalankan perintah berikut di terminal VS Code:")
    print("  python -m pip install -r requirements.txt\n")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "plantvillage dataset"
OUTPUT_DIR = BASE_DIR / "outputs" / "normalization_all"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_STATE = 42
SAMPLE_PREVIEW = 12

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
EXPECTED_SOURCES = {"color", "grayscale", "segmented"}


# ============================================================================
# DATASET SCAN AND LABELING
# ============================================================================

def scan_dataset():
    """Baca semua gambar dan buat metadata dasar dataset."""
    print("\n[1/10] scanning dataset...")

    if not DATASET_DIR.exists():
        print(f"\nERROR: Dataset folder tidak ditemukan: {DATASET_DIR}")
        sys.exit(1)

    rows = []
    for image_path in DATASET_DIR.rglob("*"):
        if not image_path.is_file() or image_path.suffix not in VALID_EXTENSIONS:
            continue

        relative_parts = image_path.relative_to(DATASET_DIR).parts
        if len(relative_parts) < 2:
            continue

        source = relative_parts[0]
        label = image_path.parent.name

        rows.append({
            "image_path": str(image_path),
            "label": label,
            "source": source,
            "filename": image_path.name,
            "extension": image_path.suffix,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("\nERROR: Total gambar 0. Periksa isi folder dataset.")
        sys.exit(1)

    unknown_sources = sorted(set(df["source"]) - EXPECTED_SOURCES)
    if unknown_sources:
        print(f"\nWARNING: Source selain color/grayscale/segmented terdeteksi: {unknown_sources}")

    print(f"Total gambar: {len(df)}")
    print(f"Jumlah kelas: {df['label'].nunique()}")
    print(f"Daftar source: {sorted(df['source'].unique())}")
    print("\nJumlah gambar per source:")
    print(df["source"].value_counts().sort_index())
    print("\nJumlah gambar per kelas:")
    print(df["label"].value_counts().sort_index())
    print("\nContoh 5 data:")
    print(df.head(5))

    return df


def encode_labels(df):
    """Encode nama kelas menjadi label_id dan simpan mapping."""
    print("\n[2/10] encoding labels...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_encoder = LabelEncoder()
    df = df.copy()
    df["label_id"] = label_encoder.fit_transform(df["label"])

    label_mapping = pd.DataFrame({
        "label_id": range(len(label_encoder.classes_)),
        "label_name": label_encoder.classes_,
    })

    mapping_csv = OUTPUT_DIR / "label_mapping.csv"
    mapping_json = OUTPUT_DIR / "label_mapping.json"

    label_mapping.to_csv(mapping_csv, index=False)
    mapping_dict = {
        str(int(row.label_id)): row.label_name
        for row in label_mapping.itertuples(index=False)
    }
    with open(mapping_json, "w", encoding="utf-8") as file:
        json.dump(mapping_dict, file, indent=4, ensure_ascii=False)

    print(f"Label mapping CSV: {mapping_csv}")
    print(f"Label mapping JSON: {mapping_json}")

    return df, label_encoder


# ============================================================================
# TENSORFLOW NORMALIZATION PIPELINE
# ============================================================================

def load_and_normalize_image(image_path, label_id):
    """Load gambar dengan TensorFlow dan normalisasi pixel ke rentang 0-1."""
    image = tf.io.read_file(image_path)

    # Decode gambar menjadi RGB
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    # Resize gambar ke 224x224
    image = tf.image.resize(image, IMAGE_SIZE)

    # Cast ke float32
    image = tf.cast(image, tf.float32)

    # Normalisasi pixel dari 0-255 menjadi 0-1 untuk Custom CNN
    image = image / 255.0

    return image, label_id


def create_tf_dataset(df):
    """Buat TensorFlow Dataset dari DataFrame metadata."""
    print("\n[3/10] creating tensorflow dataset...")

    image_paths = df["image_path"].astype(str).values
    labels = df["label_id"].values

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(
        load_and_normalize_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def calculate_normalization_stats(dataset, max_batches=30):
    """Hitung statistik pixel setelah normalisasi dari beberapa batch."""
    print("\n[4/10] calculating normalization stats...")

    min_pixel = np.inf
    max_pixel = -np.inf
    total_sum = 0.0
    total_sum_sq = 0.0
    total_pixels = 0
    batches_used = 0

    for images, _ in dataset.take(max_batches):
        images_np = images.numpy()
        min_pixel = min(min_pixel, float(images_np.min()))
        max_pixel = max(max_pixel, float(images_np.max()))
        total_sum += float(images_np.sum())
        total_sum_sq += float(np.square(images_np).sum())
        total_pixels += int(images_np.size)
        batches_used += 1

    mean_pixel = total_sum / total_pixels
    variance = (total_sum_sq / total_pixels) - (mean_pixel ** 2)
    std_pixel = float(np.sqrt(max(variance, 0.0)))

    stats = {
        "min_pixel": float(min_pixel),
        "max_pixel": float(max_pixel),
        "mean_pixel": float(mean_pixel),
        "std_pixel": std_pixel,
        "batches_used": int(batches_used),
        "normalization_valid": bool(0.0 <= min_pixel <= max_pixel <= 1.0),
    }

    stats_file = OUTPUT_DIR / "normalization_stats.json"
    with open(stats_file, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4)

    print(f"Pixel min: {stats['min_pixel']:.6f}")
    print(f"Pixel max: {stats['max_pixel']:.6f}")
    print(f"Pixel mean: {stats['mean_pixel']:.6f}")
    print(f"Pixel std: {stats['std_pixel']:.6f}")
    print(f"Stats file: {stats_file}")

    if not stats["normalization_valid"]:
        print("WARNING: Pixel setelah normalisasi tidak seluruhnya berada di rentang 0 sampai 1.")

    return stats


# ============================================================================
# REPORTS
# ============================================================================

def save_label_reports(df):
    """Simpan metadata dan distribusi label/source."""
    print("\n[5/10] saving label reports...")

    metadata_cols = ["image_path", "label", "label_id", "source", "filename", "extension"]
    metadata_file = OUTPUT_DIR / "normalized_dataset_metadata.csv"
    class_file = OUTPUT_DIR / "class_distribution_all.csv"
    source_file = OUTPUT_DIR / "source_distribution_all.csv"
    class_by_source_file = OUTPUT_DIR / "class_by_source_distribution.csv"

    df[metadata_cols].to_csv(metadata_file, index=False)

    class_distribution = (
        df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
    )
    class_distribution.to_csv(class_file, index=False)

    source_distribution = (
        df["source"]
        .value_counts()
        .rename_axis("source")
        .reset_index(name="count")
        .sort_values("source")
    )
    source_distribution.to_csv(source_file, index=False)

    class_by_source = pd.pivot_table(
        df,
        index="label",
        columns="source",
        values="image_path",
        aggfunc="count",
        fill_value=0
    )
    class_by_source.to_csv(class_by_source_file)

    print(f"Metadata: {metadata_file}")
    print(f"Class distribution: {class_file}")
    print(f"Source distribution: {source_file}")
    print(f"Class by source distribution: {class_by_source_file}")

    return {
        "metadata_file": metadata_file,
        "class_distribution_file": class_file,
        "source_distribution_file": source_file,
        "class_by_source_file": class_by_source_file,
    }


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_class_distribution(df):
    """Buat bar chart jumlah gambar per kelas."""
    print("\n[6/10] creating class distribution diagram...")

    counts = df["label"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(16, 9))
    plt.bar(range(len(counts)), counts.values, color="#2f80ed")
    plt.title("Class Distribution - All Sources")
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.xticks(range(len(counts)), counts.index, rotation=90, fontsize=8)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "class_distribution_all.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    return output_file


def plot_source_distribution(df):
    """Buat bar chart jumlah gambar per source."""
    print("\n[7/10] creating source distribution diagram...")

    counts = df["source"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, color=["#2f80ed", "#27ae60", "#f2994a"][:len(counts)])
    plt.title("Source Distribution")
    plt.xlabel("Source")
    plt.ylabel("Image Count")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "source_distribution_all.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    return output_file


def plot_class_by_source(df):
    """Buat heatmap matplotlib untuk distribusi kelas per source."""
    print("\n[8/10] creating class by source heatmap...")

    pivot = pd.pivot_table(
        df,
        index="label",
        columns="source",
        values="image_path",
        aggfunc="count",
        fill_value=0
    )

    source_order = [source for source in ["color", "grayscale", "segmented"] if source in pivot.columns]
    extra_sources = [source for source in pivot.columns if source not in source_order]
    pivot = pivot[source_order + extra_sources]

    fig_height = max(8, len(pivot) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

    ax.set_title("Class by Source Distribution")
    ax.set_xlabel("Source")
    ax.set_ylabel("Class")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                int(pivot.iloc[row_idx, col_idx]),
                ha="center",
                va="center",
                fontsize=6,
                color="black"
            )

    fig.colorbar(image, ax=ax, label="Image Count")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "class_by_source_heatmap.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    return output_file


def plot_preprocessing_flow_diagram():
    """Buat diagram alur preprocessing gambar."""
    print("\n[9/10] creating preprocessing flow diagram...")

    steps = [
        "Raw Image",
        "Decode RGB",
        "Resize 224x224",
        "Cast float32",
        "Normalize pixel 0-1",
        "Label Encoding",
        "Ready for CNN",
    ]

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.axis("off")

    y_positions = np.linspace(0.92, 0.08, len(steps))
    for idx, (step, y_pos) in enumerate(zip(steps, y_positions)):
        ax.text(
            0.5,
            y_pos,
            step,
            ha="center",
            va="center",
            fontsize=13,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#f2f6fc",
                "edgecolor": "#2f80ed",
                "linewidth": 1.5,
            },
        )

        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[idx + 1] + 0.045),
                xytext=(0.5, y_pos - 0.045),
                arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#333333"},
            )

    plt.tight_layout()

    output_file = OUTPUT_DIR / "preprocessing_flow_diagram.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    return output_file


def preview_normalized_samples(df):
    """Simpan preview 12 gambar yang sudah dinormalisasi."""
    print("\n[10/10] creating normalized sample preview...")

    sample_count = min(SAMPLE_PREVIEW, len(df))
    sample_df = df.sample(n=sample_count, random_state=RANDOM_STATE)

    cols = 4
    rows = int(np.ceil(sample_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, row in zip(axes, sample_df.itertuples(index=False)):
        image_tensor, label_id = load_and_normalize_image(
            tf.constant(row.image_path),
            tf.constant(row.label_id)
        )
        image_np = image_tensor.numpy()

        ax.imshow(np.clip(image_np, 0.0, 1.0))
        ax.set_title(
            f"{row.label}\n{row.source}\nlabel_id: {int(label_id.numpy())}",
            fontsize=8
        )
        ax.axis("off")

    plt.tight_layout()

    output_file = OUTPUT_DIR / "normalized_sample_preview.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    return output_file


# ============================================================================
# SUMMARY
# ============================================================================

def save_summary(df, stats, report_files):
    """Simpan ringkasan preprocessing dan normalisasi."""
    print("\nSaving summary...")

    summary = {
        "total_images": int(len(df)),
        "total_classes": int(df["label"].nunique()),
        "sources": sorted(df["source"].unique().tolist()),
        "image_size": list(IMAGE_SIZE),
        "normalization_method": "image / 255.0 for Custom CNN",
        "pixel_range_after_normalization": {
            "min_pixel": stats["min_pixel"],
            "max_pixel": stats["max_pixel"],
        },
        "output_directory": str(OUTPUT_DIR),
        "label_mapping_file": str(OUTPUT_DIR / "label_mapping.csv"),
        "metadata_file": str(report_files["metadata_file"]),
    }

    summary_file = OUTPUT_DIR / "normalization_summary.json"
    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return summary_file


def main():
    """Jalankan seluruh proses preprocessing dataset."""
    print("=" * 75)
    print("NORMALIZE PLANTVILLAGE DATASET - COLOR, GRAYSCALE, SEGMENTED")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = scan_dataset()
    df, _ = encode_labels(df)
    dataset = create_tf_dataset(df)
    stats = calculate_normalization_stats(dataset)
    report_files = save_label_reports(df)

    class_plot = plot_class_distribution(df)
    source_plot = plot_source_distribution(df)
    heatmap_plot = plot_class_by_source(df)
    flow_plot = plot_preprocessing_flow_diagram()
    preview_plot = preview_normalized_samples(df)
    summary_file = save_summary(df, stats, report_files)

    print("\ncompleted")
    print("=" * 75)
    print("RINGKASAN NORMALISASI")
    print("=" * 75)
    print(f"Total gambar: {len(df)}")
    print(f"Jumlah kelas: {df['label'].nunique()}")
    print(f"Jumlah source: {df['source'].nunique()}")
    print(f"Pixel min setelah normalisasi: {stats['min_pixel']:.6f}")
    print(f"Pixel max setelah normalisasi: {stats['max_pixel']:.6f}")
    print(f"Lokasi file label mapping: {OUTPUT_DIR / 'label_mapping.csv'}")
    print(f"Lokasi diagram tersimpan: {OUTPUT_DIR}")
    print("\nFile diagram:")
    print(f"  - {class_plot}")
    print(f"  - {source_plot}")
    print(f"  - {heatmap_plot}")
    print(f"  - {flow_plot}")
    print(f"  - {preview_plot}")
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()
