#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import ResNet50
import pathlib
import sklearn.metrics as skm
from sklearn.model_selection import KFold

#%% Load file paths per class
data_dir = pathlib.Path("resized_datasets/")

class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
class_names = [d.name for d in class_dirs]
print(f"Classes: {class_names}")

rng = np.random.default_rng(42)

def get_shuffled_paths(class_dir):
    paths = np.array(list(class_dir.glob("*")))
    rng.shuffle(paths)
    return paths

class_paths = {d.name: get_shuffled_paths(d) for d in class_dirs}

# Balance classes
min_count = min(len(p) for p in class_paths.values())
print(f"Samples per class (balanced): {min_count}")

# Build full balanced arrays
all_paths, all_labels = [], []
for label_idx, name in enumerate(class_names):
    paths = class_paths[name][:min_count]
    all_paths.extend([str(p) for p in paths])
    all_labels.extend([label_idx] * min_count)

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)

#%% Dataset builder
preprocess = tf.keras.applications.resnet50.preprocess_input
IMG_SIZE   = (256, 256)
BATCH_SIZE = 32

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess(img)
    return img, label

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

#%% Model builder (called fresh each fold)
def build_model():
    base = ResNet50(
        input_shape=(256, 256, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

#%% K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_paths)):
    print(f"\n{'='*40}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*40}")

    # Further split train_val into train (87.5%) and val (12.5%)
    # giving roughly 70 / 12.5 / 17.5 of total per fold
    val_size    = int(0.15 * len(train_val_idx))
    train_idx   = train_val_idx[val_size:]
    val_idx     = train_val_idx[:val_size]

    train_ds = make_dataset(all_paths[train_idx], all_labels[train_idx], shuffle=True)
    val_ds   = make_dataset(all_paths[val_idx],   all_labels[val_idx])
    test_ds  = make_dataset(all_paths[test_idx],  all_labels[test_idx])

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    model = build_model()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend((preds[:, 0] >= 0.5).astype(int))
        y_true.extend(labels.numpy())

    report = skm.classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm     = skm.confusion_matrix(y_true, y_pred)

    fold_results.append(report)

    print(skm.classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(cm)

#%% Summarize across folds
print(f"\n{'='*40}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*40}")

for cls in class_names + ['accuracy']:
    if cls == 'accuracy':
        scores = [r['accuracy'] for r in fold_results]
        print(f"Accuracy      — mean: {np.mean(scores):.3f}  std: {np.std(scores):.3f}")
    else:
        f1s = [r[cls]['f1-score'] for r in fold_results]
        print(f"{cls:>14} F1 — mean: {np.mean(f1s):.3f}  std: {np.std(f1s):.3f}")