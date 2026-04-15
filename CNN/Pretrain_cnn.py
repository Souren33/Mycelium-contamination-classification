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

min_count = min(len(p) for p in class_paths.values())
print(f"Samples per class (balanced): {min_count}")

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

#%% Model builder
def build_model():
    base = ResNet50(
        input_shape=(256, 256, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inputs  = tf.keras.Input(shape=(256, 256, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = Dense(128, activation='relu')(x)
    x       = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

#%% K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
all_y_true   = []
all_y_pred   = []
last_model   = None
last_test_ds = None

for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_paths)):
    print(f"\n{'='*40}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*40}")

    val_size  = int(0.15 * len(train_val_idx))
    train_idx = train_val_idx[val_size:]
    val_idx   = train_val_idx[:val_size]

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

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend((preds[:, 0] >= 0.5).astype(int))
        y_true.extend(labels.numpy())

    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

    report = skm.classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm     = skm.confusion_matrix(y_true, y_pred)
    fold_results.append(report)

    print(skm.classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(cm)

    last_model   = model
    last_test_ds = test_ds

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

#%% VISUALIZATION 1: Confusion Matrix (aggregated across all folds)
cm = skm.confusion_matrix(all_y_true, all_y_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, fontsize=12)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("True Label", fontsize=13)
ax.set_title("Confusion Matrix — All Folds Combined", fontsize=14)

cell_labels = [
    [f"TN\n{tn}", f"FP\n{fp}"],
    [f"FN\n{fn}", f"TP\n{tp}"]
]
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, cell_labels[i][j],
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print(f"\nTP={tp}  TN={tn}  FP={fp}  FN={fn}")

#%% VISUALIZATION 2: Single image + prediction
for images, labels in last_test_ds.take(1):
    sample_img   = images[0]
    sample_label = labels[0].numpy()

img_batch  = tf.expand_dims(sample_img, axis=0)
pred_prob  = last_model.predict(img_batch, verbose=0)[0][0]
pred_label = 1 if pred_prob >= 0.5 else 0

# Undo ResNet50 preprocessing to display correct colors
display_img = sample_img.numpy().copy()
display_img[..., 0] += 103.939
display_img[..., 1] += 116.779
display_img[..., 2] += 123.68
display_img = np.clip(display_img[..., ::-1] / 255.0, 0, 1)  # BGR→RGB

correct = pred_label == sample_label
color   = 'green' if correct else 'red'

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(display_img)
ax.set_title(
    f"True: {class_names[sample_label]}  |  "
    f"Pred: {class_names[pred_label]} ({pred_prob:.1%})",
    fontsize=12, color=color
)
ax.axis('off')
plt.tight_layout()
plt.savefig("prediction_example.png", dpi=150)
plt.show()

#%% VISUALIZATION 3: Feature maps from first conv layer
resnet_submodel  = last_model.get_layer("resnet50")
first_conv_layer = resnet_submodel.get_layer("conv1_conv")

activation_model = tf.keras.Model(
    inputs=resnet_submodel.input,
    outputs=first_conv_layer.output
)

feature_maps = activation_model.predict(img_batch, verbose=0)
feature_maps = feature_maps[0]  # (128, 128, 64)

n_maps = 32
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle(
    f"First Conv Layer Feature Maps\n"
    f"True: {class_names[sample_label]}  |  Pred: {class_names[pred_label]} ({pred_prob:.1%})",
    fontsize=13
)
for i, ax in enumerate(axes.flat):
    if i < n_maps:
        ax.imshow(feature_maps[:, :, i], cmap='viridis')
    ax.axis('off')
plt.tight_layout()
plt.savefig("feature_maps.png", dpi=150)
plt.show()

#%% VISUALIZATION 4: Grad-CAM
last_conv_layer_name = "conv5_block3_out"

grad_model = tf.keras.Model(
    inputs=last_model.input,
    outputs=[
        last_model.get_layer("resnet50").get_layer(last_conv_layer_name).output,
        last_model.output
    ]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_batch, training=False)
    tape.watch(conv_outputs)
    loss = predictions[:, 0]

grads       = tape.gradient(loss, conv_outputs)
pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
cam         = tf.reduce_sum(tf.multiply(pooled, conv_outputs[0]), axis=-1)
cam         = np.maximum(cam.numpy(), 0)
cam         = cam / (cam.max() + 1e-8)
cam_resized = tf.image.resize(cam[..., np.newaxis], IMG_SIZE).numpy()[..., 0]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f"Grad-CAM  |  True: {class_names[sample_label]}  "
    f"Pred: {class_names[pred_label]} ({pred_prob:.1%})",
    fontsize=13, color=color
)

axes[0].imshow(display_img)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(cam_resized, cmap='jet')
axes[1].set_title("Grad-CAM Heatmap")
axes[1].axis('off')

axes[2].imshow(display_img)
axes[2].imshow(cam_resized, cmap='jet', alpha=0.45)
axes[2].set_title("Overlay")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("gradcam.png", dpi=150)
plt.show()
# %%