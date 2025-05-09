import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras_cv_attention_models import swin_transformer_v2
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import DotProduct, RBF, RationalQuadratic,Matern

import warnings
warnings.filterwarnings("ignore")

# === Görüntüleri oku ve hazırla ===
def model_train(IMAGE_SIZE):
    disease_types = ['2', '3']
    data_dir = 'org_data2'
    train_data = []
    for defects_id, sp in enumerate(disease_types):
        for file in os.listdir(os.path.join(data_dir, sp)):
            train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
    train = pd.DataFrame(train_data, columns=['File', 'ID', 'Type'])
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    def read_image(filepath):
        return cv2.imread(os.path.join(data_dir, filepath))

    def resize_image(image, image_size):
        return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

    X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, file in tqdm(enumerate(train['File'].values), total=len(train)):
        image = read_image(file)
        if image is not None:
            X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
    X_Train = X_train / 255.0
    Y_train = to_categorical(train['ID'].values)
    return X_Train, Y_train

# === SWIN TRANSFORMER ===
Swin = swin_transformer_v2.SwinTransformerV2Large_window16(input_shape=(256, 256, 3), pretrained="imagenet22k")
Swin_model = Model(inputs=Swin.input, outputs=Swin.get_layer('avg_pool').output)

X_Train_256, Y_all = model_train(256)
features = Swin_model.predict(X_Train_256)
features_flat = features.reshape(features.shape[0], -1)
labels = np.argmax(Y_all, axis=1)

# === Train-test set ===
X_train, X_test, y_train, y_test = train_test_split(
    features_flat, labels, test_size=0.2, random_state=42, stratify=labels
)

# === Model pipeline ===
gpc = GaussianProcessClassifier()
pipeline = Pipeline([
    ('feature_selection', RFE(estimator=LogisticRegression())),
    ('classification', gpc)
])

# === GridSearchCV ayarları ===
feature_range = list(range(50, 1501, 5))
param_grid = {
    'feature_selection__n_features_to_select': feature_range,
    'classification__kernel': [
        1.0 * Matern(length_scale=1),
        1.0 * Matern(length_scale=1.5),
        1.0 * Matern(length_scale=2),
        1.0 * RationalQuadratic(length_scale=1, alpha=1.5),
        1.0 * RationalQuadratic(length_scale=0.5, alpha=1.5),
        1.0 * RationalQuadratic(length_scale=2, alpha=1.5),
        1.0 * RBF(length_scale=1.5),
        1.0 * RBF(length_scale=2.0),
        1.0 * DotProduct(0.5),
        1.0 * DotProduct(1),
        ....
    ],
    'classification__max_iter_predict': [5,10, 30, 50,100,200,500]  # Values between 5-500
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

grid.fit(X_train, y_train)

# === the best model
best_model = grid.best_estimator_
print("\nEn iyi parametreler:", grid.best_params_)
print(f"Train (CV) En İyi Doğruluk: {grid.best_score_:.4f}")

# === Test set evaluate
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# === Results
print("\n=== TEST PERFORMANSI ===")
print(f"Test Doğruluğu: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)
