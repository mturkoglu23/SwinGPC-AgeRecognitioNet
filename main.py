import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split


def model_train(IMAGE_SIZE):   
    disease_types=['2', '3', '4','5','6+']
    data_dir = 'org_data'
    train_dir = os.path.join(data_dir)
    
    train_data = []
    for defects_id, sp in enumerate(disease_types):
        for file in os.listdir(os.path.join(train_dir, sp)):
            train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
            
    train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
    
    SEED = 42
    train = train.sample(frac=1, random_state=SEED) 
    train.index = np.arange(len(train)) # Reset indices
    
    # IMAGE_SIZE = 256
    def read_image(filepath):
        return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
    # Resize image to target size
    def resize_image(image, image_size):
        return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
    
    X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, file in tqdm(enumerate(train['File'].values)):
        image = read_image(file)
        if image is not None:
            X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
    # Normalize the data
    X_Train = X_train / 255.
    # print('Train Shape: {}'.format(X_Train.shape))
    
    Y_train = train['DiseaseID'].values
    Y_train = to_categorical(Y_train)
    return X_Train,Y_train



from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model,Sequential
from keras.optimizers import adam_v2
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from tensorflow.keras.layers import Input
from keras_cv_attention_models import swin_transformer_v2


Swin = swin_transformer_v2.SwinTransformerV2Large_window16(input_shape=(256, 256, 3),pretrained="imagenet22k")
Swin_model = Model(inputs=Swin.input, outputs=Swin.get_layer('avg_pool').output)


X_Train_256,Y_train=model_train(256)

feature_extractor=Swin_model.predict(X_Train_256)
Swin_feats=feature_extractor.reshape(feature_extractor.shape[0],-1)

Y_trainn1 = np.argmax(Y_train, axis=1).reshape(-1,1)
 

from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

estimator = LogisticRegression()
k = 200
rfe = RFE(estimator,n_features_to_select=k)
X_new = rfe.fit_transform(Swin_feats,Y_trainn1)
selected_features = rfe.get_support(indices=True)


from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RationalQuadratic,DotProduct,RBF,WhiteKernel,Matern,ExpSineSquared
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix

param_grid = {
    'kernel': [1.0 * DotProduct(0.2),1.0 * DotProduct(0.5),1.0 * DotProduct(1),1.0 * DotProduct(2),
               1.0 * RBF(length_scale=0.5),1.0 * RBF(length_scale=1.0),1.0 * RBF(length_scale=1.5),1.0 * RBF(length_scale=2.0),
               1.0 * RationalQuadratic(1),1.0 * RationalQuadratic(1.5),1.0 * RationalQuadratic(2)],
    'max_iter_predict': [5, 10, 15,20,30, 50, 75,00],
}

# GridSearchCV kullanarak en iyi parametreleri bulma
X=X_new
y=np.ravel(Y_trainn1)
clf = GridSearchCV(GaussianProcessClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1)
clf.fit(X, y)

best_params = clf.best_params_
best_accuracy = clf.best_score_

print(best_params)
print(best_accuracy)

best_classifier = clf.best_estimator_

# Boş bir confusion matrix oluşturalım
overall_confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]

# 5 kat çapraz doğrulama işlemi
for train_idx, test_idx in StratifiedKFold(n_splits=5).split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    best_classifier.fit(X_train, y_train)
    y_pred = best_classifier.predict(X_test)
    
    # Her katman için confusion matrix hesaplayalım ve toplam confusion matrix'i güncelleyelim
    cm = confusion_matrix(y_test, y_pred)
    overall_confusion_matrix += cm

# Toplam confusion matrix'i yazdıralım
print("Overall Confusion Matrix:")
print(overall_confusion_matrix)

