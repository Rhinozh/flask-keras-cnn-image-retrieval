import shutil
import time
import copy
import sklearn
from sklearn import neighbors
import datetime

def knn(training,comp,knn_n_neighbors,knn_model_name,knn_feature_table,target_image_feature):
    distances = []; indices = []
    if training == 1:
        # 训练KNN
        print('LOGOCOMP_INFO: begin KNN training...')
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        knn = neighbors.NearestNeighbors(n_neighbors=100, algorithm='kd_tree').fit(knn_feature_table)

        # 存储KNN模型
        sklearn.externals.joblib.dump(knn, knn_model_name)
        print('LOGOCOMP_INFO: %s training done!'%knn_model_name)
        print(datetime.datetime.now().strftime('%H:%M:%S'))
    elif comp == 1:
        knn = sklearn.externals.joblib.load(knn_model_name)
        distances, indices = knn.kneighbors(target_image_feature, n_neighbors = knn_n_neighbors)
    return(distances, indices)