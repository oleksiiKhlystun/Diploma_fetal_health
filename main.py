import os
import PySimpleGUI as sg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import seaborn as sns
import matplotlib.pyplot as plt

sg.theme('Default1')
layout = [
    [sg.FileBrowse('0.Browse fetal_health.csv', target="-PATH-", file_types=[("csv", '*.csv')]), sg.I(key="-PATH-")],
    [sg.Button('1.Preprocessing'), sg.Button('Open Heatmap Picture')],
    [sg.Output(size=(65, 27), key='-OUTPUT-')],
    [sg.Button('2.RandomForest_SVM')],
    [sg.Output(size=(65, 5), key='-OUTPUT-RF_SVM-')],
    [sg.Button('3.PCA+RandomForest_SVM'), sg.Button('Open PCA Picture')],
    [sg.Output(size=(65, 8), key='-OUTPUT-PCA-RF_SVM-')],
    [sg.Button('4.UMAP+RandomForest_SVM'), sg.Button('Open UMAP Picture')],
    [sg.Output(size=(65, 7), key='-OUTPUT-UMAP-RF_SVM-')],
    [sg.Button('Exit')]
]

window = sg.Window('Fetal Health Analysis', layout)

while True:
    event, values = window.read()

    if event in (None, 'Exit'):
        break

    elif event == '1.Preprocessing':
        try:
            csv_adress = values["-PATH-"]
            fetal_data = pd.read_csv(csv_adress)
            y = fetal_data.fetal_health
            x = fetal_data.drop(["fetal_health"], axis=1)
            window['-OUTPUT-'].print("Checking the shape of data - ", fetal_data.shape)
            window['-OUTPUT-'].print("--------------------------------")
            window['-OUTPUT-'].print("Checking for null values:")
            window['-OUTPUT-'].print(fetal_data.isnull().sum())
            window['-OUTPUT-'].print("--------------------------------")
            window['-OUTPUT-'].print("Checking data types:")
            window['-OUTPUT-'].print(fetal_data.dtypes)
            # HeatMap
            corr = fetal_data.corr()
            plt.figure(figsize=(10, 7))
            plt.title('Correlation Heatmap')
            sns_plot = sns.heatmap(corr, annot=True, annot_kws={"fontsize": 10, "fontweight": "bold"},
                                   fmt=".1f", cmap='gist_earth', xticklabels=range(len(fetal_data.columns)),
                                   yticklabels=range(len(fetal_data.columns)))
            fig = sns_plot.get_figure()
            fig.savefig('HeatMap.png')
        except:
            sg.popup_auto_close("Oops! Browse 'fetal_health.csv' first!")

    elif event == 'Open Heatmap Picture':
        if os.path.isfile("HeatMap.png"):
            sg.popup_ok('Correlation Heatmap', image="HeatMap.png")
        else:
            sg.popup_auto_close("Click the button 1.Preprocessing first!")

    elif event == '2.RandomForest_SVM':
        try:
            csv_adress = values["-PATH-"]
            fetal_data = pd.read_csv(csv_adress)
            y = fetal_data.fetal_health
            x = fetal_data.drop(["fetal_health"], axis=1)
            # SPLITTING THE DATA
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            # RANDOM FOREST CLASSIFIER
            model_rf = RandomForestClassifier()
            start_time_rf = time.time()
            model_rf.fit(x_train, y_train)
            end_time_rf = time.time()
            elapsed_time_rf = end_time_rf - start_time_rf
            window['-OUTPUT-RF_SVM-'].print(f"RANDOM FOREST operation took {elapsed_time_rf:.2f} seconds.")
            # Predict on test data
            y_pred_rf = model_rf.predict(x_test)
            # Calculate evaluation metrics
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
            recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
            f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
            # Print results to the second output section
            window['-OUTPUT-RF_SVM-'].print(f"Accuracy: {accuracy_rf:.3f} Precision: {precision_rf:.3f} "
                                            f"Recall: {recall_rf:.3f} F1_Score: {f1_rf:.3f}")
            # SVM
            model_svm = SVC()
            start_time_svm = time.time()
            model_svm.fit(x_train, y_train)
            end_time_svm = time.time()
            elapsed_time_svm = end_time_svm - start_time_svm
            window['-OUTPUT-RF_SVM-'].print(f"SVM operation took {elapsed_time_svm:.2f} seconds.")
            # Predict on test data
            y_pred_svm = model_svm.predict(x_test)
            # Calculate evaluation metrics
            accuracy_svm = accuracy_score(y_test, y_pred_svm)
            precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
            recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
            f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
            # Print results to the second output section
            window['-OUTPUT-RF_SVM-'].print(f"Accuracy: {accuracy_svm:.3f} Precision: {precision_svm:.3f} "
                                            f"Recall: {recall_svm:.3f} F1_Score: {f1_svm:.3f}")
        except:
            sg.popup_auto_close("Oops! Browse 'fetal_health.csv' first!")

    elif event == '3.PCA+RandomForest_SVM':
        try:
            csv_adress = values["-PATH-"]
            fetal_data = pd.read_csv(csv_adress)
            y = fetal_data.fetal_health
            x = fetal_data.drop(["fetal_health"], axis=1)
            # 1) Right number of dimensions
            scaler = StandardScaler()  # Standardize the Data
            scaled_fetal_data = scaler.fit_transform(fetal_data)
            pca = PCA()  # method PCA
            start_time_pca = time.time()  # Record the start time PCA
            pca.fit(scaled_fetal_data)  # Train
            end_time_pca = time.time()  # Record the end time PCA
            elapsed_time_pca = end_time_pca - start_time_pca  # Calculate PCA time
            window['-OUTPUT-PCA-RF_SVM-'].print(f"PCA operation took {elapsed_time_pca:.4f} seconds.")
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            dim = np.argmax(cumsum >= 0.95) + 1
            window['-OUTPUT-PCA-RF_SVM-'].print('The number of dimensions required to preserve 95% of variance is', dim)
            # 2) Use method PCA
            scaler.fit(fetal_data)
            scaled_data = scaler.transform(fetal_data)
            pca = PCA(n_components=14)
            pca.fit(scaled_data)
            x_pca = pca.transform(scaled_data)  # Data after PCA
            window['-OUTPUT-PCA-RF_SVM-'].print('Shape before PCA is ', scaled_data.shape, 'Shape after PCA is',
                                                x_pca.shape)
            # SPLITTING THE DATA
            x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
            # 3) RANDOM FOREST CLASSIFIER
            pca_model_rf = RandomForestClassifier()
            start_time_rf = time.time()  # Record the start time RANDOM FOREST
            pca_model_rf.fit(x_train_pca, y_train)  # Train the DT model on the train data
            end_time_rf = time.time()  # Record the end time RANDOM FOREST
            elapsed_time_rf = end_time_rf - start_time_rf  # Calculate RANDOM FOREST
            window['-OUTPUT-PCA-RF_SVM-'].print(f"RANDOM FOREST operation took {elapsed_time_rf:.2f} seconds.")
            # Predict on test data
            y_pred_rf = pca_model_rf.predict(x_test_pca)
            # Calculate evaluation metrics
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
            recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
            f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
            window['-OUTPUT-PCA-RF_SVM-'].print(f"Accuracy: {accuracy_rf:.3f} Precision: {precision_rf:.3f} "
                                                f"Recall: {recall_rf:.3f} F1_Score: {f1_rf:.3f}")
            # 4) SUPPORT VECTOR MACHINES (SVM)
            model_svm = SVC()
            start_time_svm = time.time()  # Record the start time SVM
            model_svm.fit(x_train_pca, y_train)  # Train SVC on train data
            end_time_svm = time.time()  # Record the end time SVM
            elapsed_time_svm = end_time_svm - start_time_svm  # Calculate SVM
            window['-OUTPUT-PCA-RF_SVM-'].print(f"SVM operation took {elapsed_time_svm:.2f} seconds.")
            # Predict on the test data
            y_pred_svm = model_svm.predict(x_test_pca)
            # Calculate evaluation metrics
            accuracy_svm = accuracy_score(y_test, y_pred_svm)
            precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
            recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
            f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
            window['-OUTPUT-PCA-RF_SVM-'].print(f"Accuracy: {accuracy_svm:.3f} Precision: {precision_svm:.3f} "
                                                f"Recall: {recall_svm:.3f} F1_Score: {f1_svm:.3f}")
            # Visual PCA
            labels_str = ["Normal", "Suspect", "Pathological"]
            plt.figure(figsize=(12, 7))
            scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, label=labels_str, cmap='viridis')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(handles=scatter.legend_elements()[0], labels=labels_str, loc="lower right")
            plt.colorbar(label='Classes')
            plt.title('PCA Projection of Fetal Health Dataset')
            plt.savefig('PCA.png')
        except:
            sg.popup_auto_close("Oops! Browse 'fetal_health.csv' first!")

    elif event == 'Open PCA Picture':
        if os.path.isfile("PCA.png"):
            sg.popup_ok('PCA_method', image="PCA.png")
        else:
            sg.popup_auto_close("Click the button '3.PCA+RandomForest_SVM' first ")

    elif event == '4.UMAP+RandomForest_SVM':
        try:
            csv_adress = values["-PATH-"]
            fetal_data = pd.read_csv(csv_adress)
            y = fetal_data.fetal_health
            x = fetal_data.drop(["fetal_health"], axis=1)
            sg.popup_auto_close('I am working...It takes more time than PCA...')
            # 2) Use method UMAP
            scaler = StandardScaler()  # Standardize the Data
            scaled_fetal_data = scaler.fit_transform(fetal_data)
            umap = umap.UMAP()
            start_time_umap = time.time()  # Record the start time UMAP
            x_umap = umap.fit_transform(scaled_fetal_data)  # Data after UMAP
            end_time_umap = time.time()  # Record the end time UMAP
            elapsed_time_umap = end_time_umap - start_time_umap  # Calculate UMAP time
            window['-OUTPUT-UMAP-RF_SVM-'].print(f"UMAP operation took {elapsed_time_umap:.2f} seconds.")
            window['-OUTPUT-UMAP-RF_SVM-'].print('Shape before UMAP is ', x.shape, 'Shape after UMAP is', x_umap.shape)
            # SPLITTING THE DATA
            x_train_umap, x_test_umap, y_train, y_test = train_test_split(x_umap, y, test_size=0.2, random_state=42)
            # 3) RANDOM FOREST CLASSIFIER
            model_rf = RandomForestClassifier()
            # Train the DT model on the train data
            start_time_rf = time.time()  # Record the start time RANDOM FOREST
            model_rf.fit(x_train_umap, y_train)
            end_time_rf = time.time()  # Record the end time RANDOM FOREST
            elapsed_time_rf = end_time_rf - start_time_rf  # Calculate RANDOM FOREST
            window['-OUTPUT-UMAP-RF_SVM-'].print(f"RANDOM FOREST operation took {elapsed_time_rf:.2f} seconds.")
            # Predict on test data
            y_pred_rf = model_rf.predict(x_test_umap)
            # Calculate evaluation metrics
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
            recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
            f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
            window['-OUTPUT-UMAP-RF_SVM-'].print(f"Accuracy: {accuracy_rf:.3f} Precision: {precision_rf:.3f} "
                                                 f"Recall: {recall_rf:.3f} F1_Score: {f1_rf:.3f}")
            # 4) SUPPORT VECTOR MACHINES (SVM)
            model_svm = SVC()
            # Train SVC on train data
            start_time_svm = time.time()  # Record the start time SVM
            model_svm.fit(x_train_umap, y_train)
            end_time_svm = time.time()  # Record the end time SVM
            elapsed_time_svm = end_time_svm - start_time_svm  # Calculate SVM
            window['-OUTPUT-UMAP-RF_SVM-'].print(f"SVM operation took {elapsed_time_svm:.2f} seconds.")
            # Predict on the test data
            y_pred_svm = model_svm.predict(x_test_umap)
            # Calculate evaluation metrics
            accuracy_svm = accuracy_score(y_test, y_pred_svm)
            precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
            recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
            f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
            window['-OUTPUT-UMAP-RF_SVM-'].print(f"Accuracy: {accuracy_svm:.3f} Precision: {precision_svm:.3f} "
                                                 f"Recall: {recall_svm:.3f} F1_Score: {f1_svm:.3f}")
            # Visual UMAP
            labels_str = ["Normal", "Suspect", "Pathological"]
            plt.figure(figsize=(12, 7))
            scatter = plt.scatter(x_umap[:, 0], x_umap[:, 1], c=y, label=labels_str, cmap='viridis')
            plt.legend(handles=scatter.legend_elements()[0], labels=labels_str, loc="lower right")
            plt.colorbar(label='Classes')
            plt.title('UMAP Projection of Fetal Health Dataset')
            plt.savefig('UMAP.png')
        except:
            sg.popup_auto_close("Oops! Browse 'fetal_health.csv' first!")

    elif event == 'Open UMAP Picture':
        if os.path.isfile("UMAP.png"):
            sg.popup_ok('UMAP_method', image="UMAP.png")
        else:
            sg.popup_auto_close("Click the button '4.UMAP+RandomForest_SVM' first")

window.close()

#######################################################################################################
######################################################################################################
# # 1 1 1 1 1 1 1 PREPROCESSING
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# #Reading the dataset using pandas library
# fetal_data = pd.read_csv("fetal_health.csv")
#
# # 1) Checking for null values
# null = fetal_data.isnull().sum()
# print("Checking for null values:")
# print(null)
# print("--------------------------------")
#
# # 2) checking the shape of data
# print("The shape of data - ", fetal_data.shape)
# print("--------------------------------")
# # 3) checking data types
# print(fetal_data.info())
# print("--------------------------------")
# All the values in the dataset are integers or float.
# There are no string values in the dataset.
#
# # 4) Heat map
# corr = fetal_data.corr()
# plt.figure(figsize=(20,10))
# sns.heatmap(corr, annot=True, cmap='gist_earth')
# plt.title('Correlation Heatmap')
# plt.show()

# 2 2 2 2 2 2 2 2 2 2 BASE Classifier
# import pandas as pd
# #Splitting the data
# from sklearn.model_selection import train_test_split
# #Classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import time
#
# #Reading the dataset using pandas library
# fetal_data = pd.read_csv("fetal_health.csv")
#
# y=fetal_data.fetal_health
# x=fetal_data.drop(["fetal_health"],axis=1)
# # 1) SPLITTING THE DATA
# x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#
# # 2) RANDOM FOREST CLASSIFIER
# model_rf = RandomForestClassifier()
# start_time_rf = time.time() # Record the start time RANDOM FOREST
# model_rf.fit(x_train,y_train) # Train the DT model on the train data
# end_time_rf = time.time() # Record the end time RANDOM FOREST
# elapsed_time_rf = end_time_rf - start_time_rf # Calculate RANDOM FOREST
# print(f"RANDOM FOREST operation took {elapsed_time_rf:.2f} seconds.")
# # Predict on test data
# y_pred_rf = model_rf.predict(x_test)
# # Calculate evaluation metrics
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
# recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
# f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
#
# # 3) SUPPORT VECTOR MACHINES (SVM)
# model_svm = SVC()
#
# start_time_svm = time.time() # Record the start time SVM
# model_svm.fit(x_train, y_train) # Train SVC on train data
# end_time_svm = time.time() # Record the end time SVM
# elapsed_time_svm = end_time_svm - start_time_svm # Calculate SVM
# print(f"SVM operation took {elapsed_time_svm:.2f} seconds.")
# # Predict on the test data
# y_pred_svm = model_svm.predict(x_test)
# # Calculate evaluation metrics
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
# recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
# f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
#
# # 4) Define the models and their corresponding evaluation metrics
# models = ["Random Forest", "SVM"]
# accuracy = [accuracy_rf, accuracy_svm]
# precision = [precision_rf, precision_svm]
# recall = [recall_rf, recall_svm]
# f1_score = [f1_rf, f1_svm]
# # Create summary table in df
# summary_table = pd.DataFrame({
#     "Model": models,
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Recall": recall,
#     "F1_Score": f1_score
# })
# print(summary_table.round(3))

# 3 3 3 3 3 3 3 3 3 3 3 3 3 PCA
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# #Classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# import time
#
# # 0) Reading the dataset using pandas library
# fetal_data = pd.read_csv("fetal_health.csv")
# y=fetal_data.fetal_health
# x=fetal_data.drop(["fetal_health"],axis=1)
#
# # 1) Right number of dimensions
# scaler=StandardScaler() # Standardize the Data
# scaled_fetal_data = scaler.fit_transform(fetal_data)
# pca=PCA()  #method PCA
# start_time_pca = time.time() # Record the start time PCA
# pca.fit(scaled_fetal_data)  # Train
# end_time_pca = time.time() # Record the end time PCA
# elapsed_time_pca = end_time_pca - start_time_pca # Calculate PCA time
# print(f"PCA operation took {elapsed_time_pca:.2f} seconds.")
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# dim = np.argmax(cumsum >= 0.95)+1
# print('The number of dimensions required to preserve 95% of variance is',dim)
#
# # 2) Use method PCA
# scaler.fit(fetal_data)
# scaled_data=scaler.transform(fetal_data)
# pca=PCA(n_components=14)
# pca.fit(scaled_data)
# x_pca=pca.transform(scaled_data) # Data after PCA
# print('Shape before PCA is ',scaled_data.shape,'Shape after PCA is', x_pca.shape)
#
# #SPLITTING THE DATA
# x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
#
# # 3) RANDOM FOREST CLASSIFIER
# model_rf = RandomForestClassifier()
#
# start_time_rf = time.time() # Record the start time RANDOM FOREST
# model_rf.fit(x_train_pca,y_train) # Train the DT model on the train data
# end_time_rf = time.time() # Record the end time RANDOM FOREST
# elapsed_time_rf = end_time_rf - start_time_rf # Calculate RANDOM FOREST
# print(f"RANDOM FOREST operation took {elapsed_time_rf:.2f} seconds.")
# # Predict on test data
# y_pred_rf = model_rf.predict(x_test_pca)
# # Calculate evaluation metrics
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
# recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
# f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
#
# # 4) SUPPORT VECTOR MACHINES (SVM)
# model_svm = SVC()
# start_time_svm = time.time() # Record the start time SVM
# model_svm.fit(x_train_pca, y_train) # Train SVC on train data
# end_time_svm = time.time() # Record the end time SVM
# elapsed_time_svm = end_time_svm - start_time_svm # Calculate SVM
# print(f"SVM operation took {elapsed_time_svm:.2f} seconds.")
# # Predict on the test data
# y_pred_svm = model_svm.predict(x_test_pca)
# # Calculate evaluation metrics
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
# recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
# f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
#
# # 5) Define the models and their corresponding evaluation metrics
# models = ["Random Forest", "SVM"]
# accuracy = [accuracy_rf, accuracy_svm]
# precision = [precision_rf, precision_svm]
# recall = [recall_rf, recall_svm]
# f1_score = [f1_rf, f1_svm]
# # Create summary table in df
# summary_table = pd.DataFrame({
#     "Model": models,
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Recall": recall,
#     "F1_Score": f1_score
# })
# print(summary_table.round(3))
#
# # Visual PCA
# labels_str = ["Normal","Suspect","Pathological"]
# plt.figure(figsize=(15, 10))
# scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, label=labels_str, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels_str, loc="lower right")
# plt.colorbar(label='Classes')
# plt.title('PCA Projection of Fetal Health Dataset')
# plt.show()

# # 4 4 4 4 4 4 4 4 4 4 UMAP
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import umap
# from sklearn.model_selection import train_test_split
# #Classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# import time
#
# # 0) Reading the dataset using pandas library
# fetal_data = pd.read_csv("fetal_health.csv")
# y=fetal_data.fetal_health
# x=fetal_data.drop(["fetal_health"],axis=1)
#
#
# # 2) Use method UMAP
# scaler=StandardScaler() # Standardize the Data
# scaled_fetal_data = scaler.fit_transform(fetal_data)
# # umap=umap.UMAP(n_neighbors=40, min_dist=0.5, n_components=18, metric='manhattan')
# umap=umap.UMAP()
# # umap.fit(x)
# start_time_umap = time.time() # Record the start time UMAP
# x_umap=umap.fit_transform(scaled_fetal_data) # Data after UMAP
# end_time_umap = time.time() # Record the end time UMAP
# elapsed_time_umap = end_time_umap - start_time_umap # Calculate UMAP time
# print(f"UMAP operation took {elapsed_time_umap:.2f} seconds.")
# print('Shape before UMAP is ',x.shape,'Shape after UMAP is', x_umap.shape)
#
# #SPLITTING THE DATA
# x_train_umap, x_test_umap, y_train, y_test = train_test_split(x_umap, y, test_size=0.2, random_state=42)
#
# # 3) RANDOM FOREST CLASSIFIER
# model_rf = RandomForestClassifier()
# # Train the DT model on the train data
# start_time_rf = time.time() # Record the start time RANDOM FOREST
# model_rf.fit(x_train_umap, y_train)
# end_time_rf = time.time() # Record the end time RANDOM FOREST
# elapsed_time_rf = end_time_rf - start_time_rf # Calculate RANDOM FOREST
# print(f"RANDOM FOREST operation took {elapsed_time_rf:.2f} seconds.")
# # Predict on test data
# y_pred_rf = model_rf.predict(x_test_umap)
# # Calculate evaluation metrics
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
# recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
# f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
#
# # 4) SUPPORT VECTOR MACHINES (SVM)
# model_svm = SVC()
# # Train SVC on train data
# start_time_svm = time.time() # Record the start time SVM
# model_svm.fit(x_train_umap, y_train)
# end_time_svm = time.time() # Record the end time SVM
# elapsed_time_svm = end_time_svm - start_time_svm # Calculate SVM
# print(f"SVM operation took {elapsed_time_svm:.2f} seconds.")
# # Predict on the test data
# y_pred_svm = model_svm.predict(x_test_umap)
# # Calculate evaluation metrics
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
# recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
# f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
#
# # 5) Define the models and their corresponding evaluation metrics
# models = ["Random Forest", "SVM"]
# accuracy = [accuracy_rf, accuracy_svm]
# precision = [precision_rf, precision_svm]
# recall = [recall_rf, recall_svm]
# f1_score = [f1_rf, f1_svm]
# # Create summary table in df
# summary_table = pd.DataFrame({
#     "Model": models,
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Recall": recall,
#     "F1_Score": f1_score
# })
# print(summary_table.round(3))
#
# # Visual UMAP
# labels_str = ["Normal","Suspect","Pathological"]
# plt.figure(figsize=(15, 10))
# scatter = plt.scatter(x_umap[:, 0], x_umap[:, 1], c=y, label=labels_str, cmap='viridis')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels_str, loc="lower right")
# plt.colorbar(label='Classes')
# plt.title('UMAP Projection of Fetal Health Dataset')
# plt.show()

#   ###### F I N I S H ######        ###### F I N I S H ######
#   ###### F I N I S H ######        ###### F I N I S H ######
#   ###### F I N I S H ######        ###### F I N I S H ######
##################### Отправил Базилевич ######################
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Загружаем данные
# fetal_df = pd.read_csv("fetal_health.csv")
#
# # У нас есть столбец 'fetal_health' с метками классов
# y = fetal_df['fetal_health']
#
# # Удаляем столбец 'fetal_health' из данных перед снижением размерности
# fetal_df = fetal_df.drop('fetal_health', axis=1)
#
# # Создаем модель RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100, random_state=0)
# # model = RandomForestRegressor(random_state=1, max_depth=10)
#
# # Обучаем модель на данных
# model.fit(fetal_df, y)
#
# # Получаем важность признаков
# feature_importances = model.feature_importances_
#
# # Сортируем признаки по их важности
# sorted_indices = feature_importances.argsort()[::-1]
#
# # Выводим важность признаков
# for i, idx in enumerate(sorted_indices):
#     print(f"Feature {i+1}: {fetal_df.columns[idx]} - Importance: {feature_importances[idx]}")
# # Отображаем на графике
# features = fetal_df.columns
# indices = np.argsort(feature_importances)
# plt.title('Feature Importances')
# plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], color='b', align='center')
# plt.yticks(range(len(sorted_indices)), [features[i] for i in sorted_indices])
# plt.xlabel('Relative Importance')
# plt.show()
############ ТУТ НАДО ДОДЕЛАТЬ############################
# import pandas as pd
# from sklearn.decomposition import FactorAnalysis
# import matplotlib.pyplot as plt
# # Загрузите данные
# fetal_df = pd.read_csv("fetal_health.csv")
#
# # У нас есть столбец 'fetal_health' с метками классов
# y = fetal_df['fetal_health']
#
# # Удаляем столбец 'fetal_health' из данных перед снижением размерности
# fetal_df = fetal_df.drop('fetal_health', axis=1)
#
# # Создаем экземпляр класса FactorAnalyzer
# # Указываем количество факторов, которые мы хотим извлечь
# fa = FactorAnalysis(n_components=3)
#
# # Применяем факторный анализ к данным и снижаем размерность
# reduced_data = fa.fit_transform(fetal_df)
#
# # Выводим уменьшенные данные
# print("Reduced Data:")
# print(reduced_data)
#
# plt.figure(figsize=(12, 8))
# plt.title('Factor Analysis Components')
#
# # Визуализация факторов (фактор 1, фактор 2, фактор 3)
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label='Factor 1 vs Factor 2')
# plt.scatter(reduced_data[:, 1], reduced_data[:, 2], label='Factor 2 vs Factor 3')
# plt.scatter(reduced_data[:, 2], reduced_data[:, 0], label='Factor 3 vs Factor 1')
# plt.legend()
# plt.xlabel('Factor 1')
# plt.ylabel('Factor 2/3')
# plt.show()
############# ТУТ НАДО ДОДЕЛАТЬ НЕПОНЯТНО ###################


########https://distrland.blogspot.com/2020/10/12-python.html#############
# # 3.1 Доля отсутствующих значений.
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # train=pd.read_csv("fetal_health.csv")
# # print(train.isnull().sum()/len(train)*100)
# # a = train.isnull().sum()/len(train)*100
# # variables = train.columns
# # variable = [ ]
# # for i in range(0,22):
# #     if a[i]<=20:   #setting the threshold as 20%
# #         variable.append(variables[i])
# # 3.2 Фильтр низкой дисперсии
# from sklearn.ensemble import RandomForestRegressor
#
# df = pd.read_csv("fetal_health.csv")
# df=df.drop('fetal_health', axis=1)
# model = RandomForestRegressor(random_state=1, max_depth=10)
# df=pd.get_dummies(df)
# model.fit(df,train.Item_Outlet_Sales)

#### https://www.datacamp.com/tutorial/random-forests-classifier-python#######
## Data Processing
# import pandas as pd
# import numpy as np
#
# # Modelling
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
# from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from scipy.stats import randint
# import sklearn.metrics
#
# # # Tree Visualisation
# # from sklearn.tree import export_graphviz
# # from IPython.display import Image
# # import graphviz
#
# fetal_df = pd.read_csv("fetal_health.csv")
#
# # fetal_df['default'] = fetal_df['default'].map({'no':0,'yes':1,'unknown':0})
# # fetal_df['fetal_health'] = fetal_df['fetal_health'].map({'no':0,'yes':1})
#
# # Split the data into features (X) and target (y)
# X = fetal_df.drop('fetal_health', axis=1)
# y = fetal_df['fetal_health']
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.70,
#                                                     stratify=y, shuffle=True, random_state = 42)
#
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# score_cbc=model.score(X_test,y_test)
# print('Score :',score_cbc)
#
# print(sklearn.metrics.confusion_matrix(y_test,model.predict(X_test)))


######################################
## ИСПОЛЬЗУЮ В ТЕЗИСАХ №№№№№№№  !!Работает! ##
# import umap.umap_ as umap
# import pandas as pd
# import matplotlib.pyplot as plt
#
# fetal_df = pd.read_csv("fetal_health.csv")
# embedding = umap.UMAP(n_neighbors=50, min_dist=0.01).fit_transform(fetal_df.drop('fetal_health', axis=1))
# # embedding = umap.UMAP().fit_transform(fetal_df)
# plt.scatter(embedding[:, 0], embedding[:, 1], c=fetal_df['fetal_health'], cmap='viridis')
# plt.colorbar(label='fetal_health')
# plt.title('UMAP Projection of fetal_health Data')
# plt.show()
##### Улучшаем!!!
# import umap.umap_ as umap
# import pandas as pd
# import matplotlib.pyplot as plt
# import time
#
# fetal_df = pd.read_csv("fetal_health.csv")
# y = fetal_df['fetal_health']
# # Удаляем столбец 'fetal_health' из данных перед снижением размерности
# fetal_df = fetal_df.drop('fetal_health', axis=1)
# # Начинаем отсчет времени
# start_time = time.time()
# # Производим снижение размерности с помощью UMAP
# umap_method = umap.UMAP(n_neighbors=40, min_dist=0.5, n_components=2, metric='manhattan')
# embedding = umap_method.fit_transform(fetal_df)
# # embedding = umap.UMAP().fit_transform(fetal_df)
# print(f"UMAP new shape:")
# print(embedding.shape)
# print(embedding)
# # Завершаем отсчет времени
# end_time = time.time()
# # Вычисляем время выполнения
# execution_time = end_time - start_time
# print(f"UMAP execution time: {execution_time} seconds")
# # Создаем словарь с метками классов для отображения
# labels_str = ["Normal","Suspect","Pathological"]
# # Увеличиваем размер графика
# plt.figure(figsize=(15, 10))
# # Отображаем результаты, раскрашивая точки в соответствии с метками классов
# scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, label=labels_str, cmap='viridis')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels_str, loc="upper right")
# plt.colorbar(label='fetal_health')
# plt.title('UMAP Projection of fetal_health Data')
# plt.show()
############ ####################################

###  ИСПОЛЬЗУЮ В ТЕЗИСАХ  ############ GPT PSA ###### РАБОТАЕТ ПРАВИЛЬНО ###########
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import time
#
# # Загрузите данные
# fetal_df = pd.read_csv("fetal_health.csv")
# # Предполагается, что у вас есть столбец 'label' с метками классов
# y = fetal_df['fetal_health']
# # Удаляем столбец 'label' из данных перед снижением размерности
# fetal_df = fetal_df.drop('fetal_health', axis=1)
# # Начинаем отсчет времени
# start_time = time.time()
# # Производим снижение размерности с помощью PCA
# pca = PCA(0.95)
# # pca = PCA(n_components=20)
# # pca = PCA(n_components=2, whiten=True, svd_solver='auto', random_state=42)
# # pca = PCA(n_components=5, whiten=False, svd_solver='full', random_state=42)
# embedding = pca.fit_transform(fetal_df)
# # Завершаем отсчет времени
# end_time = time.time()
# # Вычисляем время выполнения
# execution_time = end_time - start_time
# print(f"PCA execution time: {execution_time} seconds")
# # PCA new shape
# print(f"PCA new shape:")
# print(embedding.shape)
# print(embedding)
# # Отримуємо відношення дисперсії для кожної компоненти
# cumsum  = np.cumsum(pca.explained_variance_ratio_)
# # Виводимо відсоток дисперсії, яку пояснює кожна компонента
# print("Explained Variance Ratio:", cumsum)
# d=np.argmax(cumsum>=0.95)+1
# print (d)
# # Создаем словарь с метками классов
# labels_str = ["Normal","Suspect","Pathological"]
# # Увеличиваем размер графика
# plt.figure(figsize=(15, 10))
# # Отображаем результаты, раскрашивая точки в соответствии с метками классов
# scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, label=labels_str, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels_str, loc="lower right")
# plt.colorbar(label='Classes')
# plt.title('PCA Projection of Fetal Health Dataset')
# plt.show()
#####УЛУЧШААЕМ!!! ""№№№№  ПРобуем стандартизацию№№№""""""
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import time
# from sklearn.preprocessing import StandardScaler
# # Загрузка даних
# fetal_df = pd.read_csv("fetal_health.csv")
# # Предполагается, что у вас есть столбец 'label' с метками классов
# y = fetal_df['fetal_health']
#
# scaler = StandardScaler()
# fetal_df_scaled = scaler.fit_transform(fetal_df)
# # Удаляем столбец 'label' из данных перед снижением размерности
# fetal_df = fetal_df.drop('fetal_health', axis=1)
# # Начинаем отсчет времени
# start_time = time.time()
# # Производим снижение размерности с помощью PCA
# pca = PCA(0.95)
# # pca = PCA(n_components=20)
# # pca = PCA(n_components=2, whiten=True, svd_solver='auto', random_state=42)
# # pca = PCA(n_components=5, whiten=False, svd_solver='full', random_state=42)
# fetal_df_pca = pca.fit_transform(fetal_df_scaled)
# # Завершаем отсчет времени
# end_time = time.time()
# # Вычисляем время выполнения
# execution_time = end_time - start_time
# print(f"PCA execution time: {execution_time} seconds")
# # Створення таблиці важливості компонент
# component_names = [f'Component {i+1}' for i in range(fetal_df_pca.shape[1])]
# explained_variance_ratio = pca.explained_variance_ratio_
#
# # Створення DataFrame
# pca_table = pd.DataFrame({'Component': component_names,
#                           'Explained Variance Ratio': explained_variance_ratio})
#
# # Додавання кумулятивної суми
# pca_table['Cumulative Explained Variance Ratio'] = np.cumsum(explained_variance_ratio)
#
# # Виведення таблиці
# print(pca_table)
#
# labels_str = ["Normal","Suspect","Pathological"]
# # Увеличиваем размер графика
# plt.figure(figsize=(15, 10))
# # Отображаем результаты, раскрашивая точки в соответствии с метками классов
# scatter = plt.scatter(fetal_df_pca[:, 0], fetal_df_pca[:, 1], c=y, label=labels_str, cmap='viridis')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(handles=scatter.legend_elements()[0], labels=labels_str, loc="lower right")
# plt.colorbar(label='Classes')
# plt.title('PCA Projection of Fetal Health Dataset')
# plt.show()


#######################https://umap-learn.readthedocs.io/en/latest/basic_usage.html ##########
# import umap.umap_ as umap
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# fetal_df = pd.read_csv("fetal_health.csv")
#
# reducer = umap.UMAP()
#
# scaled_fetal_df = StandardScaler().fit_transform(fetal_df)
# embedding = reducer.fit_transform(scaled_fetal_df)
# print(embedding.shape)
# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=[sns.color_palette()[x] for x in fetal_df.fetal_health.map({"1":0, "2":1, "3":2})])
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Fetal dataset', fontsize=24);
# plt.show()
######## https://www.kaggle.com/code/bextuychiev/beautiful-umap-tutorial-on-100-dimensional-data
# #    Вроде работает!!!
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import make_pipeline
# # fetal_df = pd.read_csv("fetal_health.csv")
# print(fetal_df.shape)
#
# X, y = fetal_df.drop("fetal_health", axis=1), fetal_df[["fetal_health"]].values.flatten()
# # Preprocess
# pipe = make_pipeline(SimpleImputer(strategy="mean"))
# X = pipe.fit_transform(X.copy())
#
# manifold = umap.UMAP().fit(X, y)
# X_reduced = manifold.transform(X)
# print(X_reduced.shape)
# # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=0.5) # ТУТ ВКЛЮЧИТЬ, ЕСЛИ НАДО
# # plt.show() # ТУТ ВКЛЮЧИТЬ, ЕСЛИ НАДО
# # So, we will choose Quantile Transformer to scale the features based on their quantiles and median.
# # This scaling method suits the dataset better since it contains many skewed and bimodal features:
# from sklearn.preprocessing import QuantileTransformer
# # Preprocess again
# pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
# X = pipe.fit_transform(X.copy())
# # Fit UMAP to processed data
# manifold = umap.UMAP().fit(X, y)
# X_reduced_2 = manifold.transform(X)
# plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1], c=y, s=0.5)
# plt.show()
################## https://habr.com/ru/articles/751050/######################
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import pandas as pd
# fetal_df = pd.read_csv("fetal_health.csv")
# # print(fetal_df.isnull().sum())
# scaler = StandardScaler()
# fetal_df = pd.DataFrame(data=scaler.fit_transform(fetal_df), columns=fetal_df.columns)
#
# kmeans = KMeans(n_clusters=3)
# cluster = kmeans.fit_predict(fetal_df)
####НЕ РАБОТАЕТ###### PCA (Principal Component Analysis) ########## Linear
# from sklearn.decomposition import PCA
# import seaborn as sns
# import matplotlib.pyplot as plt
# pca2D = PCA(n_components=2)
# #dimensions
# pca_2D = pca2D.fit_transform(fetal_df)
# pca2D_df = pd.DataFrame(data = pca_2D, columns = ['x', 'y'])
#
# pca2D_df['cluster'] = cluster
#
# sns.scatterplot(x='x', y='y', hue='cluster', data=pca2D_df)
# plt.title("PCA")
# plt.show()
################### ICA (Independent Computing Architecture)############ Linear
# from sklearn.decomposition import FastICA
# ica2D = FastICA(n_components=2)
# ica_data2D = ica2D.fit_transform(fetal_df)
# ica2D_df = pd.DataFrame(data =  ica_data2D,columns = ['x', 'y'])
#
# ica2D_df['cluster'] = cluster
#
# sns.scatterplot(x='x', y='y', hue='cluster', data=ica2D_df)
# plt.title("ICA")
# plt.show()
################# MDS（Multidimensional Scaling）######### NonLinear
# from sklearn.manifold import MDS
# mds2D = MDS(n_components=2)
#
# mds_data2D = mds2D.fit_transform(fetal_df)
# mds2D_df = pd.DataFrame(data=mds_data2D, columns=['x', 'y'])
#
# mds2D_df['cluster'] = cluster
#
# sns.scatterplot(x='x', y='y', hue='cluster', data=mds2D_df)
# plt.title("MDS")
# plt.show()
################ t-SNE (t-Distributed Stochastic Neighbor Embedding) ###################
# from sklearn.manifold import TSNE
# tsne2D = TSNE(n_components=2)
# tsne_data2D = tsne2D.fit_transform(fetal_df)
# tsne2D_df = pd.DataFrame(data =  tsne_data2D, columns = ['x', 'y'])
#
# tsne2D_df['cluster'] = cluster
#
# sns.scatterplot(x='x', y='y', hue='cluster', data=tsne2D_df)
# plt.title("T-SNE")
# plt.show()
############### UMAP (Uniform Manifold Approximation and Projection) ##########
# from umap.umap_ import UMAP
# umap2D = umap.UMAP(n_components=2)
# umap_data2D = umap2D.fit_transform(fetal_df)
# umap2D_df = pd.DataFrame(data =  umap_data2D,columns = ['x', 'y'])
#
# umap2D_df['cluster'] = cluster
#
# sns.scatterplot(x='x', y='y', hue='cluster', data=umap2D_df)
# plt.title("UMAP")
# plt.show()
########### Mnist ########################################################################
# import pandas as pd
# import umap
# import matplotlib.pyplot as plt
#
# fmnist = pd.read_csv('fashion-mnist.csv') # считываем данные
#
# embedding = umap.UMAP(n_neighbors=5).fit_transform(fmnist.drop('label', axis=1))
#
# plt.scatter(embedding[:, 0], embedding[:, 1], c=fmnist['label'], cmap='viridis')
# plt.colorbar(label='label')
# plt.title('UMAP Projection of MN Data')
# plt.show()
#####https://www.kaggle.com/code/nulldata/tsne-alternate-umap-3d-viz-on-fashion-mnist#########
##### Mnist ##
# import umap
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# fashion_train = pd.read_csv('fashion-mnist.csv')
#
# data = fashion_train.iloc[:, 1:].values.astype(np.float32)
# target = fashion_train['label'].values
#
# reduce = umap.UMAP(random_state = 223) #just for reproducibility
# embedding = reduce.fit_transform(data)
#
# df = pd.DataFrame(embedding, columns=('x', 'y'))
# df["class"] = target
#
# labels = { 0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
#           5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8 : 'Bag', 9 : 'Ankle boot'}
#
# df["class"].replace(labels, inplace=True)
#
# sns.set_style("whitegrid", {'axes.grid' : False})
# #adjusting plot dots with plot_kws
# ax = sns.pairplot(x_vars = ["x"], y_vars = ["y"],data = df,
#              hue = "class",size=11, plot_kws={"s": 4})
# ax.fig.suptitle('Fashion MNIST clustered with UMAP')
# plt.show()
############# ПО ПРИМЕРУ ВЫШЕ РАБОТАЕТ##############################
# import umap
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# fetal_train = pd.read_csv('fetal_health.csv')
#
# data = fetal_train.iloc[:, 1:].values.astype(np.float32)
# target = fetal_train['fetal_health'].values
#
# reduce = umap.UMAP(n_neighbors=50, min_dist=0.01, random_state = 223) #just for reproducibility
# # reduce = umap.UMAP(random_state = 223)
# embedding = reduce.fit_transform(data)
#
# df = pd.DataFrame(embedding, columns=('x', 'y'))
# df["class"] = target
#
# labels = { 1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
#
# df["class"].replace(labels, inplace=True)
#
# sns.set_style("whitegrid", {'axes.grid' : False})
# #adjusting plot dots with plot_kws
# ax = sns.pairplot(x_vars = ["x"], y_vars = ["y"],data = df,
#              hue = "class",size=11, plot_kws={"s": 40})
# ax.fig.suptitle('Fetal clustered with UMAP')
# plt.show()
