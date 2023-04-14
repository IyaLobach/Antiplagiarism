import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, classification_report, hinge_loss, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
import time

pairs = pd.read_csv('C:/Users/iyush/PycharmProjects/BERT/dataset/plagiarism.csv', header=None, sep=";")
pairs = pairs.iloc[1:, :]
text = pd.read_csv('C:/Users/iyush/PycharmProjects/BERT/dataset/Text_vector.csv', header=None, sep=",")


labels = pairs[3]
pairs = pairs[[0, 1]]



def get_average(l):
    size = len(l)
    res = []
    for k in range(len(l[0])):
        average = 0
        for m in range(size):
            average = average + l[m][k]
        average = average/size
        res.append(average)
    return res



# среднее

index = 1
for col_name, data in pairs.items():
    for elem in data:
        df = text.loc[text[0] == int(elem)]
        if df is not None:
            tmp = []
            for part in df[3]:
                part = part.replace('[', '')
                part = part.replace(']', '')
                result = [float(val) for val in part.split(', ')]
                tmp.append(result)
            result = get_average(tmp)
            pairs[col_name][index] = result
        index = index + 1
    index = 1


print(pairs)
pairs.to_csv(r'C:/Users/iyush/PycharmProjects/BERT/dataset/New_plag.csv', index=False, header=None)



tmp = []
for i, row in pairs.iterrows():
    col_1 = row.values[0]
    col_2 = row.values[1]
    tmp.append(np.concatenate((col_1, col_2)))
df_new = pd.DataFrame(tmp)
print(df_new)


X_train, X_test, y_train, y_test = train_test_split(df_new, labels, test_size=0.2)
print('Размер обучающей выборки = ', len(X_train))
print('Размер тестовой выборки = ', len(X_test))


# # Метод опорных векторов
#
# start = time.time()
# losses = []
# svcclassifier = None
# for i in range(0, 6000): # 6000
#     svclassifier = SVC(kernel='linear', max_iter=i)
#     svclassifier.fit(X_train, y_train)
#     y_pred = svclassifier.predict(X_test)
#     loss = hinge_loss(y_test, y_pred)
#     losses.append(loss)
# plt.plot(losses)
# plt.show()
# print('Время обучения = ', (time.time() - start)/60) # 42 минуты
#
#
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
print("Iter = ", svclassifier.n_iter_)
accuracy = svclassifier.score(X_test, y_test)
y = svclassifier.predict(X_test)
print("Accuracy of SVM:", round(accuracy, 2))
print((confusion_matrix(y_test, y))) # матрица путаницы
print(classification_report(y_test, y))

