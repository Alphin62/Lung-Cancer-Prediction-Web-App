# **_Lung-Cancer-Prediction-Web-App_**

# Click the link to visit Web application: https://alphin62-lung-cancer-prediction-web-app-streamlit-app-snrqsi.streamlit.app/

![image](https://github.com/Alphin62/Lung-Cancer-Prediction-Web-App/blob/main/image.jpg)


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('lung_cancer.csv')

data.head()

![Screenshot 2022-12-12 020925](https://user-images.githubusercontent.com/118610159/206927592-ed63362a-b1cf-4428-af09-22b073f20d31.png)

data.shape

(309, 16)

data.drop_duplicates()

![Screenshot 2022-12-12 020925](https://user-images.githubusercontent.com/118610159/206927836-9ba6d690-0cf2-43d7-aee5-0c873eef61bb.png)

data['GENDER']= data['GENDER'].map({'M':1, 'F':0})

data['LUNG_CANCER']= data['LUNG_CANCER'].map({'YES':1, 'NO':0})

## Data Visualization

plt.figure(figsize = (18,15))

plt.title('Heatmap of lung cancer data')

sns.heatmap(copy_1.corr(method = 'spearman'), annot = True)

plt.show()

![download (1)](https://user-images.githubusercontent.com/118610159/206927897-4834914b-7404-47b7-9714-123a377d4fe3.png)

plt.figure(figsize = (50,45))

sns.pairplot(copy_1)

plt.show()

![download (2)](https://user-images.githubusercontent.com/118610159/206927949-e4d5424e-1fef-4d44-a9b3-2b36c108e346.png)


sns.factorplot(data = copy_2, x = 'AGE', col = 'GENDER', kind = 'count', size = 20)

![download (3)](https://user-images.githubusercontent.com/118610159/206928019-81cd419f-e9f3-4b99-a39c-d1b592bde67b.png)

Patients age between 51 to 72 has more counts affected by lung cancer. Mostly patients age between 60 to 65 reached highest count of cases.

Female : Age between 60 to 65 count is less, from 21 - 59 & 66 - 87 records less in count but average is recorded is high.

Male : Age between 55 - 72 records high cases as average of 7 & remaining age group records less number of cases.

Therefore Male has affected more than Female

## Accuracy score of various algorithms

copy_2 = data.copy()

x_data = copy_2.iloc[:,0:15].values

y_data = copy_2.iloc[:, -1].values

x = x_data.copy()

y = y_data.copy()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)

### Prediction using Supervised learning algorithms

def supervised_learning_model(x_train, y_train):
    
    #1. Logistic Regression
    
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    
    # 2. Random Forest Classifier
    
    from sklearn.ensemble import RandomForestClassifier
    forest_c = RandomForestClassifier(n_estimators = 10)
    forest_c.fit(x_train, y_train)
    
    # 3. Decision Tree Classifier
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    
    # 4. Gaussian Naive Bayes
    
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    
    # 5. K-Nearest Neighbor
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(x_train, y_train)
    
    # 6. Support Vector Machine
    
    from sklearn.svm import SVC
    sv_classifier = SVC()
    sv_classifier.fit(x_train, y_train)
    
    # Printing accuracy
    
    print('1. Logistic Regression : ',log_reg.score(x_train, y_train))
    print('2. Random Forest Classifier : ',forest_c.score(x_train, y_train))
    print('3. Decision Tree Classifier : ',tree.score(x_train, y_train))
    print('4. Gaussian Naive Bayes : ',gnb.score(x_train, y_train))
    print('5. K-Nearest Neighbor : ',knn.score(x_train, y_train))
    print('6. Support Vector Machine : ',sv_classifier.score(x_train, y_train))
          
    return log_reg, forest_c, tree, gnb, knn, sv_classifier

print(' Accuracy Score of Supervised Learning Models\n')

model = supervised_learning_model(x_train, y_train)

    1. Logistic Regression :  0.9468599033816425
    2. Random Forest Classifier :  0.9903381642512077
    3. Decision Tree Classifier :  0.9951690821256038
    4. Gaussian Naive Bayes :  0.927536231884058
    5. K-Nearest Neighbor :  0.9371980676328503
    6. Support Vector Machine :  0.9565217391304348

Accuracy of Decision tree classifier model - 99.5% & Random forest classifier - 99%
Therefore I consider these 2 models are best to predict the output.

## Classification report & accuracy score

from sklearn.metrics import classification_report, accuracy_score

print('---------------  Classification Report  ---------------\n\n')

for i in range(len(model)):

    print('**********  Model ', i+1,'  **********')
    print(classification_report(y_test, model[i].predict(x_test)))
    print('Accuracy Score : ', accuracy_score(y_test, model[i].predict(x_test)),'\n\n')
    
![Screenshot 2022-12-12 020925](https://user-images.githubusercontent.com/118610159/206928383-8db4425e-82de-4e1c-b988-de469581b1f3.png)

![Screenshot 2022-12-12 020925](https://user-images.githubusercontent.com/118610159/206928416-523a163b-8072-4df4-8ca1-44f51706b9cd.png)

![Screenshot 2022-12-12 020925](https://user-images.githubusercontent.com/118610159/206928435-b1b54421-0b0e-4b83-a009-f037f2c2a983.png)


# To refer the full program Click on this link: https://github.com/Alphin62/Lung-Cancer-Prediction-Web-App/blob/main/Code/Sample%20Code.ipynb
