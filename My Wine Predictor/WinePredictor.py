from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    #load dataset
    wine = datasets.load_wine()

    print("Names of features:")
    print(wine.feature_names)

    print()
    print()
    print("Names of label (class_0, class_1, class_2):")
    print(wine.target_names)

    print()
    print()
    print("1st 5 records")
    print(wine.data[0:5])

    print()
    print()
    print("Targets (0:class_0,1:class_1,2:class_2):")
    print(wine.target)

    #splitting dataset 70% for training and 30% for testing
    X_train,X_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3)

    #KNN classifier
    knn = KNeighborsClassifier(n_neighbors=7)

    knn.fit(X_train,y_train)

    y_predicted = knn.predict(X_test)

    #Accuracy calculation
    print("Accuracy:",metrics.accuracy_score(y_test,y_predicted)*100,"%")

if __name__ == "__main__":
    print("------- Wine Predictor using K Nearest Neighbors algorithm -------")
    WinePredictor()
