import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def MyPlayPredictor(data_path):
    data = pd.read_csv(data_path,index_col=0)

    #print(data[0:10])
    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    le = preprocessing.LabelEncoder()

    weather_encoded = le.fit_transform(whether)
    #print(weather_encoded)

    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)

    #print(temp_encoded)
    #print(label)

    features = list(zip(weather_encoded,temp_encoded))

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(features,label)
    print()
    print("Testing of the algorithm manually:")
    print()
    while(True):
        print("Please enter the weather and temperature conditions:(-1 to quit)")
        print()
        print("Weather: \tOvercast:0 \tRainy:1 \tSunny:2")
        print()
        print("Temperature: \tCool:0 \tHot:1 \tMild:2")
        print()
        a = list(map(int,input().split()))
        if a[0]>=0 and a[0] <=2 and a[1]>=0 and a[1] <=2:
            predicted = model.predict([[a[0],a[1]]]) #0:Overcast,2:Mild

            if predicted == 0:
                print("No, you cannot play")
            else:
                print("Yes, you can play")
        else:
            break

    prediction = model.predict(features)

    print("Accuracy of the model is:",accuracy_score(label, prediction)*100)

if __name__ == "__main__":
    print("Machine Learning Application")
    print()
    print("Play Predictor Application using K Nearest Neighbor algorithm")
    print()
    MyPlayPredictor("MarvellousInfosystems_PlayPredictor.csv")
