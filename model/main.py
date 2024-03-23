# import dependecies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# data cleaning function
def getClean():
    # uses pandas to get data into variable
    data = pd.read_csv("data/data.csv")
    
    # drop unnamed data column and id column
    data = data.drop(["Unnamed: 32", "id"], axis=1)

    # maps the M (malignant) disgnosis to 1 and the B(begnin) diagnosis to 0
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    # returns data
    return data

# creates machine learning model based on data
def createModel(data):

    # drops diagnosis data
    X = data.drop(["diagnosis"], axis=1)

    # gets only diagnosis data
    Y = data['diagnosis']

    # scaling data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data into training and testing sets with an 80/20 split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X,Y,test_size=0.2, random_state=42
    )

    # train model using logistic regression
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain)

    # test model
    Ypred = model.predict(Xtest)

    # print accuracy score of our model
    print('Accuracy: ', accuracy_score(Ytest, Ypred))

    # print classification
    print('Classification: \n', classification_report(Ytest, Ypred))

    # return model and scaler
    return model, scaler


# main function
def main():

    # gets clean data
    data = getClean()

    # creates, gets, and tests model and scaler
    model, scaler = createModel(data)

    # export model and scaler
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()