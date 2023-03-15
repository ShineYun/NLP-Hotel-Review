from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def modelSelection(X_train, Y_train, X_test, Y_test):
    # return fitted bestModel
    # The model is trained by training dataset & selected based on the performace on the test dataset
    global bestModel
    models = [LogisticRegression(), Perceptron(), SGDClassifier(),
              KNeighborsClassifier(), RandomForestClassifier(n_estimators=10), DecisionTreeClassifier(), GaussianNB(),
              GradientBoostingClassifier()]
    modelsName = ['LogisticRegression', 'Perceptron', 'SGDClassifier', 'KNeighborsClassifier',
                  'RandomForestClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'GradientBoostingClassifier',
                  'GradientBoostingClassifier']
    bestModelName = ''
    bestScore = 0
    for i in range(len(models)):
        tmpModel = models[i]
        modelName = modelsName[i]
        tmpModel.fit(X_train, Y_train)
        Y_pred = tmpModel.predict(X_train)
        score = tmpModel.score(X_test, Y_test)
        print(modelName + ':', score)
        if score > bestScore:
            bestScore = score
            bestModel = tmpModel
            bestModelName = modelName
    print('BestModel is ' + bestModelName + ', score is ' + str(bestScore))
    return bestModel
# random forest classifier with n_estimators=10 (default)
