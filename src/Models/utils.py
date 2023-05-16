import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from xgboost import plot_importance
from xgboost import plot_tree
import seaborn as sns
import pandas as pd
import numpy as np


def evaluate_model(model, X_test, y_test):
    # predict the test set
    y_pred = model.predict(X_test)

    # print the accuracy score
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    # print f1 score
    print("F1 score: ", f1_score(y_test, y_pred, average='macro'))

    # print classification report
    print("Classification report: \n", classification_report(y_test, y_pred))

    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

    # plot feature importance   
    plot_importance(model)
    plt.show()

    # plot decision tree
    plot_tree(model)
    plt.show()


def oversample_dataset(df):
    # get the number of samples in each class
    class_size = df['Body_Level'].value_counts().max()
    # create an empty dataframe
    df_balanced = pd.DataFrame()
    # for each class
    for class_name in df['Body_Level'].unique():
        # get the samples of the current class
        df_class = df[df['Body_Level'] == class_name]
        # oversample the current class
        df_class_over = df_class.sample(class_size, replace=True)
        # append the oversampled class to the balanced dataframe
        df_balanced = df_balanced.append(df_class_over)
    return df_balanced


def draw_parameter_validation_curve(model, param_name, param_range, X_train, y_train, cv):

    train_scores, test_scores = validation_curve(model, X_train, y_train, param_name=param_name, param_range=param_range,cv=cv)

    # train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # plt.plot(param_range, train_mean, label='Training Score')
    plt.plot(param_range, test_mean, label='Cross-Validation Score')

    plt.title('Validation Curve')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')

    plt.show()