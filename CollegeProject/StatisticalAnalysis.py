import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('data1.csv')

def linearRegression():
    x = data[['gpa','ec','ld']]
    y = data['sat']
    
    # split into test and train data
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4, random_state = 0)
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    
    # train data
    y_train_predict = lr.predict(x_train)
    y_test_predict = lr.predict(x_test)

    #heatmap
    corr_matrix = x.corr()
    sns.heatmap(corr_matrix, annot=True,)
    plt.title('linear regression correlation')
    plt.show()
    
    #evaluate model accuracy
    train_mse = mean_squared_error(y_train, y_train_predict)
    train_r2 = r2_score(y_train, y_train_predict)
    test_mse = mean_squared_error(y_test, y_test_predict)
    test_r2 = r2_score(y_test, y_test_predict)
    results = pd.DataFrame(['Linear Regression',train_mse, train_r2, test_mse, test_r2]).transpose()
    results.columns=['method','train_mse','train_r2','test_mse','test_r2']
    print(results)
    
# Plot the actual data points
    plt.scatter(x_test['gpa'], y_test, color='blue', label='Actual')

# Plot the linear regression line
    plt.plot(x_test['gpa'], y_test_predict, color='red', linewidth=2, label='predicted')

    plt.xlabel('gpa')
    plt.ylabel('ec')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

def logisticRegression():
    # set x and y 
    y = data['admission_status']
    x = data[['gpa','sat','ec', 'ld']]
    
    # split into test and train data
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4, random_state = 0)
    lr = LogisticRegression(max_iter = 1000000)
    lr.fit(x_train,y_train)
    
    # train data
    y_train_predict = lr.predict(x_train)
    y_test_predict = lr.predict(x_test)

    #heatmap
    after_corr_matrix = x.corr()
    sns.heatmap(after_corr_matrix, annot=True)
    plt.title('Logistic Regression correlation')
    plt.show()
    #confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_predict)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    #evaluate model accuracy
    acc = accuracy_score(y_test,y_test_predict)
    print(f"accuracyscore {acc}")
    sns.set(style='whitegrid')
    sns.pairplot(data=data, hue='admission_status', markers=["o", "s", "D"], palette='Set1')
    plt.show()
    
def decisionTree():
    #set x and y 
    y = data['admission_status']
    x = data[['gpa','sat','ec', 'ld']]
    
    # split into test and train data
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4, random_state = 0)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_test_predict = dt.predict(x_test)
    y_train_predict = dt.predict(x_train)

def randomForest():
    #set x and y
    
    y = data['admission_status']
    x = data[['gpa','sat','ec', 'ld']]

    # split into test and train data
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4, random_state = 0)
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)
    y_train_predict = rf.predict(x_train)
    y_test_predict = rf.predict(x_test)

    # Get feature importances
    feature_importances = rf.feature_importances_

    print("Important Features")
    important_features = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})
    print(important_features)

# Sort the DataFrame by importance
    important_features = important_features.sort_values(by='Importance', ascending=False)

# Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(important_features['Feature'], important_features['Importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('RandomForest Feature Importances')
    plt.xticks(rotation=45)
    plt.show()
    
    # Create a DataFrame to display feature importances
    print("Important Features")
    important_features = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})
    print(important_features)
    
    #heatmap
    after_corr_matrix = x.corr()
    sns.heatmap(after_corr_matrix, annot=True)
    plt.title('RandomForest Correlation')
    plt.show()
    
    #evaluate model accuracy
    
    acc = accuracy_score(y_test, y_test_predict)
    print(f"accuracy: {acc}")
    conf_matrix = confusion_matrix(y_test, y_test_predict)
    print("Confusion Matrix:")
    print(conf_matrix)
    
def knn():
    #set x and y
    
    y = data['admission_status']
    x = data[['gpa','sat','ec', 'ld']]

    # split into test and train data
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4, random_state = 0)
# Create and fit a K-NN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

# Predict admission status
    y_test_pred = knn.predict(x_test)

    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {accuracy}")

    confusion = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

#grid search or random search
#def gridsearch():
    
def statsSummary():
    print("stats summary")
    summary = data.describe()
    print(summary)
    
#main 
linearRegression()
logisticRegression()
randomForest()
knn()
statsSummary()
