import math
import numpy as np
import pandas as pd
# from sklearn.cross_validation import train_test_split
names =['index', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',  'x7', 'x8', 'x9','class']

orig_dataframe = pd.read_csv('glass.csv', header=None, names=names)
orig_dataframe.set_index('index', inplace=True)


def selectData(dataframe, index_start, index_end):
    new_dataframe = dataframe.loc[index_start:index_end]
    return new_dataframe


def make_prediction(inputVector,train_data):
    d1 = train_data.loc[train_data['class'] == 1]
    d2= train_data.loc[train_data['class'] == 2]
    a = len(d1)
    b = len(d2)
    if a>b:
        return 1
    else:
        return 2

# reutrns postive cases
def prediction_of_test_data(test_data):
    sel_data = test_data.iloc[:, 0:9]
    print("LENTHG OF TEST DATA")
    print(len(test_data))
    # print(test_data)
    print("Lenth of Train data")
    print(len(t_data))
    # print(t_data)
    postive_cases = 0
    for row in range(len(test_data)):
        row_vector = sel_data.iloc[row,0:9]
        row_vector_2 = row_vector.ravel()
        algo_prediction =make_prediction(row_vector_2,t_data)
        predicted_values_array.append(algo_prediction)
        class_array = np.array([test_data['class']])
        test_data_real_answer = class_array[0,row]

        if algo_prediction == test_data_real_answer:
            postive_cases = postive_cases+ 1
    print("Printing error for each fold:")
    print((postive_cases/len(test_data))*100)
    print("positive cases for each fold:")
    print(postive_cases)
    print('END OF ONE FOLD')
    print('--------------------------------------------------------------------------')
    print()
    print()
    return postive_cases





predicted_values_array =[]
total_postive_predictions=0
for i in range(5):
    if i==0:
        t_data = selectData(orig_dataframe,41,200)
    elif i==1:
        t_data_1 = selectData(orig_dataframe,1 ,40 )
        t_data_2 = selectData(orig_dataframe, 81, 200)
        t_data = pd.concat([t_data_1, t_data_2])

    elif i==2:
        t_data_1 = selectData(orig_dataframe,1 ,40 )
        t_data_2 = selectData(orig_dataframe, 81, 200)
        t_data = pd.concat([t_data_1, t_data_2])

    elif i==3:
        t_data_1 = selectData(orig_dataframe,1 ,40 )
        t_data_2 = selectData(orig_dataframe, 81, 200)
        t_data = pd.concat([t_data_1, t_data_2])

    else:
        t_data = selectData(orig_dataframe,1,160)

    testing_data = selectData(orig_dataframe,1+40*i,40+i*40)
    total_postive_predictions += prediction_of_test_data(testing_data)

print()
print()
print("total Prediction error for CV of ZERO-R")
prediction_error = (total_postive_predictions/len(orig_dataframe))*100
print(prediction_error)
print("Predicted values array for ZERO -R:")
print(predicted_values_array)
print(len(predicted_values_array))





