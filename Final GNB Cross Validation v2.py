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


# 200 examples for both traning and test
# Calculating P(C1)
def prior_prob(df,classs):
    total_examples = len(df)
    df_c = df.loc[df['class']==classs]
    num_examples_for_class = len(df_c)
    prob = num_examples_for_class/total_examples
    return prob

def give_mean_variance(df,attribute,classs):
    arr = np.array(df[attribute])
    mean = np.mean(arr)
    variance = np.var(arr, ddof=1)
    return mean, variance


def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = np.pi
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


# xi is give for each example we plug into system
#df, here is the training data
# attribute is the attribute of xi
# classs is the given class
def likelihood_prob(xi,attribute,df,classs):
    df_select = df.loc[df['class'] == classs, [attribute]]
    mean, var = give_mean_variance(df_select, attribute, classs)
    sd = math.sqrt(var)
    prob = normpdf(xi,mean,sd)
    return prob


def log_prediction_prob_for_each_class(inputVector,classs):
    log_sum_of_likelihood_prob = 0
    for i in range(len(inputVector)):
        xi = inputVector[i]
        attribute = 'x' +str(i+1)
        simple_prob = likelihood_prob(xi, attribute, t_data, classs)
        log_prob = math.log1p(simple_prob)
        log_sum_of_likelihood_prob += log_prob
    log_prior_prob = math.log1p(prior_prob(t_data, classs))
    return (log_sum_of_likelihood_prob+log_prior_prob)


def make_prediction(inputVector):
    lis= []
    for i in range(2):
        lis.append(log_prediction_prob_for_each_class(inputVector, i+1))
    if lis[0]>=lis[1]:
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
        algo_prediction =make_prediction(row_vector_2)
        predicted_values_array.append(algo_prediction)
        class_array = np.array([test_data['class']])
        test_data_real_answer = class_array[0,row]

        if algo_prediction == test_data_real_answer:
            postive_cases = postive_cases+ 1
    print("Printing accuracy for each fold:")
    print((postive_cases/len(test_data))*100)
    print("positive cases for each fold:")
    print(postive_cases)
    print('END OF ONE FOLD')
    print('--------------------------------------------------------------------------')
    print()
    print()
    return postive_cases




# This array stores the predicted values for ALL 200 test cases (for all 5 folds)

predicted_values_array =[]
# gives total no. of positive predictions in the whole data.

# note that t_data is the training data, it is the global variable.
# gets defind for each fold in the below for-loop
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
print("total Prediction accuracy for CV")
prediction_error = (total_postive_predictions/len(orig_dataframe))*100
print(prediction_error)
print("Predicted values array:")
print(predicted_values_array)
print(len(predicted_values_array))
print()
print("Prediction for example no." +str(20))
print(predicted_values_array[19])
print()
print("Prediction for example no." +str(60))
print(predicted_values_array[59])
print()
print("Prediction for example no." +str(100))
print(predicted_values_array[99])
print()
print("Prediction for example no." +str(140))
print(predicted_values_array[139])
print()
print("Prediction for example no." +str(180))
print(predicted_values_array[179])
