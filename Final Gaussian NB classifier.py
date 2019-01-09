import math
import numpy as np
import pandas as pd
# from sklearn.cross_validation import train_test_split
# not using sklearn in this code.

names =['index', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',  'x7', 'x8', 'x9','class']

orig_dataframe = pd.read_csv('glass.csv', header=None, names=names)
orig_dataframe.set_index('index', inplace=True)

def selectData(dataframe, index_start, index_end):
    new_dataframe = dataframe.loc[index_start:index_end]
    return new_dataframe

#  200 examples for both training and test
# Calculating P(C1)
def prior_prob(df,classs):
    total_examples = len(df)
    df_c = df.loc[df['class']==classs]
    num_examples_for_class = len(df_c)
    prob = num_examples_for_class/total_examples
    # print('P(C) for class' +str(classs)+'is')
    # print(prob)
    return prob

def give_mean_variance(df,attribute,classs):
    # the data is already selected by the previous function.
    arr = np.array(df[attribute])
    mean = np.mean(arr)
    variance = np.var(arr, ddof=1)
    # if attribute =='x7' and classs==2:
    #     print("Mean:", mean)
    #     print("Variance: ", variance)
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
    # print("I am the earliest dataframe:")
    # print(df)
    df_select = df.loc[df['class'] == classs, [attribute]]
    # print("I am no. 2")
    # print(df)
    mean, var = give_mean_variance(df_select, attribute, classs)
    sd = math.sqrt(var)
    prob = normpdf(xi,mean,sd)
    return prob
#t_data is the training data.
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
    # hardcode
    lis= []
    for i in range(2):
        lis.append(log_prediction_prob_for_each_class(inputVector, i+1))
    if lis[0]>=lis[1]:
        return 1
    else:
        return 2

def prediction_error(test_data):
    sel_data = test_data.iloc[:, 0:9]
    print("LENTHG OF TEST DATA")
    print(len(test_data))
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

    return (postive_cases/len(test_data))*100




predicted_values_array =[]
#t_data is the training data.
# Global variable
t_data = selectData(orig_dataframe,1,200)


# in this case training data (t_data) is the same as test data, so we put t_data in the prediction error function.
error = prediction_error(t_data)
print("Total prediction accuracy:")
print(error)

# This array shows the predicted values for each of the test_data's example.
# thus we can extract the predicted class for each example thru this array.
print("Predicted values array:")
print(predicted_values_array)
print(len(predicted_values_array))

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



