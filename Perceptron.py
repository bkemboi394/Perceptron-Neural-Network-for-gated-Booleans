# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:55:29 2022

-Perceptron NN that aims to learn the gated boolean logics(OR,AND,NOR,NAND)
-Applies stochastic gradient descent as it uses one single instance of training data each time to update
the weights.
-https://www.techtarget.com/whatis/definition/logic-gate-AND-OR-XOR-NOT-NAND-NOR-and-XNOR#:~:text=The%20XOR%20(%20exclusive%2DOR%20),both%20inputs%20are%20%22true.%22

"""
import numpy as np


#The threshold function
def predict(row, weights):
    activation = weights[0] #bias weight
    #print(len(row))
    for i in range(0,len(row)-1):
            activation += weights[i + 1] * row[i]
    if activation >= 0:
         #print(activation)
         return 1
    else:
         #print(activation)
         return 0

#Function for updating and optimizing the weights 
def train_weights(train, l_rate, n_epoch):
    
    #weights are initialized via randomization
    weights = [ np.random.rand() for i in range(len(train[0]))] 
    
 #Since the last column is class-label,the extra weight at beginning is bias weight
 
    for epoch in range(n_epoch): #number of times that we'll train
        sumsquared_error = 0.0#to keep track of progress of each training session
        for row in train: #the x-values
            prediction = predict(row, weights) #prediction
            error = round((row[-1] - prediction),3) #actual-predicted
            sumsquared_error += round(error**2,3) #for consistency
           
            weights[0] = round((weights[0] + l_rate * error),3) #updating bias weight

            
            for i in range(len(row)-1):
            
            #updating the rest of the weights
              weights[i + 1] =round((weights[i + 1] + l_rate * error * row[i]),3)
  
             
        #Printing out the values for each epoch
        print("epoch:",epoch, "Learning_rate=", l_rate,"error=", sumsquared_error)
        if sumsquared_error == 0.0:
                 break
    return weights
    

#Using both predict and weight functions to implement the perceptron

def perceptron(train, test, l_rate, n_epoch):
    predictions = []
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
        
        print(row[-1],prediction) #Comparing actual vs. predicted output
    print("optimal weights:",weights) #printing out optimal weight
    
    return(predictions)

#Testing out our perceptron with booleans AND,OR,NAND,NOR
print("Applying the perceptron to gated booleans as follows:")
#Learning the AND boolean
print("\nLearning the AND boolean\n")
#Training and test data for AND boolean 
train_1=[[1,1,1],
       [1,0,0],
       [0,1,0],
       [0,0,0]]
test_1=[[0,1,0],
      [1,0,0],
      [0,0,0],
      [1,1,1]]
#After testing out several l_rates settled for 0.1
l_rate = 0.1
n_epoch = 10
#Calling our perceptron on the AND

AND_pred=perceptron(train_1, test_1, l_rate, n_epoch)

#Learning the OR boolean
print("\nLearning the OR boolean\n")
#Training and test data for OR 
train_2=[[1,1,1],
       [1,0,1],
       [0,1,1],
       [0,0,0]]
test_2=[[0,1,1],
      [1,0,1],
      [0,0,0],
      [1,1,1]]
#Calling perceptron function on OR
OR_pred=perceptron(train_2, test_2, l_rate, n_epoch)

#Learning the NAND boolean
print("\nLearning the NAND boolean\n")
#Training and test data for NAND
train_3=[[1,0,1],
        [0,1,1],
        [0,0,1],
        [1,1,0]]
test_3=[[0,1,1],
      [1,0,1],
      [1,1,0],
      [0,0,1]]
#Calling perceptron function on NAND
NAND_pred=perceptron(train_3, test_3, l_rate, n_epoch)

#Learning the NOR boolean
print("\nLearning the NOR boolean\n")
#Training and test data for NOR
train_4=[[1,1,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]]
test_4=[[0,1,0],
      [1,0,0],
      [0,0,1],
      [1,1,0]]

NOR_pred=perceptron(train_4, test_4, l_rate, n_epoch)

"""
-From the output, it looks like the OR gated boolean is the fastest to learn as the error goes to zero just 
after 3 epochs
-I can now apply this algorithm to solve a problem with a much larger data set

"""
