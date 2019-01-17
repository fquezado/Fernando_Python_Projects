import torch.nn as nn
import pandas as pd
import torch

salary_data_set = pd.read_csv('/Users/fquezado/Downloads/simple_regression/Salary_Data.csv')

x = salary_data_set.iloc[:, :-1].values
y = salary_data_set.iloc[:, 1:].values

X_train = torch.Tensor(x)
Y_train = torch.Tensor(y)


# MODEL
class LinRegModel(nn.Module):
    def __init__(self):
        super(LinRegModel, self).__init__()
        self.linear = nn.Linear(1, 1)   # one input in, one output out

    def forward(self, x):
        y_predict = self.linear(x)
        return y_predict


model = LinRegModel()
loss_function = nn.MSELoss()   # Mean Squared loss equation
learning_rate = 0.01  # setting learning rate
optimizer = torch.optim.SGD(model.parameters(), learning_rate)  # confused by this statement


# TRAINING
for i in range(1000):
    y_predict = model(X_train)  # predicting y by using x
    loss = loss_function(y_predict, Y_train)   # comparing predicted value with trained value

    optimizer.zero_grad()   # zeros out the gradience, so they don't compound
    loss.backward()    # sorta like back propagation
    optimizer.step()    # updates the weights


user_input = float(input("Type number you want to test: "))  # asking user for input of x to output a predicted y
new_variable = torch.Tensor([[user_input]])  # making that said input into a variable that pytorch can recognize
prediction_y = model(new_variable)  # passing in that variable into the model


print("Prediction of", str(user_input), "equals", model(new_variable).data[0][0])
# prints user input and predicted salary
# Need to fix model(new_variable).data[0][0], because I want it to just output number, not "tensor(number)"


# Also need to check how well the generalization is, do I need test set/validation set????
