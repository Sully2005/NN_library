from main import *



data = pd.read_csv('ufc-cleaned.csv')
data = np.array(data)

Y = data[:, -1]
X = data[:, :-1]



#normalise the data
for i in range(56):
    column = X[:, i]
    mean = column.mean()
    std = column.std()
    if std == 0:
        std = 0.00001
    
    X[:, i] = (column - mean) / std


X_train, Y_train, X_test, Y_test = split_training_data(X,Y,0.2)

Y_train = Y_train.reshape(-1,1)

network = [
    Dense(56, 40), 
    Tanh(),
    Dense(40, 1), 
    Sigmoid()
]

train(network, X_train, Y_train, mean_squared_error, mean_squared_error_derivative, 0.25, 1200)


accuracy = get_Accuracy(X_test, Y_test, network)



print("Accuracy: ", accuracy)



#Not an amazing accuracy especially considering the fact that if the model were to pick red to win everytime
#it would be correct about 58 percent of the time. This can be improved upon using XGboost and more training examples.
#Another thing that should be noticed is that it is difficult to use neural networks to predict UFC fights as it is hard 
#to capture the quality of opponent of every fighter and how theri styles interact. 

# Iteration: 0,   Error: 0.2777115653253852
# Iteration: 100,   Error: 0.20766497808897819
# Iteration: 200,   Error: 0.20181846982516402
# Iteration: 300,   Error: 0.19767560786212207
# Iteration: 400,   Error: 0.19367676707427442
# Iteration: 500,   Error: 0.18929765343780264
# Iteration: 600,   Error: 0.18440534362267155
# Iteration: 700,   Error: 0.17916971751982944
# Iteration: 800,   Error: 0.17375840682099503
# Iteration: 900,   Error: 0.1681710712354829
# Accuracy:  62.09075244112579