# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The term neural network refers to a group of interconnected units called neurons that send signals to each other. While individual neurons are simple, many of them together in a network can perform complex tasks. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression is a method for understanding the relationship between independent variables or features and a dependent variable or outcome. Outcomes can then be predicted once the relationship between independent and dependent variables has been estimated.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.


## Neural Network Model

![Screenshot 2024-03-04 135147](https://github.com/Joshitha-YUVARAJ/basic-nn-model/assets/145742770/97455801-8861-46b7-a829-c3fb2501cecb)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:YUVARAJ JOSHITHA
### Register Number:212223240189
```

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('ex-1').sheet1
rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head(15)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['Input']].values
y = df[['Output']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
ai=Sequential([
    Dense(4,activation='relu',input_shape=[1]),
    Dense(3,activation='relu'),
    Dense(1),

])

ai.compile(optimizer="rmsprop",loss="mse")

ai.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

ai.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

ai.predict(X_n1_1)

```
## Dataset Information
![Screenshot 2024-03-04 135851](https://github.com/Joshitha-YUVARAJ/basic-nn-model/assets/145742770/4a2f749b-b79a-4585-b3d0-34ee8afa91ab)

![Screenshot 2024-03-04 135906](https://github.com/Joshitha-YUVARAJ/basic-nn-model/assets/145742770/53b6b9ac-1c9f-4bfa-a4b7-46968daa4bcc)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-03-04 140014](https://github.com/Joshitha-YUVARAJ/basic-nn-model/assets/145742770/6f98a736-5e22-4963-83f9-745201b102ab)


### Test Data Root Mean Squared Error

![Screenshot 2024-03-04 135957](https://github.com/Joshitha-YUVARAJ/basic-nn-model/assets/145742770/fc90d363-23ed-43e3-9cc0-d1b4e8401215)

![Screenshot 2024-03-04 140029](https://github.com/Joshitha-YUVARAJ/basic-nn-model/assets/145742770/94f12497-14d2-45b9-aa6f-84faabd64cd7)

### New Sample Data Prediction
![Screenshot 2024-03-04 140038](https://github.com/Joshitha-YUVARAJ/basic-nn-model/assets/145742770/0301b044-d114-49e1-a159-463fbf09250f)


## RESULT

The model is successfully created and the predicted value is close to the actual value.
