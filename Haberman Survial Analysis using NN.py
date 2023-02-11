# 1. Problem Statement & Objetive

# Survival Analysis in medical field is used to predict time for a certain event. In our case study, Haberman’s Survival Analysis for breast cancer patients, we will try to determine whether the patient will survive for 5 years after surgery or not.

# 2. Type of analysis to be done

# There are 2 possible outcomes either patient will survive for 5 or above years or will die within 5 years. So, we will treat this as binary classification task and we will try to model it using Neural Network

# 3. About the dataset

# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# - Attribute Information:
#     1. Age of patient at time of operation
#     2. Patient's year of operation
#     3. Number of positive axillary nodes detected
#     4. Survival status (class attribute)
#         1 => the patient survived 5 years or longer
#         2 => the patient died within 5 years

# importing all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import backend as K

dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data",  header=None)

# converting the above downloaded data into a form suitable for DL 

dataset.columns = ["age", "year_of_operation", "no_of_axillary_nodes", "survival_status"]

print(dataset.shape)
print(dataset.columns)
print(dataset.info())
print(dataset.head())

# observations from the above. 

# 1. Dataset has 306 records and 4 attributes including the class variable (survival_status)
# 2. All features are numerical. As for class attribute survival_status, 1 value is encoded as 'the patient survived 5 years or longer' and 2 as 'the patient died within 5 year'

# Data Preparation

print(dataset.describe())

# checking for missing values

dataset.isnull().sum()

# checking for duplicates

duplicateRows = dataset[dataset.duplicated(keep = 'last')]
print('Duplicates', duplicateRows.shape)
print('Original dataset', dataset.shape)
print(duplicateRows)

# remove duplicates

dataset = dataset.drop_duplicates()
print('Modified dataset', dataset.shape)

# since all are numerical variables checking distribution of each variable by drawing histograms

dataset.hist()
plt.show()

# Observations: age has approx gaussian distribution. year_of_operation has mostly uniform distribution except for maybe some outliers in first year. no_of_auxillary_nodes has value 0 for most of the examples with a trailing long tail of other values. survival_status has unequal class distribution.

# checking relevance of each attr w.r.t target and co-relation with each other

sns.pairplot(dataset, hue="survival_status")

# Observation: None of the attributes are contributing greatly to class prediction. All patients having zero or close to zero nodes survived for more than 5 years, this density decreases as # nodes increases for all age patients. no_of_auxillary_nodes seem to be imp feature out of 3, affecting survival_status the most. 

# checking class count

print(dataset['survival_status'].value_counts())
# in terms of percentage
print(dataset['survival_status'].value_counts(1))

# Observation: Its imbalanced dataset. There is much more samples from class 1 than class 2.

# normalize all input features
dataset_norm = dataset.copy()
# using StandardScaler as it follows approx gaussian distribution
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()

dataset_norm['age'] = scaler1.fit_transform(dataset['age'].values.reshape(-1, 1))
dataset_norm['year_of_operation'] = scaler2.fit_transform(dataset['year_of_operation'].values.reshape(-1, 1))
dataset_norm['no_of_axillary_nodes'] = scaler3.fit_transform(dataset['no_of_axillary_nodes'].values.reshape(-1, 1))

dataset_norm.head()

# encoding class attr values as 0/1
dataset_norm['survival_status'] = LabelEncoder().fit_transform(dataset_norm['survival_status'])
print(dataset_norm.head())

# splting data into input & output
X, y = dataset_norm.values[:, :-1], dataset_norm.values[:, -1]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 1. To remove dulpicate data - Removed duplicates, identical rows. Also, Majority of duplicates are from class 1, this will help a little bit in decresing imbalance.
# 2. To impute or remove missing data, if present - missing values not present
# 3. To remove data inconsistencies, if present - No significant data inconsistencies
# 4. To encode categorical data - All features are numerical but we did one hot encode the target variable (0/1)
# 5. Normalization technique used - Only age feature follows gaussian distrbution so we standardized it and for others as they do not have gaussian distribution we went with min-max normalization for them.

# Deep Neural Network Architecture
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 1. Number of layers - The problem we are solving is not very complex with input features only being 3 and the dataset is not large either around 300 records. So, to lower chances of overfitting we went with single hidden layer
# 2. Number of units in each layer - Since we decided on only 1 hidden layer, for starters we kept the neurons size to medium 10, later on deciding on changing it if performance improved by tweaking it.
# 3. Activation function for hidden layers - I chose to go with most common activation for hidden layers i.e. ReLu and then compare its performance with other functions
# 4. Activation function for output layer - Since its a binary classification problem we chose sigmoid activation with one node
# 5. Total Number of trainable parameters - Going with fully connected layers, calculated paramters accrding to rule (n+1) * m, where n is no, of inputs (3) and m is no. of nerons in the dense layer (10). I did not have to set this explicitly while model creation, by default it was considered to be fully connected NN and according;y no. of params seems to have set internally as shown in above model summary. 

# Training the model
# Configure the model for training, by using appropriate optimizers and regularizations

# copied from internet for f1 score metric during training which is not available by default
def f1_score(y_true, y_pred): # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=[f1_score])

classifier_results = model.fit(X_train, 
                    y_train, 
                    epochs=200, 
                    batch_size=8, 
                    verbose=True, 
                    validation_data=(X_test,y_test))


# choice of optimizers and regulizations used and the hyperparameters tuned

# 1. Metric - Since the datatset is imbalanced, instead of accuracy we chose f1 score as metric. Since keras does not have default implementation of this metric we had to copy the custom implementation from internet and provide to keras.
# 2. Hyperpatameters - We tried different values of epochs & batch_size and settled on above values based on the performace of each combination. Increasing batch_size from 8 to 16 did not show any visible difference in performace so kept it at 8.
# 3. Optimizers - We tested f1-score values for 2 generally best choices for optimizer Adam & SGD. For, SGD f1 score came out to be very low so we decided on going with Adam.
# 4. Loss Function - Since our output activation is sigmoid, we kept cross entropy as loss function.

# Test the model

model.evaluate(X_test, y_test)
# output - loss & f1 score


# Conclusion 
# Plotting the training and validation loss

plt.plot(classifier_results.history['loss'])
plt.plot(classifier_results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = (model.predict(X_test) > 0.5).astype("int32")

# confusion matrix
confusion_mtx = confusion_matrix(y_test, predictions)
print('Confustion matrix = \n',pd.DataFrame(confusion_mtx))

# Classification Report    
print('Classfication report = \n', classification_report(y_test, 
        predictions, target_names=['class 0', 'class 1']))


# Solution to solve the business problem discussed

# 1. Since this is survival analysis, this model can be used to predict life expectancy of unknown breast cancer patients who have just came out of surgery in order to plan the course of their treatment.
# 2. However, this dataset is pretty old and hence the model will probably fall short of predicting for present scenarios with fair accuracy. I also faced challange for imbalance in dataset which we tried to overcome with choosing appropriate metric instead of going with accuracy. I did not go with oversampling since the imbalance was not severe.
# 3. I learned that even with starting with generally accepted choices of hyperparameters, activation functions and such, neural network gives faily decent performance which we can further work on improve by finding best parameters by gridSearch and so on. If more relevant data becomes available we can perhaps further the complexity of network from single hidden layer to exploit more performance.
