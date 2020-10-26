# linear algebra
import numpy as np
from numpy.core.defchararray import index 

# data processing
import pandas as pd 

# data plot
import matplotlib.pyplot as plt

# Neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

# PassengerId is the unique id of the row and it doesn't have any effect on target
# Survived is the target variable we are trying to predict (0 or 1):
# 1 = Survived; # 0 = Not Survived
# Pclass (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):
# 1 = Upper Class; # 2 = Middle Class; # 3 = Lower Class
# Name, Sex and Age are self-explanatory
# SibSp is the total number of the passengers' siblings and spouse
# Parch is the total number of the passengers' parents and children
# Ticket is the ticket number of the passenger
# Fare is the passenger fare
# Cabin is the cabin number of the passenger
# # Embarked is port of embarkation and it is a categorical feature which has 3 unique values (C, Q or S):
# C = Cherbourg; # Q = Queenstown
# S = Southampton




df_test = pd.read_csv("C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/test.csv")
df_train = pd.read_csv("C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/train.csv")

def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def correlation (df):
    print("")
    ### get the overall correlation of each columns
    overall_correlation_streght = np.abs(df.corr()).sum()-1 
    #### absolute is used to check the strength regardless of being proportional or proportially inverse
    #### sum() - 1 - because the correlation of a column with itself equals one, then all the columns has this plus one which should be subtracted
    average_correlation = (overall_correlation_streght / len(df.columns))*100
    normalized_average_correlation = average_correlation/ average_correlation.max()

    avg_strengh = normalized_average_correlation.mean()
    min_strengh = normalized_average_correlation.min()

    strong_features_mask = normalized_average_correlation >= avg_strengh
    strong_features = normalized_average_correlation[strong_features_mask]
    print( "Strong Correlation Features: \n", strong_features, "\n")
    weak_features = normalized_average_correlation[strong_features_mask==False]
    weakest_features = normalized_average_correlation[normalized_average_correlation == min_strengh]
    print( "Weak Correlation Features: \n", weak_features, "\n")
    print( "Weakest Correlation Features: \n", weakest_features, "\n")

    ### normalized_average_correlation shows which columns attribute does is not correlated to other 

def check_object_var(df):
    obj_coulumns = df.dtypes[df.dtypes == object]
    feature_pack = {}
    for feature in obj_coulumns.index:
        feauted_data = df[feature].drop_duplicates()
        if len(feauted_data) > 200:
            print(feature, " data has more than 200 unique values" )
        else:
            feature_pack [feature] = feauted_data.values
            print (feature, " has uniques: \n", feauted_data.values)

    return feature_pack

def cabin_code (feature_pack,df):
    look_up = {}
    for obj in feature_pack["Cabin"]:
        obj = str(obj)
        spliter =" "
        obja = obj.split(spliter)[0]
        if obja == "nan":
            code = "nan"
        else:
            code = obja[0]
        
        ### add cabin number
        look_up[obj] = code


    df2 = pd.DataFrame(look_up.values(), columns= ["id"]).drop_duplicates().sort_values(by = ["id"]).reset_index(drop=True)
    df["Cabin_Code"] = ""
    df["Cabin_Code_Value"] = 10

    for cabin_obj in look_up.keys():
        mask = df["Cabin"] == cabin_obj
        simple_code = look_up[cabin_obj]
        mask2 = df2["id"] == simple_code
        
        simple_code_value = df2[mask2].index[0]
        #from IPython import embed; embed()
        df.loc[mask, "Cabin_Code"] = simple_code
        df.loc[mask,"Cabin_Code_Value"] = simple_code_value
        #from IPython import embed; embed()

    return look_up, df


class MyNet(nn.Module):
    def __init__(self,input_array,min_neuron_per_layer = 10):
        super(Net, self).__init__()


        features_input = input_array.shape[1]
        features_input = input_array.shape[1]
        output_labels = 2 #"Survived or not"
        nplay = min_neuron_per_layer
        self.fc_layer1 = nn.Linear(features_input,nplay)
        self.fc_layer2 = nn.Linear(nplay,nplay*2)
        self.fc_layer3 = nn.Linear(nplay*2,nplay*2)
        self.fc_layer4 = nn.Linear(nplay*2,nplay*2)
        self.fc_layer5 = nn.Linear(nplay*2,)

        self.dropout1 = nn.Dropout(0.15)


    def forward(self, x):

        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.fc_layer2(x)
        x = F.relu(x)
        x = self.fc_layer3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc_layer4(x)
        x = F.relu(x)
        x = self.dropout1(x)

        output = F.log_softmax(x, dim=1)
        return output

my_nn = Net()
print(my_nn)
### describe
### check null values with df.isnull() -> true/false array
### df.isnull().sum() shows how many null values per columns
### which value should replace each null?

correlation(df_train)

df_train["Cabin"] = df_train["Cabin"].fillna("nan")
# s
df_test["Cabin"] = df_test["Cabin"].fillna("nan")

feature_pack = check_object_var(df_train)

look_up, df = cabin_code(feature_pack,df = df_train)
### fill na

df2 = df[["Survived","Cabin_Code"]]
ax = df2.plot.hist(by = ["Cabin_Code"])
plt.show()
from IPython import embed; embed()