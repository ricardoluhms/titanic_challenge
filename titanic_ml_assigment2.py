
# %%
# ### Import Core Libraries

# linear algebra
import numpy as np
# data processing
import pandas as pd 
# data plot
import matplotlib.pyplot as plt

# ### Import Neural Network Libraries
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import embedding

# ### Create specific functions to prepare data

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
            print(" ")
            print(feature, " data has more than 200 unique values" )
        else:
            feature_pack [feature] = feauted_data.values
            print(" ")
            print (feature, " has uniques: \n", feauted_data.values)

    return feature_pack

def cabin_code (feature_pack,df):
    look_up = {}
    for obj in feature_pack["Cabin"]:
        obj = str(obj)
        spliter =" "
        obja = obj.split(spliter)[0]
        if obja == "nan":
            code = "T"
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

def replace_na_age (df_train,df_test,by = ["Pclass"]):
    ### replacing age - train
    mask_age_is_na_t1 = df_train["Age"].isna()
    mask_age_is_not_na_t1 = mask_age_is_na_t1 == False
    va_train = df_train[mask_age_is_not_na_t1][["Age","Pclass"]] # valid_age_df_train
    na_train = df_train[mask_age_is_na_t1][["Age","Pclass"]]
    
    ### replacing age - test
    mask_age_is_na_t2 = df_test["Age"].isna()
    mask_age_is_not_na_t2 = mask_age_is_na_t2 == False
    va_test = df_test[mask_age_is_not_na_t2][["Age","Pclass"]] # valid_age_df_test
    na_test = df_test[mask_age_is_na_t2][["Age","Pclass"]]
    
    pclasses = df_train["Pclass"].drop_duplicates().to_list()
    va_all = va_train.append(va_test)
    
    for class_value in pclasses:
        class_mask = va_all["Pclass"] == class_value
        class_all = va_all[class_mask]["Age"]
        min_age = np.round(class_all.quantile(0.25))
        max_age = np.round(class_all.quantile(0.75))
        random_valid_ages = np.arange(min_age,max_age,1)
        print("")
        print("#"*10, " Class: ", class_value ," Age  Data", "#"*10)
        print("Quantile 25% ", class_all.quantile(0.25),"Quantile 75% " ,class_all.quantile(0.75))
        print("Mean: ", class_all.mean(),"Median: " ,class_all.median())
        spec_class_mask1 = na_train["Pclass"]== class_value
        spec_class_mask2 = na_test["Pclass"]== class_value
        
        for index in na_train[spec_class_mask1].index:
            random_age = np.random.choice(random_valid_ages, size=((1)))[0]
            df_train.at[index, "Age"] = random_age
            
        for index in na_test[spec_class_mask2].index:
            random_age = np.random.choice(random_valid_ages, size=((1)))[0]
            df_test.at[index, "Age"] = random_age   
    
    return df_train,df_test

# ### Load Test and Train Data

df_test = pd.read_csv("C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/test.csv")
df_train = pd.read_csv("C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/train.csv")

# ### Check train data to find:
# 
# <ol>
# <li>which has NaN (Not a Number Values)</li>
# <li>which are non - numeric values which can be replaced by numeric values</li>
# </ol>  

## df.isnull() or df.isna() returns an array with true or false - if you apply .sum() aftwars it will count how many attributes will be NaN
print(df_train.isnull().sum())

## df.isnull() or df.isna() returns an array with true or false - if you apply .sum() aftwars it will count how many attributes will be NaN
print(df_test.isnull().sum())

# ** Both age and Embarked have empty values and they must be filled.<br>
# ** How will you fill them will have effect on the learning process <br>
# ** age for instance can be filled with mean age value<br>
# ** embarked can be filled with a letter  different from the already existing ['S' 'C' 'Q' ] e.g. 'U' as Unknown<br>

df_train,df_test = replace_na_age (df_train,df_test)

### check is the values are correctly replaced
print("Train Age null values: ", df_train["Age"].isnull().sum())
print("Test Age null values: ", df_test["Age"].isnull().sum())

### replacing Embarked - train
mask_e = df_train["Embarked"].isna()
for index in df_train[mask_e].index:
    df_train.at[index, "Embarked"] = "U"
    
### replacing Fare - test
mask_f = df_test["Fare"].isna()
for index in df_test[mask_f].index:
    df_test.at[index, "Fare"] = df_test["Fare"].mean()   

### check is the values are correctly replaced
print("Train Embarked null values: ", df_train["Embarked"].isnull().sum())
print("Test Fare null values: ", df_test["Fare"].isnull().sum())

# the values in the dataframe which has non - numeric values "objects" 
# and return its unique values if they have less than 200 unique values
feature_pack = check_object_var(df_train)

#### since the cabins letter shows an approximate position where the survivor might be in the ship 
#### it is important to convert this information into umbers

### fillna will replace the value NaN into a string name "nan" so it will not be interpreted as NaN.
### all operations must be done in each dataframe to keep them in the same format
df_train["Cabin"] = df_train["Cabin"].fillna("nan")
df_test["Cabin"] = df_test["Cabin"].fillna("nan")
look_up1, df_train = cabin_code(feature_pack,df = df_train)
look_up2, df_test = cabin_code(feature_pack,df = df_test)
# surive_cabin = df_train[["Survived","Cabin_Code"]]
# surive_embarked = df_train[["Survived","Embarked"]]
# plot_by_cabin = surive_cabin.hist(by =  ["Cabin_Code"])
# plot_by_embarked = surive_embarked.hist(by =  ["Embarked"])

mask_sex1 = df_test["Sex"] == 'male' ; mask_sex2 = df_train["Sex"] == 'male' 
mask_sex3 = df_test["Sex"] == 'female'; mask_sex4 = df_train["Sex"] == 'female'
df_test.at[mask_sex1,"Sex"] = 0; df_train.at[mask_sex2,"Sex"] = 0
df_test.at[mask_sex3,"Sex"] = 1; df_train.at[mask_sex4,"Sex"] = 1

mask = df_train["Survived"]==0
valid_columns= ["Pclass","Age","SibSp","Parch","Fare","Cabin_Code_Value"]
df_train[mask][["Pclass","Age","SibSp","Parch","Fare","Cabin_Code_Value"]].corr()

mask = df_train["Survived"]==1
v_columns= ["Pclass","Age","SibSp","Parch","Fare","Cabin_Code_Value"]
df_train[mask][valid_columns].corr()

df_train_normalized = (
    ( df_train[v_columns] - df_train[v_columns].min() )/
    ( df_train[v_columns].max() - df_train[v_columns].min() ) )

y_unbatched = df_train["Survived"].values
df_train_normalized.dtypes
df_train_normalized= df_train_normalized.astype(float)

array = df_train_normalized.values
print(" Array type: ", type(array))
print(" Rows: ", array.shape[0], " Features: ", array.shape[1])

### batch size = number of rows fed in the neural network
def batch_split(array,batch_size = 892, type = "input"):
    batched = []
    batches = ( array.shape[0] // batch_size ) + 1
    for i in range(batches):
        if i == 0:
            fr = i
            to = ((i + 1) * batch_size)
        else:
            fr = to +1
            to = ((i + 1) * batch_size ) 
            if to > array.shape[0]:
                to = array.shape[0]+1
        if len(array.shape) == 1:
            row = array.shape[0]
            array = array.reshape(row,1)
        if type == "input":
            batch_tensor = torch.tensor( array[fr:to]).float().requires_grad_(True) #.cpu()           
        else:
            batch_tensor = torch.tensor( array[fr:to] ).float().requires_grad_(True) #.cpu()      
        batched.append(batch_tensor)
    return batched

x_batches = batch_split(array , type = "input")
y_batches = batch_split(y_unbatched , type = "output")

class MyNet(nn.Module):
    def __init__(self, min_neuron_per_layer = 10):
        super(MyNet, self).__init__()
        features_input = 6
        output_labels = 1 #"Survived or not"
        output_features = min_neuron_per_layer
        self.fc_layer1 = nn.Linear(features_input,output_features)
        self.fc_layer2 = nn.Linear(output_features,output_features*2)
        self.fc_layer3 = nn.Linear(output_features*2,output_features*2)
        self.fc_layer4= nn.Linear(output_features*2,output_labels)
        self.dropout1 = nn.Dropout(0.15)

    def forward(self,x):

        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.fc_layer2(x)
        x = F.relu(x)
        x = self.fc_layer3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc_layer4(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(x_batches, y_batches, epochs =100, learning_rate = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNet(min_neuron_per_layer=2).to(device)
    #print("#"*15, "Network Structure", "#"*15); print(model)
    criterion = torch.nn.MSELoss(reduction='sum')#.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
    loss_overall_data = []
    
    if len(x_batches) == len(y_batches):
        for epoch in range(epochs):
            for i, x in enumerate(x_batches):
                y = y_batches[i].to(device)
                x = x.to(device)
                y_pred = model.forward(x)
                # Calculating the model loss
                #from IPython import embed; embed()
                loss = criterion(y_pred, y)
                # setting gradient to zeros
                #from IPython import embed; embed()
                #optimizer.param_groups
                #optimizer.zero_grad()
                #from IPython import embed; embed()
                # update the gradient to new gradients
                loss.backward()
                optimizer.step()

                if (i % 300 == 0):
                    print("")
                    print("Epoch {} - loss: {}".format(epoch, loss.item()))
                    #print(list(optimizer.param_groups)[0])
    else:
        print("Input batches 'x' are different from the ouput 'y' ")
    return model,loss_overall_data



model,loss_overall_data = train(x_batches, y_batches)







# %%
