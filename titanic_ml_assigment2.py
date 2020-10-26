
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
            array2 = array.copy()
            mask = array == 0
            array2[mask] = 1
            array2[~mask] = 0
            array_out = np.hstack([array,array2])

        if type == "input":
            batch_tensor = torch.tensor( array[fr:to]).float().requires_grad_(True) #.cpu()           
        else:
            batch_tensor = torch.tensor( array_out[fr:to] ).float().requires_grad_(True) #.cpu()      

        batched.append(batch_tensor)
    return batched
# 892
x_batches = batch_split(array , type = "input")
y_batches = batch_split(y_unbatched , type = "output")

def train(x_batches, y_batches, features_input = 6, features_output = 2, min_neuron_per_layer = 50, epochs = 8000, lr=2e-4):
    output_feat = min_neuron_per_layer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
                        nn.Linear(features_input,output_feat),
                        nn.ReLU(),
                        nn.Linear(output_feat,output_feat*2),
                        nn.ReLU(),
                        nn.Linear(output_feat*2,output_feat*2),
                        nn.ReLU(),
                        nn.Linear(output_feat*2,features_output),
                        ).to(device)

    criterion = torch.nn.MSELoss(reduction='mean')#.to(device)
    criterion2 = torch.nn.MSELoss(reduction='sum')#.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-6, momentum=0.8)
    

    for epoch in range(epochs):
        for i, x in enumerate(x_batches):
            y = y_batches[i].to(device)
            x = x.to(device)
            y_pred = model.forward(x)
            y_pred2 = F.softmax(y_pred,dim=1)
            test = (y_pred2 - y).sum()
            loss = criterion(y_pred, y)
            tt_error = criterion2(y_pred, y)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0: 
            print("#"*100)
            print("Epoch {} - loss: {} - prediction error: {}  test: {}".format(epoch, loss.item(), tt_error.item(), test))
            print("")
    #from IPython import embed; embed()
    return model.cpu()

trained_model = train(x_batches, y_batches, features_input = 6, min_neuron_per_layer = 50, epochs = 8000, lr=2e-4)
x = x_batches[0]
y_pred = trained_model.forward(x).detach().clone().cpu().numpy() 
path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/saved_models"
file = "4fc_50n_8000_epochs.pth" ### pytorch file
file_path = path + "/" + file
torch.save(trained_model.state_dict(), file_path)
# - x reprent 6 features from each passenger of the ship - it is the input for the neural network
# - y_pred = the output array
#    explaining the variable y_pred and the functions within it from left to right:
#     - trained_model.forward(x) get the "x" tensor (pytorch format) which has 892 passengers x 6features
#     - return the tensor output which has 892 passengers x 2features 
#           (first column feature is the probability of the passenger being alive, second column feature is the probability of the passenger is dead)
#     - to remove the learning features from it we apply detach()
#     - then we clone the output (same as copy in numpy)
#     - to be able to use the tensor we pass it to the cpu
#     - lastly we convert it to numpy   
survival_status = (y_pred[:,0] > y_pred[:,1]) * 1
# if the probability value of the first column is higher than the second columns it means that the passenger survived.
# the boolean matrix is multiplied by 1 to convert true and false to 1 and 0.
df_train["Survived_Prediction"] = survival_status
mask = df_train["Survived"] == df_train["Survived_Prediction"]
prediction_ok = df_train[mask] 
overall_TP = prediction_ok[prediction_ok["Survived"] == 1 ] ### Model predicted that the person would survive and got it right
overall_TN = prediction_ok[prediction_ok["Survived"] == 0 ] ### Model predicted that the person would not survive and got it right

prediction_nok = df_train[~mask]
overall_FP = prediction_nok[prediction_nok["Survived_Prediction"] == 1 ] ### Model predicted incorrectly that the person would survive and got it wrong
overall_FN = prediction_nok[prediction_nok["Survived_Prediction"] == 0 ] ### Model predicted incorrectly that the person not would survive and got it wrong


precision = len(overall_TP)/(len(overall_TP)+len(overall_FP))
recall = len(overall_TP)/(len(overall_TP)+len(overall_FN))
accuracy = len(prediction_ok) / len(df_train)
print("")
print( "Accuracy: {} - Precision: {} - Recall: {}".format( accuracy, precision, recall))

### where the model failed?
### is there any pattern?

female_FN = overall_FN[overall_FN["Sex"]==1]


from IPython import embed ; embed()