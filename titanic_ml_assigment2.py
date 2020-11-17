# ### Import Core Libraries

# solve OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os

from torch.jit import TracedModule
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#################################################
# linear algebra
import numpy as np

# data processing
import pandas as pd

# data plot
import matplotlib.pyplot as plt
import seaborn as sns

# ### Import Neural Network Libraries
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import embedding

# Model Helpers and Metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Models
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# Returns a concatenated df of training and test set
def concat_df(train_data, test_data):
    
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

# special correlation function that ranks the features 
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

# function to check object datatype and how many unique values it has  
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

# specific code to the titanic - assigment get the initial letter of the cabin
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
        df.loc[mask, "Cabin_Code"] = simple_code
        df.loc[mask,"Cabin_Code_Value"] = simple_code_value


    return look_up, df

# specific code to the titanic - function to replace null age values by the median value considering each Pclass 
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
    
    df_train["Age_Group"] = pd.qcut(df_train['Age'], 8)
    df_test["Age_Group"] = pd.qcut(df_test['Age'], 8)

    return df_train,df_test

# specific code to the titanic - function to get the name title
def get_name_title(df):
    
    names = df["Name"].to_list()
    df["Title_name"] = ""
    title_names = []
    for name in names:
        code = name.split(",")[1].split(".")[0]
        title_names.append(code)
    df["Title_name"] = title_names
    
    return df

#  specific code to the titanic - get the family size 
def get_family_size(df):
    df["Family_size"] = df["SibSp"] + df["Parch"] + 1
    df["Average_Fare"] = df["Fare"]/df["Family_size"]
    df["Average_Fare_Group"] = pd.qcut(df['Average_Fare'], 5)

    return df

#  function to merge two columns into one (MS excel concatenation)
def group_types(df,typeA, typeB):
    typeC = str(typeA) + "-" + str(typeB) 
    if df[typeA].dtype == pd.Categorical:
        df[typeA] = df[typeA].astype('string')
    if df[typeB].dtype == pd.Categorical:
        df[typeB] = df[typeB].astype('string')

    df[typeC] = df[typeA].map(str)+ "-" + df[typeB].map(str)
    return df

#  function to merge multiple columns into one (MS excel concatenation)
def group_multi(df,types = []):

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    if len(types) < 1:
        print("please add at least 2 attributes in the 'types' list input")
        return None
    elif len(types) == 2:
        df = group_types(df,types[0], types[1])
    else:
        added_types = []
        for num, type in enumerate(types):
            if (type not in added_types) and num < len(types):
                if num == 0:
                    df = group_types(df,types[num], types[num+1])
                    previous_type = types[num] + "-" + str(types[num+1])
                    added_types.append(types[num])
                    added_types.append(types[num+1])
                else:
                    df = group_types(df, previous_type, types[num])
                    previous_type = previous_type + "-" + str(types[num])
                    added_types.append(types[num])
    return df

#  function plot the correlation between variables
def corr_plot(df):

    # link ---> https://likegeeks.com/seaborn-heatmap-tutorial/
    sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
    fig=plt.gcf()
    fig.set_size_inches(10,8)
    plt.show()

#  customized histogram plot
def histogram_plot ( df, x_axis_name, y_axis_name = 'Survived', group_title =""):

    fig, axs = plt.subplots(figsize=(22, 9))
    sns.countplot(x=x_axis_name, hue= y_axis_name, data=df)

    plt.xlabel(x_axis_name, size=15, labelpad=20)
    plt.ylabel(y_axis_name + 'Count', size=15, labelpad=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    y_values = df[y_axis_name].drop_duplicates().to_list()

    plt.legend(y_values, loc='upper right', prop={'size': 15})
    if group_title == "":
        plt.title('{} Counts in {} Feature'.format(y_axis_name, x_axis_name), size=15, y=1.05)
    else:
        plt.title('{} Counts in {} group {} Feature'.format(y_axis_name, group_title, x_axis_name), size=15, y=1.05)
    plt.show()

# relative risk calculates the average risk of a dataframe column feature y and generates a new dataframe with the output risks
def relative_risk (df,x_axis_name, y_axis_name = "Survived"):

    sur_filter =df[y_axis_name]== 1

    df_surv = df[sur_filter]
    df_N_surv = df[~sur_filter]

    x_axis_cat = df[x_axis_name].drop_duplicates().reset_index(drop = True).to_list()
    data_output = []
    for cat in x_axis_cat:
        df_N_surv

        f_cat_surv =df_surv[x_axis_name] == cat
        df_cat_surv = df_surv[f_cat_surv]
        count_surv_cat = len(df_cat_surv)

        filt_cat_tt = df[x_axis_name] == cat
        df_cat_tt = df[filt_cat_tt]
        count_surv_tt = len(df_cat_tt)
        if count_surv_tt > 0:
            data_output.append([x_axis_name, cat ,count_surv_cat, 
                            count_surv_tt - count_surv_cat, 
                            count_surv_tt, 
                            count_surv_cat/count_surv_tt])
    
    df_output = pd.DataFrame(data_output, columns= ["cat_type", "cat", "risk_count", "not_risk_count", "tt_count" , "risk_rate"])

    df_output = df_output.sort_values(by=['risk_rate']).reset_index(drop=True)
    
    if df_output['risk_rate'].min() == 0:

        fzero = df_output['risk_rate'] == 0
        df_output_nz = df_output[~fzero]

    else:
        df_output_nz = df_output
    df_output["relative_risk"] = df_output['risk_rate'] / df_output_nz['risk_rate'].min() 

    return df_output

# calculate the relative risk for all columns in a given dataframe
def overall_relative_risk (df):

    columns = df.columns
    count = 0
    for column in columns:

        cat_check = df[column].drop_duplicates().reset_index(drop = True).to_list()
        if len(cat_check) <100:
            if count == 0:
                df_output = relative_risk (df,column)
            else:
                df2 = relative_risk (df,column)
                df_output = df_output.append(df2) 
            count += 1        

    return df_output

# update the risk rate of a certain feature in the main dataframe
def look_up_risk (df, df_risk, x_axis_name):
    df['risk_rate'] = 0
    cat_ls = df_risk["cat"].tolist()

    for cat in  cat_ls:
        filt_risk = df_risk["cat"] == cat
        risk_value = float(df_risk[filt_risk]["risk_rate"])

        filt = df[x_axis_name] == cat
        df.loc[df[filt].index,'risk_rate'] = risk_value

    return df

# specific code for neural network - Pytorch, transform the dataset into batches
# batch size = number of rows fed in the neural network
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

#  specific code to the titanic - preprocessing data
def preprocessing(csv_test_path,csv_train_path, plot = True, save = True):

    print("#"*20, " Preprocessing start", "#"*50)
    df_test = pd.read_csv(csv_test_path)
    df_train = pd.read_csv(csv_train_path)

    # ### Check train data to find:
    # which has NaN (Not a Number Values)</li>
    # which are non - numeric values which can be replaced by numeric values</li>  

    ## df.isnull() or df.isna() returns an array with true or false - if you apply .sum() aftwars it will count how many attributes will be NaN
    print( "\n ## Train datataset has {} null values:".format(df_train.isnull().sum() ) )

    ## df.isnull() or df.isna() returns an array with true or false - if you apply .sum() aftwars it will count how many attributes will be NaN
    print( "\n ## Test datataset has {} null values:".format( df_test.isnull().sum() ) )

    # ** Both age and Embarked have empty values and they must be filled.<br>
    # ** How will you fill them will have effect on the learning process <br>
    # ** age for instance can be filled with mean age value<br>
    # ** embarked can be filled with a letter  different from the already existing ['S' 'C' 'Q' ] e.g. 'U' as Unknown<br>
    print( "\n ## Replacing age NaN values" )
    df_train,df_test = replace_na_age (df_train,df_test)
    ### check is the values are correctly replaced
    print("\n ## * Train Age null values: ", df_train["Age"].isnull().sum())
    print("\n ## * Test Age null values: ", df_test["Age"].isnull().sum())
    
    print( "\n ## Get Name Tiltle from Name Columns" )
    df_test = get_name_title(df_test)
    df_train = get_name_title(df_train)

    print( "\n ## Get Family size from Parch and SibSp " )
    df_test = get_family_size(df_test)
    df_train = get_family_size(df_train)

    if plot:
        histogram_plot(df_train,"Age_Group")

        df_male = df_train[df_train["Sex"] == "male"]
        df_female = df_train[df_train["Sex"] == "female"]

        histogram_plot(df_male,"Age_Group", group_title = "Male")
        histogram_plot(df_female,"Age_Group", group_title = "Female")


        pclasses = df_train["Pclass"].drop_duplicates().reset_index(drop=True).to_list()
        for pclass in pclasses:
            df_male_class = df_male[ df_male["Pclass"] == pclass]
            df_female_class = df_female[ df_female["Pclass"] == pclass]
            histogram_plot(df_male_class,"Age_Group", group_title = "Male "+"pclass- "+str(pclass))
            histogram_plot(df_female_class,"Age_Group", group_title = "Female "+"pclass- "+str(pclass))

    print( "\n ## Replacing Embarked and Fare NaN " )
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
    print( "\n ## Check pandas object variables " )
    feature_pack = check_object_var(df_train)

    #### since the cabins letter shows an approximate position where the survivor might be in the ship 
    #### it is important to convert this information into umbers

    print( "\n ## Transform split Cabin Letter from Cabin and tranform it into number " )
    ### fillna will replace the value NaN into a string name "nan" so it will not be interpreted as NaN.
    ### all operations must be done in each dataframe to keep them in the same format
    df_train["Cabin"] = df_train["Cabin"].fillna("nan")
    df_test["Cabin"] = df_test["Cabin"].fillna("nan")
    _, df_train = cabin_code(feature_pack,df = df_train)
    _, df_test = cabin_code(feature_pack,df = df_test)

    print( "\n ## Concatenate Sex,Age_Group,Pclass to find a unique variable" )
    df_train = group_multi(df_train,types = ["Sex","Age_Group",'Pclass'])
    df_test = group_multi(df_test,types = ["Sex","Age_Group",'Pclass'])

    print( "\n ## Calculate the survival rate for the newly created group: Sex-Age_Group-Pclass" )

    df_survival_rate1 = relative_risk (df_train, "Sex")
    print ("\n ## * Survival rate table for Sex only", df_survival_rate1)

    df_survival_rate1 = relative_risk (df_train, "Pclass")
    print ("\n ## * Survival rate table for Pclass only", df_survival_rate1)

    df_survival_rate1 = relative_risk (df_train, "Age_Group")
    print ("\n ## * Survival rate table for Age_Group only", df_survival_rate1)

    df_survival_rate = relative_risk (df_train, "Sex-Age_Group-Pclass")
    print ("\n ## * Survival rate table for Sex-Age_Group-Pclass", df_survival_rate)

    df_train = look_up_risk (df_train, df_survival_rate,"Sex-Age_Group-Pclass" )
    df_test = look_up_risk (df_test, df_survival_rate,"Sex-Age_Group-Pclass" )


    if save:
        csv_out1 = csv_test_path.split(".csv")[0]+"_processed.csv"
        csv_out2 = csv_train_path.split(".csv")[0]+"_processed.csv"
        df_train.to_csv(csv_out1)
        df_train.to_csv(csv_out2)

    print("\n","#"*20, " Preprocessing complete", "#"*50)

    return df_train, df_test

#  specific code to the titanic - select which features and data will be fed into the AI models 
def features_to_feed(df, valid_columns= ["Family_size","Fare","Cabin_Code_Value","risk_rate"]):

    print( "\n ## Select columns to feed the Neural Network and normalize the values" )
    # the survival rate of each combination of Sex-Age_Group-Pclass were calculated and they will be fed into the network
    df_normalized = (
        ( df[valid_columns] - df[valid_columns].min() )/
        ( df[valid_columns].max() - df[valid_columns].min() ) )

    print( "\n ## Select columns to feed the Neural Network" )

    df_normalized= df_normalized.astype(float)

    array = df_normalized.values
    print(" ## * Array type: ", type(array))
    print(" ## * Rows: ", array.shape[0], " Features: ", array.shape[1])
    print( "\n ## Split values into batches if necessary" )
    x_batches = batch_split(array, batch_size = len(df), type = "input")
    if "Survived" in df.columns:
        y_unbatched = df["Survived"].values
        y_batches = batch_split(y_unbatched, batch_size = len(df_normalized), type = "output")
        return x_batches, y_batches
    else:
        return x_batches, None

#  specific code to the titanic - split dataset 
def split_n_shuffle(df_train,df_test,csv_test_surv):
    df_test_surv = pd.read_csv(csv_test_surv)
    df_test = df_test.merge(df_test_surv, left_on='PassengerId', right_on='PassengerId')

    shuffle_ids = np.arange(df_test["PassengerId"].min(),df_test["PassengerId"].max()+1,1)
    np.random.shuffle(shuffle_ids)
    train_ids = len(df_train)
    test_ids = len(df_test)
    tt_ids = train_ids + test_ids

    thresh = len(shuffle_ids)//2
    test_split = shuffle_ids[0:thresh]
    valid_split = shuffle_ids[thresh+1:len(shuffle_ids)]

    df_test = df_test.set_index("PassengerId", drop=False)
    df_test_splitted = df_test.loc[test_split]
    df_val_splitted = df_test.loc[valid_split]

    print("\n","#"*20, " Shuffle Test and split {}'%' Test and {}'%' Validation  start".format( thresh / tt_ids,
                                                                                                ( len(shuffle_ids) - thresh ) / tt_ids),
                                                                                                "#"*20
                                                                                            )
    return df_test_splitted, df_val_splitted

#  specific to Neural Networks
class NN_4layered():

    def __init__(self, features_input = 6, features_output = 2, min_neuron_per_layer = 100, epochs = 8000, lr=2e-4):
        output_feat = min_neuron_per_layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.Sequential(
                        nn.Linear(features_input,output_feat),
                        nn.ReLU(),
                        nn.Linear(output_feat,output_feat*2),
                        nn.ReLU(),
                        nn.Linear(output_feat*2,output_feat*2),
                        nn.ReLU(),
                        nn.Linear(output_feat*2,features_output),
                        nn.ReLU(),
                        ).to(self.device)

        self.criterion = torch.nn.MSELoss(reduction='mean')#.to(device)
        self.criterion2 = torch.nn.MSELoss(reduction='sum')#.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.epochs = epochs

    def train(self, x_batches, y_batches):

        self.complete_stats = []
        for epoch in range(self.epochs):
            epoch_stats = []
            for i, x in enumerate(x_batches):
                y = y_batches[i].to(self.device)
                x = x.to(self.device)
                y_pred = self.model.forward(x)
                y_pred2 = F.softmax(y_pred,dim=1)
                test = (y_pred2 - y).sum()
                loss = self.criterion(y_pred, y)
                tt_error = self.criterion2(y_pred, y)
                loss.backward()
                self.optimizer.step()
                epoch_stats.append([loss.item(),tt_error.item(),test.cpu().detach().numpy()])
                self.complete_stats.append([loss.item(),tt_error.item(),test.cpu().detach().numpy()])
            stats_arr = np.array(epoch_stats)
            if epoch % 100 == 0: 
                print("#"*100)
                print("Epoch {} - loss: {} - prediction error: {}  test: {}".format(epoch, 
                                                                                    stats_arr[:,0].mean(), 
                                                                                    stats_arr[:,1].mean(), 
                                                                                    stats_arr[:,2].mean())
                                                                                    )
                print("")

        return self.model.cpu()

    def save(self, path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/saved_models",
                   file = "custom4feat_4fc_50n_8000_epochs.pth" ):
    
        file_path = path + "/" + file
        torch.save(self.model.state_dict(), file_path)
        print("\n ## Model Saved")

    def eval_model(self,x_input,df,by_column = False, column = "Sex", model_name = "NN4L_rev0", stage = "Train"):
        x = None
        if type(x_input) == list:
            if len(x_input)==1 and torch.is_tensor(x_input[0]):
                x = x_input[0]
            else:
                for i in range(len(x_input)):
                    xi = x_input[i]
                    if i ==0:
                        x = xi    
                    else:
                        x = torch.cat([x, xi])

        y_pred = self.model.forward(x).detach().clone().cpu().numpy() 
        exp = np.exp(y_pred)
        sf_maxS = exp[:,0] / exp.sum(axis=1)
        sf_maxD = exp[:,1] / exp.sum(axis=1)
        survival_status = (np.round(sf_maxS) >= np.round(sf_maxD)) * 1
        # if the probability value of the first column is higher than the second columns it means that the passenger survived.
        # the boolean matrix is multiplied by 1 to convert true and false to 1 and 0.
        df["Survived_Prediction"] = survival_status
        mask = df["Survived"] == df["Survived_Prediction"]
        df["Survival_Output"] = sf_maxS
        prediction_ok = df[mask] 
        overall_TP = prediction_ok[prediction_ok["Survived"] == 1 ] ### Model predicted that the person would survive and got it right
        overall_TN = prediction_ok[prediction_ok["Survived"] == 0 ] ### Model predicted that the person would not survive and got it right

        
        prediction_nok = df[~mask]
        overall_FP = prediction_nok[prediction_nok["Survived_Prediction"] == 1 ] ### Model predicted incorrectly that the person would survive and got it wrong
        overall_FN = prediction_nok[prediction_nok["Survived_Prediction"] == 0 ] ### Model predicted incorrectly that the person not would survive and got it wrong

        check_tpr = len(overall_TP) + len(overall_FN)
        if check_tpr == 0:
            check_tpr = (overall_TP+0.00001) * 100000

        check_tnr = len(overall_TN) + len(overall_FP)
        if check_tnr == 0:
            check_tnr = (overall_TN+0.00001) * 100000

        check_ppv = len(overall_TP) + len(overall_FP)
        if check_ppv == 0:
            check_ppv = (overall_TP+0.00001) * 100000

        #tn, fp, fn, tp = confusion_matrix(self.Y_train, y_pred).ravel()
        tpr = len(overall_TP) / ( check_tpr ) ## sensitivity, recall, hit rate, or true positive rate (TPR)
        tnr = len(overall_TN) /( check_tnr ) ## specificity, selectivity or true negative rate (TNR)
        ppv = len(overall_TP) /( check_ppv ) ## precision or positive predictive value (PPV)
        acc = ( len(overall_TP) + len(overall_TN) ) / ( len(overall_TN) + 
                                                        len(overall_TP) + 
                                                        len(overall_FP) + 
                                                        len(overall_FN)  ) ## accuracy (ACC)

        f1_score = ( 2*len(overall_TP) ) / ( 2*len(overall_TP) + 
                                               len(overall_FP) + 
                                               len(overall_FN) )# F1 score

        results_columns = ["stage", "model_name",
                           "True Positive (TP)",
                           "True Negative (TN)",
                           "False Positive (FP)",
                           "False Negative (FN)",   
                            "recall-sensitivity","specificity", "precision-ppv","accuracy","F1_score"]
                            
        model_data = [  stage,
                        model_name + "_overall", 
                        len(overall_TP), 
                        len(overall_TN),
                        len(overall_FP), 
                        len(overall_FN), 
                        tpr, tnr, ppv, acc, f1_score]

        df_result = pd.DataFrame( [model_data] , columns= results_columns)
        print("")
        print( "Model Overall Accuracy: {} - Precision: {} - Recall: {}".format( acc, ppv, tpr))
        if by_column:

            columns_values = df[column].drop_duplicates().to_list()
            for value in columns_values:
                v_FN = overall_FN[overall_FN[column]==value]
                v_FP = overall_FP[overall_FP[column]==value]
                v_TP = overall_TP[overall_TP[column]==value]
                v_TN = overall_TN[overall_TN[column]==value]

                tpr = len(v_TP) / ( len(v_TP) + len(v_FN) ) ## sensitivity, recall, hit rate, or true positive rate (TPR)
                tnr = len(v_TN) /( len(v_TN) + len(v_FP) ) ## specificity, selectivity or true negative rate (TNR)
                ppv = len(v_TP) /( len(v_TP) + len(v_FP) ) ## precision or positive predictive value (PPV)
                acc = ( len(v_TP) + len(v_TN) ) / ( len(v_TN) + 
                                                    len(v_TP) + 
                                                    len(v_FP) + 
                                                    len(v_FN)  ) ## accuracy (ACC)

                f1_score = ( 2*len(v_TP) ) / ( 2*len(v_TP) + 
                                                len(v_FP) + 
                                                len(v_FN) )# F1 score

                model_data = [ model_name + "col_" + str(column), 
                                len(v_TP), 
                                len(v_TN),
                                len(v_FP), 
                                len(v_FN), 
                                tpr, tnr, ppv, acc, f1_score]

                print("#"*20, "Model Performance by column {} and cvalue {}:".format(column,value))
                print( "## * Accuracy: {} Precision: {} - Recall: {}".format( acc, ppv, tpr))
                df_result2 = pd.DataFrame( [model_data] , columns= results_columns)
                df_result = df_result.append(df_result2)

        return df_result

#  several sklearn models into a single class for comparison purpose
class Models_Comp():
    def __init__(self,df,valid_columns= ["Family_size","Fare","Cabin_Code_Value","risk_rate"]):

        ### Adapted from https://www.kaggle.com/ricardoluhms/titanic-81-1-leader-board-score-guaranteed/edit?rvi=1
        
        self.valid_columns = valid_columns
        df_normalized = (
        ( df[valid_columns] - df[valid_columns].min() )/
        ( df[valid_columns].max() - df[valid_columns].min() ) )

        print( "\n ## Selected columns to feed the Neural Network: ", valid_columns )

        df_normalized= df_normalized.astype(float)
        self.df = df
        array = df_normalized.values
        self.X_train = StandardScaler().fit_transform(array)
        self.Y_train = df['Survived'].values

    def sgd_classifier_model(self):
        self.sgd = linear_model.SGDClassifier(max_iter=25, tol=None)
        self.sgd.fit(self.X_train, self.Y_train)
        #Y_pred = self.sgd.predict(self.X_test)
        self.sgd.score(self.X_train, self.Y_train)
        self.acc_sgd = round(self.sgd.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## Stochastic Gradient Descent Classifier Accuracy: ", self.acc_sgd)
        
    def random_forest_model(self):    
        self.random_forest = RandomForestClassifier(n_estimators=125)
        self.random_forest.fit(self.X_train, self.Y_train)
        #Y_prediction = random_forest.predict(X_test)
        self.random_forest.score(self.X_train, self.Y_train)
        self.acc_random_forest = round(self.random_forest.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## Random Forest Classifier Accuracy: ", self.acc_random_forest)
    
    def logistic_regression_model(self):
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.Y_train)
        #Y_pred = logreg.predict(X_test)
        self.acc_log = round(self.logreg.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## Logistic Regrassion Classifier Accuracy: ", self.acc_log)
    
    def k_nearest_neighbor_model(self):
        self.knn = KNeighborsClassifier(n_neighbors = 3) 
        self.knn.fit(self.X_train, self.Y_train)  
        #Y_pred = knn.predict(X_test)  
        self.acc_knn = round(self.knn.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## K Nearest Neighbor Classifier Accuracy: ", self.acc_knn)

    def gaussian_naive_bayes_model(self):
        self.gaussian = GaussianNB() 
        self.gaussian.fit(self.X_train, self.Y_train)  
        #Y_pred = gaussian.predict(X_test)  
        self.acc_gaussian = round(self.gaussian.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## Gaussian Naive Bayes Classifier Accuracy: ", self.acc_gaussian)

    def perceptron_model(self):
        self.perceptron = Perceptron(max_iter=200)
        self.perceptron.fit(self.X_train, self.Y_train)
        #Y_pred = perceptron.predict(X_test)
        self.acc_perceptron = round(self.perceptron.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## Perceptron Classifier Accuracy: ", self.acc_perceptron)

    def lin_svc_model(self):
        self.linear_svc = LinearSVC()
        self.linear_svc.fit(self.X_train, self.Y_train)
        #Y_pred = linear_svc.predict(X_test)
        self.acc_linear_svc = round(self.linear_svc.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## Linear Support Vector Classifier Accuracy: ", self.acc_linear_svc)

    def decision_tree_model(self):
        self.decision_tree = DecisionTreeClassifier() 
        self.decision_tree.fit(self.X_train, self.Y_train)  
        #Y_pred = decision_tree.predict(X_test)  
        self.acc_decision_tree = round(self.decision_tree.score(self.X_train, self.Y_train) * 100, 2)
        print( "\n ## Decision Tree Classifier Accuracy: ", self.acc_decision_tree)
    
    def run_train_all(self):
        self.sgd_classifier_model()
        self.random_forest_model()
        self.logistic_regression_model()
        self.k_nearest_neighbor_model()
        self.gaussian_naive_bayes_model()
        self.perceptron_model()
        self.lin_svc_model()
        self.decision_tree_model()

        results = pd.DataFrame({
                'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                          'Random Forest', 'Gaussian Naive Bayes', 'Perceptron', 
                          'Stochastic Gradient Decent', 
                          'Decision Tree'],
                'Score': [self.acc_linear_svc, self.acc_knn, self.acc_log, 
                          self.acc_random_forest, self.acc_gaussian, self.acc_perceptron, 
                          self.acc_sgd, self.acc_decision_tree]})

        simple_result_df = results.sort_values(by='Score', ascending=False)
        simple_result_df = simple_result_df.set_index('Score')
        print("\n ## Dataframe Results")
        return simple_result_df

    def single_model_metrics(self, df_result = None, model_name = None, stage = "Train"):
        current_model = None
        if model_name not in self.models.keys() or model_name == None:
            print("Selected model name is not in the model list")
            print("Please check the available models: ", self.models.keys())
        else:
            current_model = self.models[model_name]

        y_pred = current_model.predict(self.X_train)

        tn, fp, fn, tp = confusion_matrix(self.Y_train, y_pred).ravel()

        tpr = tp /(tp+fn) ## sensitivity, recall, hit rate, or true positive rate (TPR)
        tnr = tn /(tn+fp) ## specificity, selectivity or true negative rate (TNR)
        ppv = tp /(tp+fp) ## precision or positive predictive value (PPV)
        acc = ( tp + tn ) / ( tn + tp + fp + fn) ## accuracy (ACC)
        f1_score = ( 2*tp ) / ( 2*tp + fp + fn)# F1 score

        model_data = [ stage , model_name, tp, tn, fp, fn, tpr, tnr, ppv, acc, f1_score]

        results_columns = ["stage", "model_name",
                           "True Positive (TP)",
                           "True Negative (TN)",
                           "False Positive (FP)",
                           "False Negative (FN)",   
                            "recall-sensitivity","specificity", "precision-ppv","accuracy","F1_score"]              
        try:
            df_result2 = pd.DataFrame( [model_data] , columns= results_columns)
            df_result = df_result.append(df_result2)

        except:    
            df_result = pd.DataFrame( [model_data] , columns= results_columns)
        
        return df_result

    def all_model_metrics(self,df_result, stage = "Train"):

        self.models = {'Support Vector Machines':self.linear_svc, 
            'KNN':self.knn, 
            'Logistic Regression': self.logreg, 
            'Random Forest': self.random_forest, 
            'Gaussian Naive Bayes': self.gaussian, 
            'Perceptron':self.perceptron, 
            'Stochastic Gradient Decent': self.sgd,
            'Decision Tree': self.decision_tree}

        for model_name in  self.models.keys():

            df_result = self.single_model_metrics(df_result = df_result, model_name = model_name, stage = stage)
        
        return df_result
    
    def update_input(self,df):
        df_normalized = (
        ( df[self.valid_columns] - df[self.valid_columns].min() )/
        ( df[self.valid_columns].max() - df[self.valid_columns].min() ) )

        df_normalized= df_normalized.astype(float)
        self.df = df
        array = df_normalized.values
        self.X_train = StandardScaler().fit_transform(array)
        self.Y_train = df['Survived'].values

    def update_n_eval_metrics(self, df ,df_result, stage):
        self.update_input(df)
        df_result = self.all_model_metrics(df_result, stage = stage )
        return df_result

    def get_best_model (self, df, df_result, complete_csv_path,submission_csv_path):
        
        df_normalized = (
        ( df[self.valid_columns] - df[self.valid_columns].min() )/
        ( df[self.valid_columns].max() - df[self.valid_columns].min() ) )

        df_normalized= df_normalized.astype(float)
        array = df_normalized.values
        self.X_test = StandardScaler().fit_transform(array)

        # get the best model for this application
        mask1 = df_result["stage"] == "Validation"
        mask2 = df_result[mask1]["accuracy"] == df_result[mask1]["accuracy"].max()
        candidates = df_result[mask1][mask2]

        if len(candidates)>1:
            mask3 = candidates["F1_score"] == candidates["F1_score"].max()
            candidates = candidates[mask3]

        best_model_ser = candidates.iloc[0]
        model_name = best_model_ser["model_name"]

        csv_all_results_path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/all_results.csv" 
        df_result.to_csv(csv_all_results_path)
        
        Y_out = self.models[model_name].predict(self.X_test)
        
        df_output = pd.DataFrame( {"PassengerId": df["PassengerId"], "Survived":Y_out} )

        csv_sub_path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/submission.csv" 
        df_output.to_csv(csv_sub_path)

# complete analysis
if __name__ == "__main__":
    csv_test_path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/test.csv" 
    csv_train_path= "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/train.csv"
    csv_test_surv = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/gender_submission.csv"

    df_train, df_test = preprocessing(csv_test_path,csv_train_path, plot = False, save = True)

    histogram_plot(df_train, x_axis_name = 'Sex-Age_Group-Pclass', y_axis_name = 'Survived', group_title ="")

    histogram_plot(df_train, x_axis_name = 'Family_size', y_axis_name = 'Survived', group_title ="")

    histogram_plot(df_train, x_axis_name = 'Cabin_Code_Value', y_axis_name = 'Survived', group_title ="")

    corr_plot(df_train)

    df_test_splitted, df_val_splitted = split_n_shuffle (df_train,df_test, csv_test_surv = csv_test_surv)

    x_batches, y_batches = features_to_feed(df_train, valid_columns= ["Family_size","Fare","Cabin_Code_Value","risk_rate"])

    x_batches_test, y_batches_test = features_to_feed(df_test_splitted, valid_columns= ["Family_size","Fare","Cabin_Code_Value","risk_rate"])

    x_batches_val, y_batches_val = features_to_feed(df_val_splitted, valid_columns= ["Family_size","Fare","Cabin_Code_Value","risk_rate"])

    model_pack = NN_4layered(features_input = 4, features_output = 2, min_neuron_per_layer = 200, epochs = 10000, lr=2e-7)

    trained_model = model_pack.train(x_batches, y_batches)

    path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/saved_models"
    afile = "model_final.pth"

    model_pack.save(path = path,
                    file = afile )

    df_result = model_pack.eval_model(x_batches, df_train, by_column = False, column = "Sex")

    df_result_test = model_pack.eval_model(x_batches_test, df_test_splitted, by_column = False, column = "Sex")

    df_result_val = model_pack.eval_model(x_batches_val, df_val_splitted, by_column = False, column = "Sex")

    df_result = df_result.append(df_result_test)
    df_result = df_result.append(df_result_val)

    other_models = Models_Comp(df_train, valid_columns= ["Family_size","Fare","Cabin_Code_Value","risk_rate"])

    simple_result_df = other_models.run_train_all()

    df_result = other_models.all_model_metrics(df_result)

    df_result = other_models.update_n_eval_metrics( df_test_splitted, df_result, stage = "Test")

    df_result = other_models.update_n_eval_metrics( df_val_splitted, df_result, stage = "Validation")
    
    csv_sub_path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/submission.csv" 
    csv_all_results_path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/all_results.csv" 

    other_models.get_best_model (df_test, df_result, csv_all_results_path,csv_sub_path)

    # - x reprent 6 features from each passenger of the ship - it is the input for the neural network
    # - y_pred = the output array
    #    explaining the variable y_pred and the functions within it from left to right:
    #     - trained_model.forward(x) get the "x" tensor (pytorch format) which has 891 passengers x 6features
    #     - return the tensor output which has 891 passengers x 2features 
    #           (first column feature is the probability of the passenger being alive, second column feature is the probability of the passenger is dead)
    #     - to remove the learning features from it we apply detach()
    #     - then we clone the output (same as copy in numpy)
    #     - to be able to use the tensor we pass it to the cpu
    #     - lastly we convert it to numpy   
    # 1.  first trial with 6 features did not performed well for women
    # 1.1. v_columns= ["Pclass","Age","SibSp","Parch","Fare","Cabin_Code_Value"] 
    # 1.2. hypothesis: since Cabin_Code_Value, Farem and Pclass are higly correlated their influence over the outcome are heavier than the other features
    # 2. second trial with 4 features without any feature engineering

    # 3. third trial feature engineering identified that sex, age_group and pclass play a major role in survivability
    # added feature engineering columns['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
    #       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Age_Group',
    #       'Title_name', 'Family_size', 'Average_Fare', 'Average_Fare_Group',
    #       'Cabin_Code', 'Cabin_Code_Value','Sex-Age_Group', 'Sex-Age_Group-Pclass', 'risk_rate']

