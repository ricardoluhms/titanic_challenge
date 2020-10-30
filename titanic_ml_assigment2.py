
# %%
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

# Model Helpers
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


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
    
    df_train["Age_Group"] = pd.qcut(df_train['Age'], 8)
    df_test["Age_Group"] = pd.qcut(df_test['Age'], 8)

    return df_train,df_test

def get_name_title(df):
    
    names = df["Name"].to_list()
    df["Title_name"] = ""
    title_names = []
    for name in names:
        code = name.split(",")[1].split(".")[0]
        title_names.append(code)
    df["Title_name"] = title_names
    
    return df
# ### Load Test and Train Data

def get_family_size(df):
    df["Family_size"] = df["SibSp"] + df["Parch"] + 1
    df["Average_Fare"] = df["Fare"]/df["Family_size"]
    df["Average_Fare_Group"] = pd.qcut(df['Average_Fare'], 5)

    return df

def group_types(df,typeA, typeB):
    typeC = str(typeA) + "-" + str(typeB) 
    #from IPython import embed; embed()
    if df[typeA].dtype == pd.Categorical:
        df[typeA] = df[typeA].astype('string')
    if df[typeB].dtype == pd.Categorical:
        df[typeB] = df[typeB].astype('string')

    df[typeC] = df[typeA].map(str)+ "-" + df[typeB].map(str)
    return df

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

def corr_plot(df):

    # link ---> https://likegeeks.com/seaborn-heatmap-tutorial/
    sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
    fig=plt.gcf()
    fig.set_size_inches(10,8)
    plt.show()

def histogram_plot (df, x_axis_name, group_title =""):


    fig, axs = plt.subplots(figsize=(22, 9))
    sns.countplot(x=x_axis_name, hue='Survived', data=df)

    plt.xlabel(x_axis_name, size=15, labelpad=20)
    plt.ylabel('Passenger Count', size=15, labelpad=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
    if group_title == "":
        plt.title('Survival Counts in {} Feature'.format(x_axis_name), size=15, y=1.05)
    else:
        plt.title('Survival Counts in {} group {} Feature'.format(group_title, x_axis_name), size=15, y=1.05)

    plt.show()

def relative_risk (df,x_axis_name):

    sur_filter =df["Survived"]== 1

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
    
    df_output = pd.DataFrame(data_output, columns= ["cat_type", "cat", "survival_count", "death_count", "tt_count" , "survival_rate"])

    df_output = df_output.sort_values(by=['survival_rate']).reset_index(drop=True)
    
    if df_output['survival_rate'].min() == 0:

        fzero = df_output['survival_rate'] == 0
        df_output_nz = df_output[~fzero]

    else:
        df_output_nz = df_output
    df_output["relative_risk"] = df_output['survival_rate'] / df_output_nz['survival_rate'].min() 

    return df_output

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

def look_up_risk (df, df_risk, x_axis_name):
    df['survival_rate'] = 0
    cat_ls = df_risk["cat"].tolist()

    for cat in  cat_ls:
        # from IPython import embed; embed()
        filt_risk = df_risk["cat"] == cat
        risk_value = float(df_risk[filt_risk]["survival_rate"])

        filt = df[x_axis_name] == cat
        df.loc[df[filt].index,'survival_rate'] = risk_value

    return df

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

def features_to_feed(df, valid_columns= ["Family_size","Fare","Cabin_Code_Value","survival_rate"]):

    print( "\n ## Select columns to feed the Neural Network and normalize the values" )
    # the survival rate of each combination of Sex-Age_Group-Pclass were calculated and they will be fed into the network
    #from IPython import embed; embed()
    df_normalized = (
        ( df[valid_columns] - df[valid_columns].min() )/
        ( df[valid_columns].max() - df[valid_columns].min() ) )

    print( "\n ## Select columns to feed the Neural Network" )

    df_normalized= df_normalized.astype(float)

    array = df_normalized.values
    print(" ## * Array type: ", type(array))
    print(" ## * Rows: ", array.shape[0], " Features: ", array.shape[1])
    print( "\n ## Split values into batches if necessary" )
    x_batches = batch_split(array, batch_size = 892, type = "input")
    if "Survived" in df.columns:
        y_unbatched = df["Survived"].values
        y_batches = batch_split(y_unbatched, batch_size = 892, type = "output")
        return x_batches, y_batches
    else:
        return x_batches, None

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

    def eval_train(self,x_input,df,by_column = False, column = "Sex"):
        eval_info = {}
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
        #from IPython import embed; embed()
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

        prediction_nok = df_train[~mask]
        overall_FP = prediction_nok[prediction_nok["Survived_Prediction"] == 1 ] ### Model predicted incorrectly that the person would survive and got it wrong
        overall_FN = prediction_nok[prediction_nok["Survived_Prediction"] == 0 ] ### Model predicted incorrectly that the person not would survive and got it wrong


        precision = len(overall_TP)/(len(overall_TP)+len(overall_FP))
        recall = len(overall_TP)/(len(overall_TP)+len(overall_FN))
        accuracy = len(prediction_ok) / len(df_train)
        d = {"precision":precision, "recall":recall, "accuracy":accuracy }
        eval_info["overall"] = d

        print("")
        print( "Model Overall Accuracy: {} - Precision: {} - Recall: {}".format( accuracy, precision, recall))
        if by_column:

            columns_values = df[column].drop_duplicates().to_list()
            for value in columns_values:
                v_FN = overall_FN[overall_FN[column]==value]
                v_FP = overall_FP[overall_FP[column]==value]
                v_TP = overall_TP[overall_TP[column]==value]
                V_TN = overall_TN[overall_TN[column]==value]

                #from IPython import embed; embed()
                precision = len(v_TP)/(len(v_TP)+len(v_FP))
                recall = len(v_TP)/(len(v_TP)+len(v_FN))
                accuracy = len(V_TN + v_TP) / len(v_FN + v_FP + v_TP + V_TN)
                d = {"precision":precision, "recall":recall, "accuracy":accuracy }
                eval_info[value] = d

                print("#"*20, "Model Performance by column {} and cvalue {}:".format(column,value))
                print( "## * Accuracy: {} Precision: {} - Recall: {}".format( accuracy, precision, recall))

        return eval_info

class Models_Comp():
    def __init__(self,df,valid_columns= ["Family_size","Fare","Cabin_Code_Value","survival_rate"]):
        pass
        df_normalized = (
        ( df[valid_columns] - df[valid_columns].min() )/
        ( df[valid_columns].max() - df[valid_columns].min() ) )

        print( "\n ## Select columns to feed the Neural Network" )

        df_normalized= df_normalized.astype(float)

        array = df_normalized.values
        self.X_train = StandardScaler().fit_transform(array)
        self.Y_train = df['Survived'].values

    def sgd_classifier(self):
        sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
        sgd.fit(self.X_train, self.Y_train)
        #Y_pred = sgd.predict(self.X_test)
        sgd.score(self.X_train, self.Y_train)
        acc_sgd = round(sgd.score(self.X_train, self.Y_train) * 100, 2)
        print(acc_sgd)


csv_test_path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/test.csv" 
csv_train_path= "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/train.csv"

df_train, df_test = preprocessing(csv_test_path,csv_train_path, plot = False, save = True)

x_batches, y_batches = features_to_feed(df_train, valid_columns= ["Family_size","Fare","Cabin_Code_Value","survival_rate"])

model_pack = NN_4layered(features_input = 4, features_output = 2, min_neuron_per_layer = 200, epochs = 12000, lr=2e-7)

trained_model = model_pack.train(x_batches, y_batches)

path = "C:/Users/ricar/Desktop/Schulich/MMAI AI intro/titanic/saved_models"
file = "model_test.pth"

model_pack.save(path = path,
                file = file )

model_performance = model_pack.eval_train (x_batches,df_train,by_column = True, column = "Sex")

other_models = Models_Comp(df_train, valid_columns= ["Family_size","Fare","Cabin_Code_Value","survival_rate"])

other_models.sgd_classifier()

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
# 1.  first trial with 6 features did not performed well for women
# 1.1. v_columns= ["Pclass","Age","SibSp","Parch","Fare","Cabin_Code_Value"] 
# 1.2. hypothesis: since Cabin_Code_Value, Farem and Pclass are higly correlated their influence over the outcome are heavier than the other features
# 2. second trial with 4 features without any feature engineering

# 3. third trial feature engineering identified that sex, age_group and pclass play a major role in survivability
# added feature engineering columns['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Age_Group',
#       'Title_name', 'Family_size', 'Average_Fare', 'Average_Fare_Group',
#       'Cabin_Code', 'Cabin_Code_Value','Sex-Age_Group', 'Sex-Age_Group-Pclass', 'survival_rate']
#### Check the training performance

#a = torch.load(path+"/"+file)

#model.load_state_dict(torch.load(file_path))
#model.eval()



from IPython import embed ; embed()


