import pandas as pd
import numpy as np
import pymysql
import datetime as dt
import math
import random 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.regression import MeanSquaredError
from sklearn.metrics import precision_score, recall_score
from torch import nn
from torchsummary import summary
from torchviz import make_dot
import torch.nn.functional as F


def extractData(action, myhost='127.0.0.1', myuser='root', mypassword='rootroot', databasename='transaction'):
    connection = pymysql.connect(host=myhost, user=myuser, password=mypassword, database=databasename)
    cursor = connection.cursor()
    query = action
    cursor.execute(action)
    features = [column[0] for column in cursor.description]
    cursor.close()
    connection.close()
    return pd.DataFrame(cursor.fetchall(), columns=features)

def dataCleaning(data):
    transaction = data.copy()

    # data types conversion
    transaction['num_prev_purchase'] = transaction['num_prev_purchase'].astype('Int64')
    transaction['age'] = transaction['age'].astype('Int64')
    transaction['list_price'] = transaction['list_price'].astype('Float64')
    transaction['standard_cost'] = transaction['standard_cost'].astype('Float64')
    transaction['transaction_date'] = pd.to_datetime(transaction['transaction_date'])
    transaction['DOB'] = pd.to_datetime(transaction['DOB'])
    
    # abnormality
    transaction.loc[(transaction.gender == 'U').values, 'gender'] = np.nan
    transaction.loc[(transaction.job_title == 'NA').values, 'job_title'] = np.nan
    transaction.loc[(transaction.job_title == '').values, 'job_title'] = np.nan
    transaction.loc[(transaction.job_industry == 'n/a').values, 'job_industry'] = np.nan
    transaction.loc[(transaction.online_order == '').values, 'online_order'] = np.nan
    transaction.loc[(transaction.age == 123).values, 'age'] = np.nan
    transaction.loc[(transaction.age == 0).values, 'age'] = np.nan
    return transaction
    
def dataPrepForRcSys(data):
    transaction = data.copy()

    customer_info = ['customer_id', 'gender', 'num_prev_purchase', 'age', 'job_industry', 'wealth_segment', 'owns_car', 'state']
    product_info = ['product_id', 'brand', 'product_line', 'product_class', 'product_size', 'online_order']
    
    # create all possible combination of customer and product id
    all_possible_comb = pd.DataFrame(product(transaction['customer_id'].unique(), transaction['product_id'].unique()))
    all_possible_comb.rename(columns={0: "customer_id", 1: "product_id"},inplace=True)
    
    # deriving rating based on quantity, price, and transaction_date
    transaction['transaction_date'] = pd.to_datetime(transaction['transaction_date'])
    snapshot_date = transaction['transaction_date'].min() - dt.timedelta(days=1)
    user_item_table = transaction.groupby(['customer_id','product_id']).agg({
        'transaction_id': 'count',
        'list_price': 'first',
        'transaction_date': lambda x: (x.max()-snapshot_date).days
    }).reset_index()
    user_item_table.rename(columns={'transaction_id': 'quantity', 'list_price': 'price', 'transaction_date': 'day'}, inplace=True)

    # merge with all possible options 
    user_item_table = pd.merge(user_item_table,all_possible_comb, on=['customer_id','product_id'], how='outer')
    user_item_table = user_item_table.fillna(0)
    
    # merge the rating table with customer and product features
    user_item_table = pd.merge(user_item_table,transaction[product_info].groupby(['product_id']).first().reset_index(), on=['product_id'], how='left')
    user_item_table = pd.merge(user_item_table,transaction[customer_info].groupby(['customer_id']).first().reset_index(), on=['customer_id'], how='left')
    user_item_table.loc[(user_item_table.job_industry.isnull()).values, 'job_industry'] = 'Others'
    user_item_table.dropna(inplace=True)
    user_item_table.reset_index(inplace=True,drop=True)

    # Normalization
    user_item_table['quantity_norm'] = pd.Series(MinMaxScaler((0,1)).fit_transform(np.array(user_item_table['quantity']).reshape(-1, 1)).reshape(-1))
    user_item_table['day_norm'] = pd.Series(MinMaxScaler().fit_transform(np.array(user_item_table['day']).reshape(-1, 1)).reshape(-1))

    price = user_item_table['price']
    listMax = price.max()
    price_norm = [i/listMax for i in price]
    price_norm = [1/(1+math.exp(-i)) for i in price_norm]
    user_item_table['price_norm'] = price_norm

    # standardize other features and encode categorical features
    prev_purchase_scaler = MinMaxScaler(feature_range=(0,1))
    age_scaler = MinMaxScaler(feature_range=(0,1))
    user_item_table['num_prev_purchase'] = pd.Series(prev_purchase_scaler.fit_transform(np.array(user_item_table['num_prev_purchase']).reshape(-1, 1)).reshape(-1))
    user_item_table['age'] = pd.Series(age_scaler.fit_transform(np.array(user_item_table['age']).reshape(-1, 1)).reshape(-1))

    # label encoding
    gender_encoder = LabelEncoder()
    user_item_table['gender'] = gender_encoder.fit_transform(user_item_table.gender)
    owns_car_encoder = LabelEncoder()
    user_item_table['owns_car'] = gender_encoder.fit_transform(user_item_table.owns_car)
    online_order_encoder = LabelEncoder()
    user_item_table['online_order'] = gender_encoder.fit_transform(user_item_table.online_order)

    # one-hot encoding
    for i in ['job_industry', 'wealth_segment', 'state', 'brand', 'product_line', 'product_class', 'product_size']:
        user_item_table = pd.concat([user_item_table, pd.get_dummies(user_item_table[i], prefix=i)],axis=1)
        user_item_table.drop(i, axis=1, inplace=True)
      
    # generate binary target
    user_item_table['target'] = 0
    user_item_table.loc[user_item_table.quantity!=0, 'target'] = 1

    user_item_table.sort_values(by=['customer_id','product_id'],inplace=True)
    
    le_users = LabelEncoder()
    le_items = LabelEncoder()
    user_item_table['customer_id'] = le_users.fit_transform(user_item_table.customer_id.to_numpy())
    user_item_table['product_id'] = le_items.fit_transform(user_item_table.product_id.to_numpy())
    np.save('user_encoder.npy', le_users.classes_)
    np.save('item_encoder.npy', le_items.classes_)
    
    return user_item_table

class transactionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.users = df.customer_id.to_numpy()
        self.items = df.product_id.to_numpy()
        self.price = df.price_norm.to_numpy()
        self.gender = df.gender.to_numpy()
        self.owns_car = df.owns_car.to_numpy()
        self.online_order = df.online_order.to_numpy()
        self.age = df.age.to_numpy()
        self.prev_purchase = df.num_prev_purchase.to_numpy()
        self.cat_features = df[df.columns[13:-1]].to_numpy()
        self.target = df[df.columns[-1]].to_numpy()
        self.dense_features = df[['price','age','num_prev_purchase','gender','owns_car','online_order']].to_numpy(dtype='float64')

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.dense_features[idx], self.users[idx], self.items[idx], self.cat_features[idx],self.target[idx]

    def print_sample(self, idx=0):
        return {'customer_id': self.users[idx], 
                'product_id': self.items[idx], 
                'target': self.target[idx],
                'quantity': self.quantity[idx],
                'price': self.price[idx],
                'gender': self.gender[idx],
                'owns_car': self.owns_car[idx],
                'online_order': self.online_order[idx],
                'age': self.age[idx],
                'prev_purchase': self.prev_purchase[idx],
                'cat_features': self.cat_features[idx]
               }
    
    def get_Dataset(self):
        return self.df

    def get_num_user(self):
        return len(np.unique(self.users))
    
    def get_num_item(self):
        return len(np.unique(self.items))

def train_tet_val_split(data):
    train, test = train_test_split(data, test_size=0.20, random_state=107)
    test, val = train_test_split(test, test_size=0.33, random_state=107)
    return transactionDataset(train), transactionDataset(test), transactionDataset(val)

def trainModel(epochs, model, loss_func, trainloader, testloader, optimizer, patience, model_path,
               save_model = True, save_model_name = 'best_model',
               scheduler=None, clip_threshold=1.0, recallatKinputs=None):
    train_loss_lst = []
    train_loss = 0
    test_loss_lst = []
    test_loss = 0
    accuracy_lst = []
    auc_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    e = 0
    best_f1 = 0
    best_test_loss = 10000000
    best_epoch = 0
    earlystop = False
    print('---------Start Training---------')
    while e < epochs | earlystop:
        # train
        model.train()
        for i, train_batch in enumerate(trainloader):
            optimizer.zero_grad()
            predict = model(*train_batch[:-1]).to(torch.float64)
            if predict.isnan().sum().item() > 0:
                print('NAN')
            actual = torch.unsqueeze(train_batch[-1],1).to(torch.float64)
            loss = loss_func(predict, actual)
            train_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
            optimizer.step()
        train_loss /= len(trainloader)
        train_loss_lst.append(train_loss.item())
        
        # evaluate
        with torch.no_grad():
            model.eval()
            prediction = []
            target = []
            for test_batch in testloader:
                predict = model(*test_batch[:-1]).to(torch.float64)
                if predict.isnan().sum().item() > 0:
                    print('NAN FOUND')
                actual = torch.unsqueeze(test_batch[-1],1)
                prediction.extend(nn.Sigmoid()(predict).squeeze(1).tolist())
                target.extend(test_batch[-1].tolist())
                loss = loss_func(predict, actual.to(torch.float64))
                test_loss += loss
            accuracy = sum(1 for x,y in zip([round(i) for i in prediction],target) if x == y) / len(prediction)
            accuracy_lst.append(accuracy)
            auc = roc_auc_score(target,prediction)
            auc_lst.append(auc)
            precision = precision_score(target,[round(i) for i in prediction])
            precision_lst.append(precision)
            recall = recall_score(target,[round(i) for i in prediction])
            recall_lst.append(recall)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_lst.append(f1)
            test_loss /= len(testloader)
            test_loss_lst.append(test_loss.item())
        
        print('Epoch:', e, 'Train loss:', train_loss.item(), 'Test loss:', test_loss.item(), '\n', 'Test accuracy:', accuracy, 'Test AUC:', auc, '\n','Test Precision:', precision, 'Test Recall:', recall, '\n', 'F1 Score:', f1)
        print('Positive prediction:', sum(1 for x,y in zip([round(i) for i in prediction],target) if x == y and y == 1))
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = e
            if save_model:
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'epoch': e
                }, model_path+'/'+save_model_name+'.pth')        
        elif e - best_epoch > patience:
            print("Early stopped training at epoch %d" %e + ', with the best epoch at %d' %best_epoch)
            print('The best F1 Score is ' + str(best_f1))
            break  
        print('--------------------------------------')
        if recallatKinputs != None:
            print('Recall@10: ', recallAtK(model,recallatKinputs['current'],recallatKinputs['future'],recallatKinputs['k']))
            print('Precision@10: ', precisionAtK(model,recallatKinputs['current'],recallatKinputs['future'],recallatKinputs['k']))
            print('--------------------------------------')


        if scheduler:
            scheduler.step()
        train_loss = 0
        test_loss = 0
        e += 1
        
    return train_loss_lst, test_loss_lst, accuracy_lst, auc_lst, precision_lst, recall_lst

def evalModel(model, evaldata):
    with torch.no_grad():
        model.eval()
        prediction = []
        target = []
        for test_batch in evaldata:
            predict = model(*test_batch[:-1]).to(torch.float64)
            if predict.isnan().sum().item() > 0:
                print('NAN FOUND')
            actual = torch.unsqueeze(test_batch[-1],1)
            prediction.extend(nn.Sigmoid()(predict).squeeze(1).tolist())
            target.extend(test_batch[-1].tolist())
        accuracy = sum(1 for x,y in zip([round(i) for i in prediction],target) if x == y) / len(prediction)
        auc = roc_auc_score(target,prediction)
        precision = precision_score(target,[round(i) for i in prediction])
        recall = recall_score(target,[round(i) for i in prediction])
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            'Accuracy': accuracy,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Actual Positive': sum(target),
            'Predicted Positive': sum([round(i) for i in prediction])
        }
    
def getNextNRecommendation(model, user_id, n, full_data):
    input_data = transactionDataset(full_data[full_data.customer_id==user_id])
    input_data = DataLoader(dataset=input_data, batch_size=input_data.get_num_item())
    model.eval()
    prediction = []
    for batch in input_data:
        predict = model(*batch[:-1]).to(torch.float64)
        if predict.isnan().sum().item() > 0:
            print('NAN FOUND')
        prediction.extend(nn.Sigmoid()(predict).squeeze(1).tolist())
    purchased_item = full_data[np.logical_and(full_data.customer_id==user_id, full_data.target==1)].product_id.tolist()
    prediction = np.argsort(prediction)
    item_to_recommend = [i for i in prediction if i not in purchased_item]
    return item_to_recommend[::-1][:n]




    