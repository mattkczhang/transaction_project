#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
sys.path.append('/Users/chonge/Documents/Document/University_of_Chicago/transaction_project')
from util import *
from models import *

def main():
    # some hyperparamters 
    num_features = 38
    batch_size = 16
    num_epoch = 1000
    model_path = '/Users/chonge/Documents/Document/University_of_Chicago/transaction_project/models/binary_classification'

    raw_data = extractData(action = 'SELECT * FROM transaction LEFT JOIN dim_customer USING(customer_id)     LEFT JOIN dim_product USING(product_id)     LEFT JOIN dim_time USING(transaction_id)     LEFT JOIN dim_order USING(transaction_id)')
    final_data = dataCleaning(raw_data)
    user_item_table = dataPrepForRcSys(final_data)

    train_bi, test_bi, val_bi = train_tet_val_split(user_item_table)

    train_loader_bi = DataLoader(dataset=train_bi, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader_bi = DataLoader(dataset=test_bi, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_bi = DataLoader(dataset=val_bi, batch_size=batch_size, shuffle=True, drop_last=True)

    torch.set_default_dtype(torch. float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DeepFM train
    num_user = len(np.unique(user_item_table.customer_id))
    num_item = len(np.unique(user_item_table.product_id))
    pos_weight = (len(train_loader_bi.dataset.target)-train_loader_bi.dataset.target.sum())/train_loader_bi.dataset.target.sum()
    RecSys_model = DeepFM(num_users=num_user, num_items=num_item, emb_size=64, deep_param=[1024,512,256], num_features=num_features)
    optimizer = torch.optim.Adam(RecSys_model.parameters(), lr=0.00001, weight_decay=0.00001)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight*1.5))
    deepfm_train_loss_list, deepfm_test_loss_list, deepfm_accuracy_list, deepfm_auc_list, deepfm_precision_list, deepfm_recall_list = trainModel(
        num_epoch, RecSys_model, loss_function, train_loader_bi, test_loader_bi, 
        optimizer, patience=3, model_path=model_path, clip_threshold=0.2, save_model=True, save_model_name = 'final_model')

    best_rcsys = RecSys(num_users=num_user, num_items=num_item, emb_size=64, deep_param=[1024,512,256], num_features=num_features)
    optimizer = torch.optim.Adam(best_rcsys.parameters(), lr=0.00001, weight_decay=0.00001)

    best_model_checkpoint = torch.load(model_path+'/best_model.pth')
    best_rcsys.load_state_dict(best_model_checkpoint['model'])
    optimizer.load_state_dict(best_model_checkpoint['optimizer'])
    best_epoch = best_model_checkpoint['epoch']

    print(evalModel(best_rcsys, test_loader_bi))
    print(evalModel(best_rcsys, val_loader_bi))

    prediction = []
    for i in range(len(np.unique(user_item_table.customer_id))):
        prediction.append([i,getNextNRecommendation(best_rcsys,i,10,user_item_table)])
    prediction_output = pd.DataFrame(prediction, columns=['User_ID', 'Predicted_Item_ID'])

    prediction_output.to_csv('final_data/prediction_output.csv',index=False)

if __name__ == '__main__':
    main()

