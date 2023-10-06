import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import main
from sklearn.preprocessing import LabelEncoder

img_path = '../images/'
dta_path = '../final_data/'
# Functions

def importpredresult():
    recommendation_result = pd.read_csv(dta_path+'prediction_output.csv', converters={'Predicted_Item_ID': pd.eval})
    user_encoder = LabelEncoder()
    user_encoder.classes_ = np.load('user_encoder.npy')
    item_encoder = LabelEncoder()
    item_encoder.classes_ = np.load('item_encoder.npy')
    recommendation_result.User_ID = user_encoder.inverse_transform(recommendation_result.User_ID)
    for i in recommendation_result.Predicted_Item_ID:
        i = item_encoder.inverse_transform(i)
    return recommendation_result

def check_credentials(username, password):
    # Replace this with your authentication logic
    usernamelist = ["admin",'Winnie','Matt']
    passwordlist = ['abc','abc','abc']
    index = -1
    try:
        index = usernamelist.index(username)
        return password == passwordlist[index]
    except:
        return False

# Inputs 

recommendation_result = importpredresult()

trans = pd.read_csv(dta_path+'transaction.csv')
trans['post_code'] = trans['post_code'].astype('string')

user_item_tab = pd.read_csv(dta_path+'user_item_table.csv')


##################################################################################################################
################################################## Format the page ###############################################
##################################################################################################################

# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = "login"
if 'user' not in st.session_state:
    st.session_state.user = ''

# Main app content
if st.session_state.app_state == "login":
    st.image(img_path+'dfm.png')
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_credentials(username, password):
            st.success("Login Successful!")
            st.session_state.app_state = "main_app"
            st.session_state.user = username
            st.experimental_rerun() 
        else:
            st.error("Invalid Credentials")

elif st.session_state.app_state == "main_app":
    
    page = st.sidebar.selectbox(
        "Customer Analysis Services",
        ("Check User Profile", "Big Picture Visualization"),
        index=0
    )
    if page == "Big Picture Visualization" and st.session_state.app_state != "gen_viz_app":
        st.session_state.app_state = "gen_viz_app"
        st.experimental_rerun() 

    st.subheader("Welcome back " + st.session_state.user)
    st.title("Check User Profile through User ID")
    if st.button('Get the Most Up-to-Date Result (WARNING: THIS TAKES A LONG TIME TO RUN!)'):
        main.main()
        st.success("The model is retrained on the most up-to-date data!")
        recommendation_result = importpredresult()
    
    user_id = st.number_input("Enter a User ID", step=1)
    if user_id in set(recommendation_result.User_ID):
        pred = recommendation_result[recommendation_result.User_ID == user_id]['Predicted_Item_ID'].values[0]
        
        with st.expander("User's Personal Information",expanded=True):
            user_info1 = trans[trans.customer_id==user_id][['customer_id','first_name',
           'last_name', 'gender', 'age','street', 'state',
           'country', 'post_code']].drop_duplicates().rename(columns={"customer_id": "user_id"}).set_index('user_id')
            user_info2 = trans[trans.customer_id==user_id][['job_title',
           'job_industry', 'wealth_segment', 'owns_car','num_prev_purchase']].drop_duplicates()

            st.dataframe(user_info1)
            st.dataframe(user_info2,hide_index=True)

        with st.expander("Recommended Products Information",expanded=True):
            products_info = trans[trans.product_id.isin(recommendation_result[recommendation_result.User_ID == user_id]['Predicted_Item_ID'].values[0])][['product_id', 'brand', 'product_line', 'product_class',
           'product_size', 'list_price', 'standard_cost']].drop_duplicates().sort_values(by=['product_id']).set_index('product_id')
            st.dataframe(products_info)

        with st.expander("Previous Purchase Information",expanded=True):
            prev_products_info = trans[trans.customer_id==user_id][['product_id', 'brand', 'product_line', 'product_class',
           'product_size', 'list_price', 'standard_cost', 'transaction_date',
           'day_of_week', 'online_order', 'order_status']].set_index('product_id').sort_values(by=['transaction_date'])

            st.dataframe(prev_products_info)
    
            prev_products_info['transaction_date'] = pd.to_datetime(prev_products_info.transaction_date)
            total_trans = prev_products_info.groupby([
                (prev_products_info.transaction_date.dt.year),
                (prev_products_info.transaction_date.dt.month)]).agg(monthly_transaction=('list_price','sum'))
            total_trans.index.names = ['year','month']
            total_trans.reset_index(inplace=True)
            if 1 not in total_trans.month.values:
                total_trans=pd.concat([total_trans,pd.DataFrame({'year':[2017],'month':[1],'monthly_transaction':[None]})],ignore_index=True)
            if 12 not in total_trans.month.values:
                total_trans=pd.concat([total_trans,pd.DataFrame({'year':[2017],'month':[12],'monthly_transaction':[None]})],ignore_index=True)
            total_trans['ym'] = pd.to_datetime(total_trans.year.astype('string')+'-'+total_trans.month.astype('string')+'-01')
            st.line_chart(total_trans, x="ym", y="monthly_transaction")

    if st.button("Logout"):
        st.success("Login Successful!")
        st.session_state.app_state = "login"
        st.session_state.user = ''
        st.experimental_rerun() 

elif st.session_state.app_state == "gen_viz_app":
    
    page = st.sidebar.selectbox(
        "Customer Analysis Services?",
        ("Check User Profile", "Genearl Visualization"),
        index=1
    )
    if page == 'Check User Profile' and st.session_state.app_state != "main_app":
        st.session_state.app_state = "main_app"
        st.experimental_rerun() 
        
    st.title("Customers Profile")
    st.image(img_path+'Customer_profile.png')
    st.title("Products/Orders Profile")
    st.image(img_path+'Product_Order_profile.png')
    
    if st.button("Logout"):
        st.success("Login Successful!")
        st.session_state.app_state = "login"
        st.session_state.user = ''
        st.experimental_rerun() 


        