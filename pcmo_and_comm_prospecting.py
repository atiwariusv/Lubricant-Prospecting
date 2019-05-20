# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:49:40 2019

@author: A012050
"""

import pandas as pd
import pyodbc
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.cluster import KMeans
import sklearn.metrics
#Finding optimal no. of clusters
from scipy.spatial.distance import cdist

lubes_pcmo_cust = pd.read_csv('consolidated_customers.csv')

#link it to  unique CustomerKey
#Read in the SQL Query for DIM Customers
with open('all_cust_ids_sql.txt', 'r') as myfile:
    dim_cust_query = myfile.read()

#Connect to the Azure Database
cnxn = pyodbc.connect('Driver={ODBC Driver 13 for SQL Server}; \
                      Server=tcp:dbdaz001.database.windows.net,1433; \
                      Database=IADevDB1;Uid=IAUSER1;Pwd=IADBUser1$; \
                      Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;')

#Read in the Customer list from Azure Dim Customers. 
dim_cust_list = pd.read_sql(dim_cust_query, cnxn)

lubes_pcmo_cust.sourcesystemid = lubes_pcmo_cust.sourcesystemid.astype(str).apply(lambda x: x.zfill(6))

lubes_cust_w_customerkeys = pd.merge(lubes_pcmo_cust, dim_cust_list, left_on = ['source','sourcesystemid'], 
                                     right_on = ['Source','SourceSystemID'], how = 'left')


lubes_cust_w_customerkeys.dropna(subset = ['CustomerKey'], axis = 0, inplace = True)


#read in dedupe list from colin to assign the enterprise keys
colins_dedupe_list = pd.read_csv('dedupe_list_from_colin.csv')

#merge in the lubes_cust_w_customerkeys with colins list to link the enterprise keys. 
#infogroup records are arranged by enterprise keys.
lubes_customerenterprisekeys = pd.merge(lubes_cust_w_customerkeys, colins_dedupe_list, left_on = 'CustomerKey', 
                                        right_on = 'CustomerKey (DimCustomer)', how = 'left')

lubes_customerenterprisekeys['CustomerEnterpriseKey'] = 'A' + lubes_customerenterprisekeys['CustomerEnterpriseKey'].astype(str)
lubes_customerenterprisekeys[['CustomerEnterpriseKey', 'drop']] = lubes_customerenterprisekeys['CustomerEnterpriseKey'].str.split('.',expand=True)
#now we read in infogroup file
infogroup_file = pd.read_csv('2019_03_20_IG_File_Marketing_Fields_Matchup_to_USV_Customer_Master_File - Copy.csv', encoding = "ISO-8859-1")

#now we merge in the 
lubes_ig = pd.merge(lubes_customerenterprisekeys, infogroup_file, left_on = 'CustomerEnterpriseKey', 
                    right_on = 'CustomerEnterpriseKey', how = 'left')


lubes_ig['BE_Primary_SIC_Code'] = lubes_ig['BE_Primary_SIC_Code'].astype(str)
lubes_ig['BE_Primary_SIC_Code'] = lubes_ig['BE_Primary_SIC_Code'].str[:4]
lubes_ig['BE_NAICS_Code8'] = lubes_ig['BE_NAICS_Code8'].astype(str)
lubes_ig['BE_NAICS_Code8'] = lubes_ig['BE_NAICS_Code8'].str[:6]
lubes_ig = lubes_ig.drop_duplicates(subset='CustomerEnterpriseKey', keep='first')

prospecting_table = lubes_ig

#prospecting_table.dropna(subset = ['CustomerGroup'], inplace = True)

#data cleanup
prospecting_table = prospecting_table.drop(prospecting_table.columns[2:17],1)

prospecting_table.drop(['MailingAddress2', 'MailingCity', 'MailingState', 
                        'ZipCode', 'Zip4', 'MailingCountry', 'BE_Match_Level', 
                        'BE_Match_Score', 'BE_Record_Type_Code', 'BE_Contact_Manager','BE_Selected_SIC_Description', 
                        'BE_Franchise_Specialty_Description1', 'BE_Franchise_Specialty_Description2', 
                        'BE_Franchise_Specialty_Description3', 'BE_Franchise_Specialty_Description4', 
                        'BE_Franchise_Specialty_Description5', 'BE_Franchise_Specialty_Code6',
                        'BE_Franchise_Specialty_Description6', 'BE_Primary_SIC_Description', 
                        'BE_Secondary_SIC_Description_1', 'BE_Secondary_SIC_Description_2', 
                        'BE_Secondary_SIC_Description_3', 'BE_Secondary_SIC_Description_4', 
                        'BE_NAICS_Description', 'BE_Location_Employment_Size_Description', 
                        'BE_Corporate_Employment_Size_Description', 'BE_Selected_SIC_Code'], axis = 1, inplace = True)


#drop additional columns that were dropped in the previous iteration
prospecting_table.drop(['BE_Production_Date','BE_Obsolescence_Date', 'BE_Production_Date_Formatted', 
                        'BE_Source', 'BE_Book_Number'], axis=1, inplace= True)
    
#Dropping the lubes variables because the customers that we are interested in as prospects wont have any Lubes sales
prospecting_table = prospecting_table.loc[:, ~prospecting_table.columns.str.startswith('BE_Contact')]
prospecting_table.dropna(thresh=prospecting_table.shape[0]*0.5, how='all', axis=1, inplace = True)

#recode the fields 'BE_Year_SIC_Added_to_Record', 'BE Year First Appeared in Yellow Pages', 'BE Year Established', '
prospecting_table['BE_yellow_pages_number_of_years'] = 2019 - prospecting_table['BE_Year_First_Appeared_in_Yellow_Pages']

#subsequently drop the above fields
prospecting_table.drop(['BE_Year_First_Appeared_in_Yellow_Pages'],axis=1,inplace=True)


prospecting_table.drop(['CustomerPhone_y', 'BE_Site_Number', 'BE_Tele_Research_Date',
                        'BE_Call_Status_Code','BE_Call_Status_Description',
                        'BE_Business_Credit_Score_Description', 'BE_Delivery_Point_Bar_Code'], axis = 1, inplace = True)

#drop the geographic information since this  information will not matter if prospecting in a different territory
prospecting_table.drop(['BE_Primary_Address', 'BE_Primary_City_Name', 'BE_Primary_State_Abbreviation', 
                        'BE_Primary_ZIP_Code', 'BE_Primary_ZIP4_Code', 'BE_Primary_ZIP_Code_Plus_4', 
                        'BE_Primary_Carrier_Route_Code', 'BE_Primary_State_Code', 
                        'BE_County_Code', 'BE_County_Description', 'BE_CBSA_Code', 'BE_CBSA_Description', 
                        'BE_CSA_Code'], axis = 1, inplace = True)
    
    
#add in the abi number and the related fields   
sizes = prospecting_table['BE_ABI_Number'].value_counts()
low_abi = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_ABI_Number'].isin(low_abi), 'BE_ABI_Number'] = "other_abi_busiinesses"
abi_dummy = pd.get_dummies(prospecting_table.BE_ABI_Number, drop_first=True)
abi_dummy.columns = ['abi_' + str(col) for col in abi_dummy.columns]
prospecting_table = pd.concat([prospecting_table,abi_dummy], axis = 1)

#URL
sizes = prospecting_table['BE_Web_Address_URL'].value_counts()
low_url_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Web_Address_URL'].isin(low_url_codes), 
                      'BE_Web_Address_URL'] = "other_url_sites"
url_dummy = pd.get_dummies(prospecting_table['BE_Web_Address_URL'], drop_first=True)
url_dummy.columns = ['web_address_' + str(col) for col in url_dummy.columns]
prospecting_table = pd.concat([prospecting_table,url_dummy], axis = 1)

#metro micro indicator
metro_ind_dummy = pd.get_dummies(prospecting_table['BE_Metro_Micro_Indicator'], drop_first=True)
metro_ind_dummy.columns = ['metro_ind_' + str(col) for col in metro_ind_dummy.columns]
prospecting_table = pd.concat([prospecting_table,metro_ind_dummy], axis = 1)

#sic_code
sizes = prospecting_table['BE_Primary_SIC_Code'].value_counts()
low_sic_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Primary_SIC_Code'].isin(low_sic_codes), 
                      'BE_Primary_SIC_Code'] = "other_sic_codes"
sic_codes_dummy = pd.get_dummies(prospecting_table['BE_Primary_SIC_Code'], drop_first=True)
sic_codes_dummy.columns = ['sic_codes_' + str(col) for col in sic_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,sic_codes_dummy], axis = 1)

#naics_code
sizes = prospecting_table['BE_NAICS_Code8'].value_counts()
low_naics_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_NAICS_Code8'].isin(low_naics_codes), 
                      'BE_NAICS_Code8'] = "other_naics_codes"
naics_codes_dummy = pd.get_dummies(prospecting_table['BE_NAICS_Code8'], drop_first=True)
naics_codes_dummy.columns = ['naics_codes_' + str(col) for col in naics_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,naics_codes_dummy], axis = 1)

#employment_size_code
sizes = prospecting_table['BE_Location_Employment_Size_Code'].value_counts()
low_employment_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Location_Employment_Size_Code'].isin(low_employment_codes), 
                      'BE_Location_Employment_Size_Code'] = "other_employment_size_codes"
employment_codes_dummy = pd.get_dummies(prospecting_table['BE_Location_Employment_Size_Code'], drop_first=True)
employment_codes_dummy.columns = ['employment_size_codes_' + str(col) for col in employment_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,employment_codes_dummy], axis = 1)

#sales volume description
sizes = prospecting_table['BE_Location_Sales_Volume_Description'].value_counts()
#low_sales_vol = sizes.index[sizes < prospecting_table.shape[0]*0.02]
#prospecting_table.loc[prospecting_table['BE_Location_Sales_Volume_Description'].isin(low_sales_vol), 
#                      'BE_Location_Sales_Volume_Description'] = "other_employment_size_codes"
sales_vol_dummy = pd.get_dummies(prospecting_table['BE_Location_Sales_Volume_Description'], drop_first=True)
sales_vol_dummy.columns = ['sales_vol_desc_' + str(col) for col in sales_vol_dummy.columns]
prospecting_table = pd.concat([prospecting_table,sales_vol_dummy], axis = 1)

#business status
sizes = prospecting_table['BE_Business_Status_Code'].value_counts()
low_bus_status = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Business_Status_Code'].isin(low_bus_status), 
                      'BE_Business_Status_Code'] = "other_business_status_codes"
bus_status_dummy = pd.get_dummies(prospecting_table['BE_Business_Status_Code'], drop_first=True)
bus_status_dummy.columns = ['business_status_' + str(col) for col in bus_status_dummy.columns]
prospecting_table = pd.concat([prospecting_table,bus_status_dummy], axis = 1)

#public private code
sizes = prospecting_table['BE_Public_Private_Code'].value_counts()
low_public_codes = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Public_Private_Code'].isin(low_public_codes), 
                      'BE_Public_Private_Code'] = "other_private_public_codes"
public_private_dummy = pd.get_dummies(prospecting_table['BE_Public_Private_Code'], drop_first=True)
public_private_dummy.columns = ['public_private_' + str(col) for col in public_private_dummy.columns]
prospecting_table = pd.concat([prospecting_table,public_private_dummy], axis = 1)

#BE_Asset_Size_Indicator
sizes = prospecting_table['BE_Asset_Size_Indicator'].value_counts()
low_asset_size_codes = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Asset_Size_Indicator'].isin(low_asset_size_codes), 
                      'BE_Asset_Size_Indicator'] = "other_low_asset_size_codes"
low_asset_size_dummy = pd.get_dummies(prospecting_table['BE_Asset_Size_Indicator'], drop_first=True)
low_asset_size_dummy.columns = ['public_private_' + str(col) for col in low_asset_size_dummy.columns]
prospecting_table = pd.concat([prospecting_table,low_asset_size_dummy], axis = 1)


#BE_Public_Filing_Indicator
public_filing_dummy = pd.get_dummies(prospecting_table['BE_Public_Filing_Indicator'], drop_first=True)
public_filing_dummy.columns = ['public_filing_' + str(col) for col in public_filing_dummy.columns]
prospecting_table = pd.concat([prospecting_table,public_filing_dummy], axis = 1)

#BE_Yellow_Page_Code
sizes = prospecting_table['BE_Yellow_Page_Code'].value_counts()
yellow_page_codes = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Yellow_Page_Code'].isin(low_asset_size_codes), 
                      'BE_Yellow_Page_Code'] = "other_yellow_page_codes"
yellow_page_dummy = pd.get_dummies(prospecting_table['BE_Yellow_Page_Code'], drop_first=True)
yellow_page_dummy.columns = ['yellow_page_' + str(col) for col in yellow_page_dummy.columns]
prospecting_table = pd.concat([prospecting_table, yellow_page_dummy], axis = 1)

#BE_Affluent_Neighborhood_Location_Indicator
affluent_neighborhood_dummy = pd.get_dummies(prospecting_table['BE_Affluent_Neighborhood_Location_Indicator'], drop_first=True)
affluent_neighborhood_dummy.columns = ['affluent_neighborhood_' + str(col) for col in affluent_neighborhood_dummy.columns]
prospecting_table = pd.concat([prospecting_table, affluent_neighborhood_dummy], axis = 1)

#BE_Affluent_Neighborhood_Location_Indicator
big_business_dummy = pd.get_dummies(prospecting_table['BE_Big_Business_Indicator'], drop_first=True)
big_business_dummy.columns = ['big_business_' + str(col) for col in big_business_dummy.columns]
prospecting_table = pd.concat([prospecting_table, big_business_dummy], axis = 1)

#BE_Growing_Shrinking_Indicator
sizes = prospecting_table['BE_Growing_Shrinking_Indicator'].value_counts()
growing_shrinking_indicator = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Growing_Shrinking_Indicator'].isin(growing_shrinking_indicator), 
                      'BE_Growing_Shrinking_Indicator'] = "other_growing_shrinking_indicator_codes"
growing_shrinking_indicator_dummy = pd.get_dummies(prospecting_table['BE_Growing_Shrinking_Indicator'], drop_first=True)
growing_shrinking_indicator_dummy.columns = ['growing_shrinking_indicator_' + str(col) for col in growing_shrinking_indicator_dummy.columns]
prospecting_table = pd.concat([prospecting_table, growing_shrinking_indicator_dummy], axis = 1)

#BE_High_Income_Executive_Indicator
high_income_dummy = pd.get_dummies(prospecting_table['BE_High_Income_Executive_Indicator'], drop_first=True)
high_income_dummy.columns = ['high_income_' + str(col) for col in high_income_dummy.columns]
prospecting_table = pd.concat([prospecting_table, high_income_dummy], axis = 1)

#BE_High_Tech_Business_Indicator
high_tech_dummy = pd.get_dummies(prospecting_table['BE_High_Tech_Business_Indicator'], drop_first=True)
high_tech_dummy.columns = ['high_tech_' + str(col) for col in high_tech_dummy.columns]
prospecting_table = pd.concat([prospecting_table, high_tech_dummy], axis = 1)

#BE_Medium_Business_Entrepreneur_Indicator
medium_business_dummy = pd.get_dummies(prospecting_table['BE_Medium_Business_Entrepreneur_Indicator'], drop_first=True)
medium_business_dummy.columns = ['medium_business_' + str(col) for col in medium_business_dummy.columns]
prospecting_table = pd.concat([prospecting_table, medium_business_dummy], axis = 1)

#BE_Small_Business_Entrepreneur_Indicator
small_business_dummy = pd.get_dummies(prospecting_table['BE_Small_Business_Entrepreneur_Indicator'], drop_first=True)
small_business_dummy.columns = ['small_business_' + str(col) for col in small_business_dummy.columns]
prospecting_table = pd.concat([prospecting_table, small_business_dummy], axis = 1)


#prospecting_table["BE Census Tract"] check!!
prospecting_table.drop(['BE_Telephone_NumberF1', 'BE_Web_Address_URL', 'BE_Primary_SIC_Code', 
                        'BE_Secondary_SIC_Code_1', 'BE_CSA_Description', 'BE_Location_Employment_Size_Code',
                        'BE_Location_Sales_Volume_Code', 
                        'BE_Asset_Size_Indicator', 'BE_ABI_Number', 'BE_Business_Status_Code', 
                        'BE_Business_Status_Description', 'BE_Public_Private_Code', 'BE_Public_Filing_Indicator',
                        'BE_Individual_Firm_Code', 'BE_Individual_Firm_Description', 'BE_Yellow_Page_Code', 
                        'BE_Business_Credit_Score_Code2', 'BE_Affluent_Neighborhood_Location_Indicator', 
                        'BE_Big_Business_Indicator', 'BE_Female_Business_Exec_Owner_Indicator', 
                        'BE_Growing_Shrinking_Indicator', 'BE_High_Income_Executive_Indicator', 
                        'BE_High_Tech_Business_Indicator', 'BE_Medium_Business_Entrepreneur_Indicator', 
                        'BE_Small_Business_Entrepreneur_Indicator', 'BE_White_Collar_Indicator', 
                        'BE_1849_Record_Marker', 'BE_Modeled_Employment_Size_Indicator', 
                        'BE_Location_Sales_Volume_Description', 'BE_NAICS_Code8'], axis =1, inplace =True)


temp = prospecting_table.dropna()
temp.reset_index(drop=True, inplace=True)
temp_keys = temp[['source', 'sourcesystemid', 'BE_Company_Name']]
temp.drop(['source', 'sourcesystemid', 'BE_Company_Name'], axis = 1, inplace= True)
X_train = temp
#X_train = temp.iloc[:700,:]
#X_test = temp.iloc[700:,:]

# fit the model
rng = np.random.RandomState(158563)
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_train_probas = clf.decision_function(X_train)
#Train set distribution
n, bins, patches = plt.hist(y_pred_train_probas)
plt.show()

#############################################################################

temp = pd.concat([temp, 
                  pd.Series(y_pred_train_probas)], axis= 1)
temp.rename(columns={ temp.columns[-1]: "prospect" }, inplace = True)
temp['result'] = np.where(temp.prospect > 0, 1, 0)

#since very few of the customers are non prospects, lets add the non 
# prospects artificially
from imblearn.over_sampling import SMOTE 
#x_train, x_val, y_train, y_val = train_test_split(temp, temp[-1],
#                                                  test_size = 0,
#                                                  random_state=12)
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(X_train, temp['result'])
balanced_dataset = pd.concat([pd.DataFrame(x_train_res), pd.Series(y_train_res)], axis = 1)
temp.drop('prospect', axis =1 , inplace = True)
balanced_dataset.columns = temp.columns
balanced_dataset.to_csv('prospect_list_datarobot.csv')
#############################################################################


# Test set distribution
y_pred_test = clf.predict(X_test)
y_predict_probas = clf.decision_function(X_test)
n, bins, patches = plt.hist(y_predict_probas)
plt.show()

#############################################################################################################
#now lets predict our prospects
kristen_list = pd.read_csv('prospect_list.csv')
kristen_list['BE_Primary_SIC_Code'] = kristen_list['BE_Primary_SIC_Code'].astype(str)
kristen_list['BE_Primary_SIC_Code'] = kristen_list['BE_Primary_SIC_Code'].str[:4]
kristen_list['BE_NAICS_Code8'] = kristen_list['BE_NAICS_Code8'].astype(str)
kristen_list['BE_NAICS_Code8'] = kristen_list['BE_NAICS_Code8'].str[:6]

prospecting_table = kristen_list

#data cleanup
prospecting_table = prospecting_table.drop(prospecting_table.columns[2:17],1) #check this line!!

prospecting_table.drop(['MailingAddress2', 'MailingCity', 'MailingState', 
                        'ZipCode', 'Zip4', 'MailingCountry', 'BE_Match_Level', 
                        'BE_Match_Score', 'BE_Record_Type_Code', 'BE_Contact_Manager','BE_Selected_SIC_Description', 
                        'BE_Franchise_Specialty_Description1', 'BE_Franchise_Specialty_Description2', 
                        'BE_Franchise_Specialty_Description3', 'BE_Franchise_Specialty_Description4', 
                        'BE_Franchise_Specialty_Description5', 'BE_Franchise_Specialty_Code6',
                        'BE_Franchise_Specialty_Description6', 'BE_Primary_SIC_Description', 
                        'BE_Secondary_SIC_Description_1', 'BE_Secondary_SIC_Description_2', 
                        'BE_Secondary_SIC_Description_3', 'BE_Secondary_SIC_Description_4', 
                        'BE_NAICS_Description', 'BE_Location_Employment_Size_Description', 
                        'BE_Corporate_Employment_Size_Description', 'BE_Selected_SIC_Code'], axis = 1, inplace = True)


#drop additional columns that were dropped in the previous iteration
prospecting_table.drop(['BE_Production_Date','BE_Obsolescence_Date', 'BE_Production_Date_Formatted', 
                        'BE_Source', 'BE_Book_Number'], axis=1, inplace= True)
    
#Dropping the lubes variables because the customers that we are interested in as prospects wont have any Lubes sales
prospecting_table = prospecting_table.loc[:, ~prospecting_table.columns.str.startswith('BE_Contact')]
prospecting_table.dropna(thresh=prospecting_table.shape[0]*0.5, how='all', axis=1, inplace = True)

#recode the fields 'BE_Year_SIC_Added_to_Record', 'BE Year First Appeared in Yellow Pages', 'BE Year Established', '
prospecting_table['BE_yellow_pages_number_of_years'] = 2019 - prospecting_table['BE_Year_First_Appeared_in_Yellow_Pages']

#subsequently drop the above fields
prospecting_table.drop(['BE_Year_First_Appeared_in_Yellow_Pages'],axis=1,inplace=True)


prospecting_table.drop([ 'BE_Site_Number', 'BE_Tele_Research_Date',
                        'BE_Call_Status_Code','BE_Call_Status_Description',
                        'BE_Business_Credit_Score_Description', 'BE_Delivery_Point_Bar_Code'], axis = 1, inplace = True)

#drop the geographic information since this  information will not matter if prospecting in a different territory
prospecting_table.drop(['BE_Primary_Address', 'BE_Primary_City_Name', 'BE_Primary_State_Abbreviation', 
                        'BE_Primary_ZIP_Code', 'BE_Primary_ZIP4_Code', 'BE_Primary_ZIP_Code_Plus_4', 
                        'BE_Primary_Carrier_Route_Code', 'BE_Primary_State_Code', 
                        'BE_County_Code', 'BE_County_Description', 'BE_CBSA_Code', 'BE_CBSA_Description', 
                        'BE_CSA_Code'], axis = 1, inplace = True)
    
    
#add in the abi number and the related fields   
sizes = prospecting_table['BE_ABI_Number'].value_counts()
low_abi = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_ABI_Number'].isin(low_abi), 'BE_ABI_Number'] = "other_abi_busiinesses"
abi_dummy = pd.get_dummies(prospecting_table.BE_ABI_Number, drop_first=True)
abi_dummy.columns = ['abi_' + str(col) for col in abi_dummy.columns]
prospecting_table = pd.concat([prospecting_table,abi_dummy], axis = 1)

#URL
sizes = prospecting_table['BE_Web_Address_URL'].value_counts()
low_url_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Web_Address_URL'].isin(low_url_codes), 
                      'BE_Web_Address_URL'] = "other_url_sites"
url_dummy = pd.get_dummies(prospecting_table['BE_Web_Address_URL'], drop_first=True)
url_dummy.columns = ['web_address_' + str(col) for col in url_dummy.columns]
prospecting_table = pd.concat([prospecting_table,url_dummy], axis = 1)

#metro micro indicator
metro_ind_dummy = pd.get_dummies(prospecting_table['BE_Metro_Micro_Indicator'], drop_first=True)
metro_ind_dummy.columns = ['metro_ind_' + str(col) for col in metro_ind_dummy.columns]
prospecting_table = pd.concat([prospecting_table,metro_ind_dummy], axis = 1)

#sic_code
sizes = prospecting_table['BE_Primary_SIC_Code'].value_counts()
low_sic_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Primary_SIC_Code'].isin(low_sic_codes), 
                      'BE_Primary_SIC_Code'] = "other_sic_codes"
sic_codes_dummy = pd.get_dummies(prospecting_table['BE_Primary_SIC_Code'], drop_first=True)
sic_codes_dummy.columns = ['sic_codes_' + str(col) for col in sic_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,sic_codes_dummy], axis = 1)

#naics_code
sizes = prospecting_table['BE_NAICS_Code8'].value_counts()
low_naics_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_NAICS_Code8'].isin(low_naics_codes), 
                      'BE_NAICS_Code8'] = "other_naics_codes"
naics_codes_dummy = pd.get_dummies(prospecting_table['BE_NAICS_Code8'], drop_first=True)
naics_codes_dummy.columns = ['naics_codes_' + str(col) for col in naics_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,naics_codes_dummy], axis = 1)

#employment_size_code
sizes = prospecting_table['BE_Location_Employment_Size_Code'].value_counts()
low_employment_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Location_Employment_Size_Code'].isin(low_employment_codes), 
                      'BE_Location_Employment_Size_Code'] = "other_employment_size_codes"
employment_codes_dummy = pd.get_dummies(prospecting_table['BE_Location_Employment_Size_Code'], drop_first=True)
employment_codes_dummy.columns = ['employment_size_codes_' + str(col) for col in employment_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,employment_codes_dummy], axis = 1)

#sales volume description
sizes = prospecting_table['BE_Location_Sales_Volume_Description'].value_counts()
low_sales_vol = sizes.index[sizes < prospecting_table.shape[0]*0.02]
prospecting_table.loc[prospecting_table['BE_Location_Sales_Volume_Description'].isin(low_sales_vol), 
                      'BE_Location_Sales_Volume_Description'] = "other_employment_size_codes"
sales_vol_dummy = pd.get_dummies(prospecting_table['BE_Location_Sales_Volume_Description'], drop_first=True)
sales_vol_dummy.columns = ['sales_vol_desc_' + str(col) for col in sales_vol_dummy.columns]
prospecting_table = pd.concat([prospecting_table,sales_vol_dummy], axis = 1)

#business status
sizes = prospecting_table['BE_Business_Status_Code'].value_counts()
low_bus_status = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE_Business_Status_Code'].isin(low_bus_status), 
                      'BE_Business_Status_Code'] = "other_business_status_codes"
bus_status_dummy = pd.get_dummies(prospecting_table['BE_Business_Status_Code'], drop_first=True)
bus_status_dummy.columns = ['business_status_' + str(col) for col in bus_status_dummy.columns]
prospecting_table = pd.concat([prospecting_table,bus_status_dummy], axis = 1)

#public private code
sizes = prospecting_table['BE_Public_Private_Code'].value_counts()
low_public_codes = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Public_Private_Code'].isin(low_public_codes), 
                      'BE_Public_Private_Code'] = "other_private_public_codes"
public_private_dummy = pd.get_dummies(prospecting_table['BE_Public_Private_Code'], drop_first=True)
public_private_dummy.columns = ['public_private_' + str(col) for col in public_private_dummy.columns]
prospecting_table = pd.concat([prospecting_table,public_private_dummy], axis = 1)

#BE_Asset_Size_Indicator
sizes = prospecting_table['BE_Asset_Size_Indicator'].value_counts()
low_asset_size_codes = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Asset_Size_Indicator'].isin(low_asset_size_codes), 
                      'BE_Asset_Size_Indicator'] = "other_low_asset_size_codes"
low_asset_size_dummy = pd.get_dummies(prospecting_table['BE_Asset_Size_Indicator'], drop_first=True)
low_asset_size_dummy.columns = ['public_private_' + str(col) for col in low_asset_size_dummy.columns]
prospecting_table = pd.concat([prospecting_table,low_asset_size_dummy], axis = 1)


#BE_Public_Filing_Indicator
public_filing_dummy = pd.get_dummies(prospecting_table['BE_Public_Filing_Indicator'], drop_first=True)
public_filing_dummy.columns = ['public_filing_' + str(col) for col in public_filing_dummy.columns]
prospecting_table = pd.concat([prospecting_table,public_filing_dummy], axis = 1)

#BE_Yellow_Page_Code
sizes = prospecting_table['BE_Yellow_Page_Code'].value_counts()
yellow_page_codes = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Yellow_Page_Code'].isin(low_asset_size_codes), 
                      'BE_Yellow_Page_Code'] = "other_yellow_page_codes"
yellow_page_dummy = pd.get_dummies(prospecting_table['BE_Yellow_Page_Code'], drop_first=True)
yellow_page_dummy.columns = ['yellow_page_' + str(col) for col in yellow_page_dummy.columns]
prospecting_table = pd.concat([prospecting_table, yellow_page_dummy], axis = 1)

#BE_Affluent_Neighborhood_Location_Indicator
affluent_neighborhood_dummy = pd.get_dummies(prospecting_table['BE_Affluent_Neighborhood_Location_Indicator'], drop_first=True)
affluent_neighborhood_dummy.columns = ['affluent_neighborhood_' + str(col) for col in affluent_neighborhood_dummy.columns]
prospecting_table = pd.concat([prospecting_table, affluent_neighborhood_dummy], axis = 1)

#BE_Affluent_Neighborhood_Location_Indicator
big_business_dummy = pd.get_dummies(prospecting_table['BE_Big_Business_Indicator'], drop_first=True)
big_business_dummy.columns = ['big_business_' + str(col) for col in big_business_dummy.columns]
prospecting_table = pd.concat([prospecting_table, big_business_dummy], axis = 1)

#BE_Growing_Shrinking_Indicator
sizes = prospecting_table['BE_Growing_Shrinking_Indicator'].value_counts()
growing_shrinking_indicator = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE_Growing_Shrinking_Indicator'].isin(growing_shrinking_indicator), 
                      'BE_Growing_Shrinking_Indicator'] = "other_growing_shrinking_indicator_codes"
growing_shrinking_indicator_dummy = pd.get_dummies(prospecting_table['BE_Growing_Shrinking_Indicator'], drop_first=True)
growing_shrinking_indicator_dummy.columns = ['growing_shrinking_indicator_' + str(col) for col in growing_shrinking_indicator_dummy.columns]
prospecting_table = pd.concat([prospecting_table, growing_shrinking_indicator_dummy], axis = 1)

#BE_High_Income_Executive_Indicator
high_income_dummy = pd.get_dummies(prospecting_table['BE_High_Income_Executive_Indicator'], drop_first=True)
high_income_dummy.columns = ['high_income_' + str(col) for col in high_income_dummy.columns]
prospecting_table = pd.concat([prospecting_table, high_income_dummy], axis = 1)

#BE_High_Tech_Business_Indicator
high_tech_dummy = pd.get_dummies(prospecting_table['BE_High_Tech_Business_Indicator'], drop_first=True)
high_tech_dummy.columns = ['high_tech_' + str(col) for col in high_tech_dummy.columns]
prospecting_table = pd.concat([prospecting_table, high_tech_dummy], axis = 1)

#BE_Medium_Business_Entrepreneur_Indicator
medium_business_dummy = pd.get_dummies(prospecting_table['BE_Medium_Business_Entrepreneur_Indicator'], drop_first=True)
medium_business_dummy.columns = ['medium_business_' + str(col) for col in medium_business_dummy.columns]
prospecting_table = pd.concat([prospecting_table, medium_business_dummy], axis = 1)

#BE_Small_Business_Entrepreneur_Indicator
small_business_dummy = pd.get_dummies(prospecting_table['BE_Small_Business_Entrepreneur_Indicator'], drop_first=True)
small_business_dummy.columns = ['small_business_' + str(col) for col in small_business_dummy.columns]
prospecting_table = pd.concat([prospecting_table, small_business_dummy], axis = 1)


#prospecting_table["BE Census Tract"] check!!
prospecting_table.drop(['BE_Telephone_NumberF1', 'BE_Web_Address_URL', 'BE_Primary_SIC_Code', 
                        'BE_Secondary_SIC_Code_1', 'BE_CSA_Description', 'BE_Location_Employment_Size_Code',
                        'BE_Location_Sales_Volume_Code', 
                        'BE_Asset_Size_Indicator', 'BE_ABI_Number', 'BE_Business_Status_Code', 
                        'BE_Business_Status_Description', 'BE_Public_Private_Code', 'BE_Public_Filing_Indicator',
                        'BE_Individual_Firm_Code', 'BE_Individual_Firm_Description', 'BE_Yellow_Page_Code', 
                        'BE_Business_Credit_Score_Code2', 'BE_Affluent_Neighborhood_Location_Indicator', 
                        'BE_Big_Business_Indicator', 'BE_Female_Business_Exec_Owner_Indicator', 
                        'BE_Growing_Shrinking_Indicator', 'BE_High_Income_Executive_Indicator', 
                        'BE_High_Tech_Business_Indicator', 'BE_Medium_Business_Entrepreneur_Indicator', 
                        'BE_Small_Business_Entrepreneur_Indicator', 'BE_White_Collar_Indicator', 
                        'BE_1849_Record_Marker', 'BE_Modeled_Employment_Size_Indicator', 
                        'BE_Location_Sales_Volume_Description', 'BE_NAICS_Code8'], axis =1, inplace =True)






























































af_prospects = prospecting_table.dropna()
af_prospects = af_prospects[af_prospects.CustomerGroup == 'AF']
af_prospects.drop(['Lubes Combined', 'AF_Yr16','Gain_Yr16', 'Oil_Yr16', 'PE_Yr16', 'AF_Yr17', 
           'Gain_Yr17', 'Oil_Yr17', 'PE_Yr17', 'AF_Yr18', 'Gain_Yr18', 'Oil_Yr18',
           'PE_Yr18', 'Total Rev.'],axis = 1, inplace = True)
af_prospect_keys = af_prospects[['CustomerEnterpriseKey', 'CustomerKey_y']]
af_prospects.drop(['CustomerEnterpriseKey', 'CustomerKey_y'], axis = 1, inplace= True)

X_af_prospects = af_prospects.drop(['CustomerGroup'], axis = 1)
y_af_prospect_predict = clf.predict(X_af_prospects)
y_af_prospect_probas = clf.decision_function(X_af_prospects)

n, bins, patches = plt.hist(y_af_prospect_probas)
plt.show()

#connect the prospects back to the keys
rf_model_dataset = X_af_prospects.reset_index()
y_prospect_probas = pd.DataFrame(y_af_prospect_probas)
rf_model_dataset = pd.concat([rf_model_dataset,y_prospect_probas], axis = 1)
rf_model_dataset.rename(columns={0:'probability'}, inplace = True)
#rf_model_dataset['prospect'] = np.where(rf_model_dataset.probability < 0.08, 0, 1)
#rf_model_dataset.to_csv('lubes_pcmo.csv')
#rf_model_dataset.drop(['probability'], axis =1, inplace = True)
#combine the rf model dataset with the af_prospect keys
af_prospect_keys = af_prospect_keys.reset_index()
lubes_output = pd.merge(af_prospect_keys, rf_model_dataset, 
                        left_on = 'index', right_on = 'index', how = 'inner')
lubes_output = lubes_output[['CustomerEnterpriseKey', 'probability']]
lubes_output.to_csv('lubes_output.csv')
lubes_prospects = pd.read_csv('lubes_output.csv')

colins_dedupe_list = pd.read_csv('dedupe_list_from_colin.csv')
colins_dedupe_list.CustomerEnterpriseKey = 'A' + colins_dedupe_list.CustomerEnterpriseKey.astype(str)
customerkeys = pd.merge(lubes_output, colins_dedupe_list, left_on = 'CustomerEnterpriseKey', 
                        right_on = 'CustomerEnterpriseKey', how = 'left')
customerkeys.drop_duplicates(subset = 'CustomerEnterpriseKey' , keep = 'first', inplace  = True)
customerkeys = list(customerkeys['CustomerKey (DimCustomer)'])
query_for_customerkeys = "select * from dw.DimCustomer where CustomerKey in " + str(customerkeys)
query_for_customerkeys = query_for_customerkeys.replace('[', '(')
query_for_customerkeys = query_for_customerkeys.replace(']', ')')

#Connect to the database to pull the Address and Name information for the deduped IDs
cnxn = pyodbc.connect('Driver={ODBC Driver 13 for SQL Server}; \
                      Server=tcp:dbdaz001.database.windows.net,1433; \
                      Database=IADevDB1;Uid=IAUSER1;Pwd=IADBUser1$; \
                      Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;')


customer_information = pd.read_sql(query_for_customerkeys, cnxn)
customer_information = pd.merge(lubes_prospects, customer_information, left_on = 'CustomerKeys', 
                                right_on = 'CustomerKey', how = 'left')
customer_information.drop(['Unnamed: 0', 'CustomerKeys', 'CreditLimit', 'CustomerType'], 
                          axis = 1, inplace = True)
###Data clean up and save
customer_information[['ZipCode', 'Zip4', 'Zip_drop']] = customer_information['MailingPostalCode'].str.split('-',expand=True)

# convert all the states to 2 digits
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

us_country_abbrev = {
        'United States' : 'USA',
        'Canada': 'CAN',
        'Bahamas': 'BHS'
        }

#replace the state and the country using the abbraviations to standardize the data
customer_information['MailingState'] = customer_information['MailingState'].replace(us_state_abbrev)
customer_information['MailingCountry'] = customer_information['MailingCountry'].replace(us_country_abbrev)
#fill in the missing country information if state matches a US state
customer_information['MailingCountry'] = customer_information['MailingCountry'].replace('', np.NaN)
#fill in the country if the state matches the US states
customer_information['MailingCountry']=np.where(customer_information['MailingState'].isin(list(us_state_abbrev.values())),customer_information['MailingCountry'].fillna('USA'),customer_information['MailingCountry'])

#change city and customer name to title
def change_title(string):   
    return string.title()

customer_information.MailingCity = customer_information.apply(lambda x: change_title(x.MailingCity), axis=1)
customer_information.CustomerName = customer_information.apply(lambda x: change_title(x.CustomerName), axis=1)
customer_information.MailingAddress = customer_information.apply(lambda x: change_title(x.MailingAddress), axis=1)

#drop all the rows that is missing the address and the phone number
cols = ['MailingAddress', 'MailingCity','MailingState', 'CustomerPhone']
for col in cols: 
    customer_information[col] = customer_information[col].replace('', np.NaN)

customer_information.dropna(subset=['MailingAddress', 'MailingAddress2', 'MailingCity',
       'MailingState', 'CustomerPhone'],inplace=True, how='all')


import re
#format phone numbers
def phone_format(n):       
    if n == 'NotAvailable':        
        return n        
    else:
        n = re.sub('[^0-9]','', n)
        if len(n) < 10: 
            return('Not Available')
        else: 
            return format(int(n[:-1]), ",").replace(",", "-") + n[-1]


#replace nan with not available
customer_information['ZipCode'] = customer_information['ZipCode'].replace('', np.NaN)
#customer_information['Zip4'] = customer_information['Zip4'].replace('', np.NaN)
customer_information = customer_information.replace(np.nan, 'Not Available', regex=True)

##phone number formatting
customer_information.CustomerPhone = customer_information.CustomerPhone.str.strip()
customer_information.CustomerPhone = customer_information.CustomerPhone.str.replace(' ','')
customer_information.CustomerPhone = customer_information.CustomerPhone.str.replace(r'[^a-zA-Z0-9\n\.]', '')

customer_information.CustomerPhone = customer_information.apply(lambda x: phone_format(x.CustomerPhone), axis=1)
# make all the columns that are text to str    
for col in customer_information.columns: 
    if col != 'probability':
        customer_information[col] = customer_information[col].apply('="{}"'.format)
customer_information.drop(['MailingPostalCode', 'Zip_drop'], axis = 1, inplace = True)
customer_information.to_csv('final_prospect_list.csv')
