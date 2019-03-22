# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:49:40 2019

@author: A012050
"""

import pandas as pd
import pyodbc
import numpy as np
import datetime

infogroup_file = pd.read_csv('infogroup_raw_file.csv',dtype=str)
infogroup_file.columns

infogroup_subset = infogroup_file
infogroup_subset.CustomerKey = infogroup_subset.CustomerKey.str[1:]

#infogroup_subset_wo_pe = infogroup_subset[infogroup_subset.division != 'PE']
#infogroup_subset_pe = infogroup_subset[infogroup_subset.division == 'PE']
#infogroup_subset_pe.CustomerKey =infogroup_subset_pe.CustomerKey.apply(lambda x: '{0:0>6}'.format(x))
##merge in both the dfs
#
#infogroup_subset = pd.concat([infogroup_subset_wo_pe, infogroup_subset_pe], axis = 0)
#read in the oil customers 
#Connect to the SQLPDW Database
cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=SQLPDW,1433;"
                      "Database=USVBIAnalytics;"
                      "Trusted_Connection=yes;")

   
oil_cust = pd.read_sql('SELECT CAST(\'Oil\' as varchar(50)) AS Source, CustomerKey AS info_key, \
                       CustomerAccountId AS sourceid FROM USVBIAnalytics.DWOil.vwDimCustomer', cnxn)

af_cust = pd.read_sql('SELECT CAST(\'AutoForce\' as varchar(50)) AS Source, CustomerKey AS info_key, \
                       CustomerID AS sourceid FROM USVBIAnalytics.DWAF.vwDimCustomer', cnxn)

gain_cust = pd.read_sql('SELECT CAST(\'Gain\' as varchar(50)) AS Source, CustomerKey AS info_key, \
                       CustomerID AS sourceid FROM USVBIAnalytics.DWGain.vwDimCustomer', cnxn)

#Connect to the PE Ventus Database
cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=APPWS131,1433;"
                      "Database=Ventus-USPE;"
                      "Trusted_Connection=yes;")

pe_cust = pd.read_sql('SELECT CAST(\'PE\' as varchar(50)) AS Source, CustomerNo AS info_key, \
                       CustomerNo AS sourceid FROM [Ventus-USPE].AR.Customer', cnxn)

#Connect to the P21Prod Database
cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=USVSQLP08B\SQLP08B,1433;"
                      "Database=P21Prod;"
                      "Trusted_Connection=yes;")

lubricants_cust = pd.read_sql('SELECT CAST(\'Lubricants (half)\' as varchar(50)) AS Source, customer_id AS info_key, \
                       customer_id AS sourceid FROM P21Prod.dbo.p21_customer_view', cnxn)


#Combine all the IDs
all_ids_combined = pd.concat([af_cust, oil_cust], axis= 0)
all_ids_combined = pd.concat([all_ids_combined, pe_cust], axis= 0)
all_ids_combined = pd.concat([all_ids_combined, gain_cust], axis= 0)
all_ids_combined = pd.concat([all_ids_combined, lubricants_cust], axis= 0)

all_ids_combined.info_key = all_ids_combined.info_key.astype(str)
all_ids_combined[['info_key','decimal_del']] = all_ids_combined['info_key'].str.split('.',expand=True)
all_ids_combined.drop(['decimal_del'], axis = 1, inplace = True) 
#merge in the infogroup with the all ids to get the sourceid linked to the customers
infogroup_sourceid = pd.merge(infogroup_subset, all_ids_combined, left_on = ['division','CustomerKey'], 
                              right_on = ['Source', 'info_key'], how ='outer')
infogroup_sourceid = infogroup_sourceid.dropna(axis=0, how='all', subset=['CustomerKey','division'])
infogroup_sourceid.drop(['Source','info_key'], axis = 1, inplace = True)

#drop all the rows where the sourceid is nan . these source ids are all from pe
infogroup_sourceid.dropna(subset=['sourceid'], inplace =True)

infogroup_sourceid.division = np.where(infogroup_sourceid.division == 'AutoForce', 'AF',infogroup_sourceid.division)
infogroup_sourceid.division = np.where(infogroup_sourceid.division == 'Lubricants (half)', 'Lubes',infogroup_sourceid.division)

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
infogroup_sourceid.sourceid = infogroup_sourceid.sourceid. astype(str)
infogroup_sourceid[['sourceid','decimal_del']] = infogroup_sourceid['sourceid'].str.split('.',expand=True)

temp = pd.merge(infogroup_sourceid, dim_cust_list, left_on = ['division', 'sourceid'], right_on = ['Source','SourceSystemID'], how='left')
del pe_cust, oil_cust, lubricants_cust, gain_cust, af_cust


#Read in the de-dupe file from Colin
dedup_file = pd.read_csv('dedupe_list_from_colin.csv')

#Merge on the above tables so that we now can link the actual Customer IDs 
#to the newly created ones
infogroup_enterprise_key = pd.merge(temp, dedup_file, 
                                left_on=['CustomerKey_y'], 
                                right_on = ['CustomerKey (DimCustomer)'], 
                                how='left')
infogroup_enterprise_key['CustomerEnterpriseKey'] = 'A' + infogroup_enterprise_key['CustomerEnterpriseKey'].astype(str)
infogroup_enterprise_key['CustomerEnterpriseKey'] = infogroup_enterprise_key['CustomerEnterpriseKey'].str[:-2]

infogroup_enterprise_key['BE Primary SIC Code'] = infogroup_enterprise_key['BE Primary SIC Code'].astype(str)
infogroup_enterprise_key['BE Primary SIC Code'] = infogroup_enterprise_key['BE Primary SIC Code'].str[:4]
infogroup_enterprise_key['BE NAICS Code8'] = infogroup_enterprise_key['BE NAICS Code8'].astype(str)
infogroup_enterprise_key['BE NAICS Code8'] = infogroup_enterprise_key['BE NAICS Code8'].str[:6]
infogroup_enterprise_key = infogroup_enterprise_key.drop_duplicates(subset='CustomerEnterpriseKey', keep='first')
#infogroup_enterprise_key = infogroup_enterprise_key[['CustomerEnterpriseKey','BE Primary SIC Code', 'BE NAICS Code8']]

#pivot on customer enterprise key
customer_master_file = pd.read_csv('customer_master_file_delete.csv')
final_file = pd.merge(customer_master_file, infogroup_enterprise_key, left_on = 'CustomerEnterpriseKey', right_on = 'CustomerEnterpriseKey', how='left')
final_file.rename(columns = {'sourceid_0':'SourceID'},inplace=True)
final_file = final_file.loc[:, ~final_file.columns.str.startswith('sourceid_')]

#save the final_file into csv
final_file.to_csv('lubes_prospectingv2.csv')

#############################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
import os
from sklearn.model_selection  import train_test_split
from sklearn.cluster import KMeans
import sklearn.metrics
#Finding optimal no. of clusters
from scipy.spatial.distance import cdist


prospecting_table = pd.read_csv('lubes_prospectingv2.csv')
prospecting_table.dropna(subset=['CustomerGroup'],inplace=True)
#prospecting_table.dropna(subset = ['CustomerGroup'], inplace = True)

#data cleanup
prospecting_table.drop(['Lubes Customer','AF Cust'],axis=1,inplace= True)
prospecting_table.drop(['SeqCntNum','CustomerName_x','addressType','addressLine','addressLine2', 'customerPhone',
                        'BE Match Level', ], axis=1, inplace= True)
    
#drop the industrial customers
industrial_customers = ['A45830', 'A48848', 'A48653', 'A61582', 'A61315', 'A22554']
prospecting_table = prospecting_table[~prospecting_table.CustomerEnterpriseKey.isin(industrial_customers)]

prospecting_table = prospecting_table[prospecting_table['country'] == 'United States']
prospecting_table.drop(['country','BE Match Score','BE Contact Manager','BE Primary Address',
                        'BE Primary City Name','BE Primary State Abbreviation','BE Primary ZIP Code','BE Primary ZIP4 Code',
                        'BE Primary State Code','BE Selected SIC Description','BE Secondary SIC Description 1',
                        'BE Secondary SIC Description 3', 'BE Secondary SIC Description 4', 'BE NAICS Description',
                        'BE Location Employment Size Description','BE Production Date','BE Obsolescence Date', 'BE Production Date Formatted', 
                        'BE Source', 'BE Book Number', 'decimal_del'], axis=1, inplace= True)
    
#Dropping the lubes variables because the customers that we are interested in as prospects wont have any Lubes sales
prospecting_table = prospecting_table.loc[:, ~prospecting_table.columns.str.startswith('Lubes_')]
prospecting_table = prospecting_table.loc[:, ~prospecting_table.columns.str.startswith('BE Contact')]
prospecting_table.dropna(thresh=prospecting_table.shape[0]*0.5, how='all', axis=1, inplace = True)

#recode the fields 'BE_Year_SIC_Added_to_Record', 'BE Year First Appeared in Yellow Pages', 'BE Year Established', '
prospecting_table['BE_yellow_pages_number_of_years'] = 2019 - prospecting_table['BE Year First Appeared in Yellow Pages']

#subsequently drop the above fields
prospecting_table.drop(['BE Year First Appeared in Yellow Pages'],axis=1,inplace=True)

prospecting_table['Customer_yr_start'] = 2019 - prospecting_table['FirstInvoiceDate'].str[-4:].astype(float)
prospecting_table.drop(['FirstInvoiceDate', 'LastInvoiceDate'], axis = 1, inplace = True)
prospecting_table.drop(['BE Company Name', 'BE Primary ZIP Code Plus 4', 'BE Telephone NumberF1', 
                        'BE Primary SIC Description','BE ABI Number', 'BE Site Number', 'BE Tele Research Date',
                        'BE Call Status Code', 'BE Call Status Description', 'BE Business Credit Score Description', 
                        'BE Delivery Point Bar Code','Source','CustomerName_y','CustomerPhone','CustomerName', 
                        'sourceid', 'SourceSystemID','CustomerKey (DimCustomer)', 'SourceID', 'CustomerKey_x', 'division',
                        'zipCode'], axis = 1, inplace = True)

#states
sizes = prospecting_table['state'].value_counts()
low_states = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['state'].isin(low_states), 'state'] = "other_states"
state_dummy = pd.get_dummies(prospecting_table.state, drop_first=True)
state_dummy.columns = ['state_' + str(col) for col in state_dummy.columns]
prospecting_table = pd.concat([prospecting_table,state_dummy], axis = 1)

#cities
sizes = prospecting_table['city'].value_counts()
low_cities = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['city'].isin(low_cities), 'city'] = "other_cities"
city_dummy = pd.get_dummies(prospecting_table.city, drop_first=True)
city_dummy.columns = ['city_' + str(col) for col in city_dummy.columns]
prospecting_table = pd.concat([prospecting_table,city_dummy], axis = 1)

#county desc
sizes = prospecting_table['BE County Description'].value_counts()
low_counties = sizes.index[sizes < prospecting_table.shape[0]*0.05]
prospecting_table.loc[prospecting_table['BE County Description'].isin(low_counties), 
                      'BE County Description'] = "other_counties"
counties_dummy = pd.get_dummies(prospecting_table['BE County Description'], drop_first=True)
counties_dummy.columns = ['counties_' + str(col) for col in counties_dummy.columns]
prospecting_table = pd.concat([prospecting_table,counties_dummy], axis = 1)

#cbsa code
sizes = prospecting_table['BE CBSA Code'].value_counts()
low_cbsa_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE CBSA Code'].isin(low_cbsa_codes), 
                      'BE CBSA Code'] = "other_cbsa_codes"
cbsa_dummy = pd.get_dummies(prospecting_table['BE CBSA Code'], drop_first=True)
cbsa_dummy.columns = ['cbsa_' + str(col) for col in cbsa_dummy.columns]
prospecting_table = pd.concat([prospecting_table,cbsa_dummy], axis = 1)

#metro micro indicator
metro_ind_dummy = pd.get_dummies(prospecting_table['BE Metro Micro Indicator'], drop_first=True)
metro_ind_dummy.columns = ['metro_ind_' + str(col) for col in metro_ind_dummy.columns]
prospecting_table = pd.concat([prospecting_table,metro_ind_dummy], axis = 1)

#csa code
sizes = prospecting_table['BE CSA Code'].value_counts()
low_csa_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE CSA Code'].isin(low_csa_codes), 
                      'BE CSA Code'] = "other_csa_codes"
csa_dummy = pd.get_dummies(prospecting_table['BE CSA Code'], drop_first=True)
csa_dummy.columns = ['csa_' + str(col) for col in csa_dummy.columns]
prospecting_table = pd.concat([prospecting_table,csa_dummy], axis = 1)

#sic_code
sizes = prospecting_table['BE Primary SIC Code'].value_counts()
low_sic_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE Primary SIC Code'].isin(low_sic_codes), 
                      'BE Primary SIC Code'] = "other_sic_codes"
sic_codes_dummy = pd.get_dummies(prospecting_table['BE Primary SIC Code'], drop_first=True)
sic_codes_dummy.columns = ['sic_codes_' + str(col) for col in sic_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,sic_codes_dummy], axis = 1)

#naics_code
sizes = prospecting_table['BE NAICS Code8'].value_counts()
low_naics_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE NAICS Code8'].isin(low_naics_codes), 
                      'BE NAICS Code8'] = "other_naics_codes"
naics_codes_dummy = pd.get_dummies(prospecting_table['BE NAICS Code8'], drop_first=True)
naics_codes_dummy.columns = ['naics_codes_' + str(col) for col in naics_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,naics_codes_dummy], axis = 1)

#employment_size_code
sizes = prospecting_table['BE Location Employment Size Code'].value_counts()
low_employment_codes = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE Location Employment Size Code'].isin(low_employment_codes), 
                      'BE Location Employment Size Code'] = "other_employment_size_codes"
employment_codes_dummy = pd.get_dummies(prospecting_table['BE Location Employment Size Code'], drop_first=True)
employment_codes_dummy.columns = ['employment_size_codes_' + str(col) for col in employment_codes_dummy.columns]
prospecting_table = pd.concat([prospecting_table,employment_codes_dummy], axis = 1)

#sales volume description
sizes = prospecting_table['BE Location Sales Volume Description'].value_counts()
low_sales_vol = sizes.index[sizes < prospecting_table.shape[0]*0.02]
prospecting_table.loc[prospecting_table['BE Location Sales Volume Description'].isin(low_sales_vol), 
                      'BE Location Sales Volume Description'] = "other_employment_size_codes"
sales_vol_dummy = pd.get_dummies(prospecting_table['BE Location Sales Volume Description'], drop_first=True)
sales_vol_dummy.columns = ['sales_vol_desc_' + str(col) for col in sales_vol_dummy.columns]
prospecting_table = pd.concat([prospecting_table,sales_vol_dummy], axis = 1)

#business status
sizes = prospecting_table['BE Business Status Code'].value_counts()
low_bus_status = sizes.index[sizes < prospecting_table.shape[0]*0.01]
prospecting_table.loc[prospecting_table['BE Business Status Code'].isin(low_bus_status), 
                      'BE Business Status Code'] = "other_business_status_codes"
bus_status_dummy = pd.get_dummies(prospecting_table['BE Business Status Code'], drop_first=True)
bus_status_dummy.columns = ['business_status_' + str(col) for col in bus_status_dummy.columns]
prospecting_table = pd.concat([prospecting_table,bus_status_dummy], axis = 1)

#public private code
sizes = prospecting_table['BE Public Private Code'].value_counts()
low_public_codes = sizes.index[sizes < prospecting_table.shape[0]*0.1]
prospecting_table.loc[prospecting_table['BE Public Private Code'].isin(low_public_codes), 
                      'BE Public Private Code'] = "other_private_public_codes"
public_private_dummy = pd.get_dummies(prospecting_table['BE Public Private Code'], drop_first=True)
public_private_dummy.columns = ['public_private_' + str(col) for col in public_private_dummy.columns]
prospecting_table = pd.concat([prospecting_table,public_private_dummy], axis = 1)

#prospecting_table["BE Census Tract"] check!!
prospecting_table.drop(['state', 'city','BE Primary Carrier Route Code','BE County Code',
       'BE County Description','BE CBSA Code', 'BE CBSA Description','BE Metro Micro Indicator',
       'BE CSA Code', 'BE CSA Description', 'BE Census Tract','BE Selected SIC Code', 'BE Primary SIC Code', 
       'BE Secondary SIC Code 1', 'BE NAICS Code8', 'BE Location Employment Size Code',
       'BE Actual Location Employment Size5','BE Modeled Employment Size Indicator', 
       'BE Location Sales Volume Code','BE Location Sales Volume Code',
       'BE Location Sales Volume Description','BE Actual Location Sales Volume', 'BE Business Status Code',
       'BE Business Status Description', 'BE Public Private Code','BE Public Filing Indicator', 
       'BE Individual Firm Code', 'BE Individual Firm Description'], axis =1, inplace =True)

#dropping these columns because they are unrelated
prospecting_table.drop(['BE Block Group', 'BE Asset Size Indicator', 'BE Yellow Page Code', 
                        'BE Business Credit Score Code2', 'BE Ad Size Code','BE Ad Size Description',
                        'BE Square Footage8','BE Building Number of Multi Tenant Location',
                        'BE Affluent Neighborhood Location Indicator','BE Big Business Indicator',
                        'BE Female Business Exec Owner Indicator','BE Growing Shrinking Indicator',
                        'BE High Income Executive Indicator','BE High Tech Business Indicator',
                        'BE Medium Business Entrepreneur Indicator','BE Small Business Entrepreneur Indicator',
                        'BE White Collar Percentage Formatted','BE White Collar Indicator',
                        'BE 1849 Record Marker'], axis =1, inplace =True)


#temp = prospecting_table[['CustomerGroup','AF_Yr17', 'Lubes_Yr17','Total Rev.', 'city','state','BE Selected SIC Code',
#                          'BE NAICS Code8','BE Location Employment Size Code','BE Actual Location Employment Size5',
#                          'BE Location Sales Volume Code','BE Location Sales Volume Description', 
#                          'BE Affluent Neighborhood Location Indicator', 'BE Big Business Indicator','BE Growing Shrinking Indicator',
#                          'BE White Collar Percentage Formatted']]

#prospecting_table = pd.merge(prospecting_table, dedup_file, 
#                             left_on = 'CustomerEnterpriseKey', right_on = 'CustomerEnterpriseKey', how = 'left')
#prospecting_table.drop(['CustomerName'], axis = 1, inplace = True)

#temp = temp[temp.AF_Yr17 > 0]
#temp = temp[temp.Lubes_Yr17 > 0]
#temp =temp[temp['Total Rev.'] >0]
#temp['log_af_yr17'] = np.log(temp.AF_Yr17)
#temp['log_lubes_yr17'] = np.log(temp.Lubes_Yr17)
#temp['log_total_rev'] = np.log(temp['Total Rev.'])
#temp['log_location_emp'] = np.log(temp['BE Actual Location Employment Size5'])
#
#temp.drop(['AF_Yr17', 'Lubes_Yr17', 'Total Rev.',
#       'BE Actual Location Employment Size5'], axis = 1, inplace = True)
temp = prospecting_table.dropna()
temp.CustomerGroup.value_counts()
#AF       15914
#Both       974
#Lubes      579
temp_keys = temp[['CustomerEnterpriseKey', 'CustomerKey_y']]
temp.drop(['CustomerEnterpriseKey', 'CustomerKey_y'], axis = 1, inplace= True)
temp = temp[temp.CustomerGroup == 'Both']
# drop additional non useful variables
temp.drop(['Lubes Combined', 'AF_Yr16','Gain_Yr16', 'Oil_Yr16', 'PE_Yr16', 'AF_Yr17', 
           'Gain_Yr17', 'Oil_Yr17', 'PE_Yr17', 'AF_Yr18', 'Gain_Yr18', 'Oil_Yr18',
           'PE_Yr18', 'Total Rev.'],axis = 1, inplace = True)
X = temp.drop(['CustomerGroup'], axis = 1)
X_train = X.iloc[:700,:]
X_test = X.iloc[700:,:]

# fit the model
rng = np.random.RandomState(1235)
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_train_probas = clf.decision_function(X_train)
#Train set distribution
n, bins, patches = plt.hist(y_pred_train_probas)
plt.show()
# Test set distribution
y_pred_test = clf.predict(X_test)
y_predict_probas = clf.decision_function(X_test)
n, bins, patches = plt.hist(y_predict_probas)
plt.show()
#now lets predict our prospects
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