# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:54:35 2019

@author: kehu.chengbohua
"""

import  pandas as pd
import  numpy as np
import os
import gc
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import lightgbm as lgb

##加载文件位置
os.chdir('E:/td_union_model/')

union_data=pd.read_excel('union_td_buildingmodel_20190708.xlsx')
union_data['credentials_no_md5(客户身份证）']=union_data['credentials_no_md5(客户身份证）'].map(lambda x:x.replace('\t',''))
union_data['cust_name_md5（客户姓名）']=union_data['cust_name_md5（客户姓名）'].map(lambda x:x.replace('\t',''))
union_data['mobile_md5（客户手机号）']=union_data['mobile_md5（客户手机号）'].map(lambda x:x.replace('\t',''))
union_data['applydate']=union_data['applydate'].map(lambda x:x.replace('\t',''))
union_data['applydate']=union_data['applydate'].str.slice(0,10)
td_data=pd.read_csv('cd_cust_data.csv')
td_data['applydate']=td_data['apply_date_x'].str.slice(0,10)
combine_data=pd.merge(union_data,td_data,left_on=['credentials_no_md5(客户身份证）', 'cust_name_md5（客户姓名）', 'mobile_md5（客户手机号）','applydate']
      ,right_on=['credentials_no_md5_x', 'cust_name_md5_x', 'mobile_md5_x','applydate'],how='inner')
combine_data=combine_data.drop_duplicates(['credentials_no_md5(客户身份证）', 'cust_name_md5（客户姓名）', 'mobile_md5（客户手机号）','applydate'])
del td_data,union_data
gc.collect()
new_td_data=pd.read_csv('cd_cust_index_add.csv')
new_td_data['applydate']=new_td_data['apply_date_x'].str.slice(0,10)
combine_data=pd.merge(combine_data,new_td_data,on=['credentials_no_md5_x', 'cust_name_md5_x', 'mobile_md5_x','applydate'],how='left')
combine_data=combine_data.drop_duplicates(['credentials_no_md5_x', 'cust_name_md5_x', 'mobile_md5_x','applydate'])
del new_td_data
gc.collect()

## 车贷反欺诈
car_data=combine_data[(combine_data.fraud_y.isin([0,1])) & (combine_data['applydate']<='2018-12-01 00:00:00')]
del combine_data
gc.collect()
car_data=car_data.fillna(-9999)


def CalcIV(Xvar, Yvar):
   N_0 = np.sum(Yvar == 0)
   N_1 = np.sum(Yvar == 1)
   cuts = Xvar.quantile(np.arange(0, 1.1, 0.1)).values
   cuts=np.unique(cuts)
   if len(np.unique(Xvar))<=10 :
      N_0_group = np.zeros(np.unique(Xvar).shape)
      N_1_group = np.zeros(np.unique(Xvar).shape)
      unique_value=np.unique(Xvar)
   else:
      N_0_group = np.zeros((len(cuts)-1,))
      N_1_group = np.zeros((len(cuts)-1,))

      cuts=cuts.tolist()
      cuts[0] = -np.inf
      cuts[len(cuts)-1] = np.inf
      Xvar = pd.cut(Xvar, cuts, right=True)
      unique_value=Xvar.value_counts().index.tolist()

   for i in range(len(unique_value)):
         N_0_group[i] = Yvar[(Xvar == unique_value[i]) & (Yvar == 0)].count()
         N_1_group[i] = Yvar[(Xvar ==unique_value[i]) & (Yvar == 1)].count()
   iv = np.sum((N_0_group / N_0 - N_1_group / N_1) * np.log((N_0_group / N_0) / (N_1_group / N_1)))
   return iv


def caliv_batch(df, Yvar):
   df_Xvar = df.drop(Yvar, axis=1)
   ivlist = []
   names=[]
   for col in df_Xvar.columns:
      try:
          iv = CalcIV(df_Xvar[col], df[Yvar])
          ivlist.append(iv)
          names.append(col)
      except:
          pass
   iv_df = pd.DataFrame({'Var': names, 'Iv': ivlist}, columns=['Var', 'Iv'])

   return iv_df

iv_df=caliv_batch(car_data,'fraud_y')
iv_df.loc[iv_df['Iv']==np.inf,'Iv']=np.nan
iv_df.to_excel('iv_df_fraud_20190716.xlsx',index=False)
iv_df=iv_df.drop(iv_df[iv_df.Iv<0.02].index,axis=0)
iv_df=iv_df[iv_df.Var.isin(['fraud_y', 'hsdate_x','apply_y','age_x','hsdate_x_y','hsdate_x_x'])==False]
iv_df=iv_df.sort_values('Iv',ascending=False)


## 用lightgbm筛选变量
y = car_data['fraud_y']
x = car_data[iv_df.Var.values.tolist()]
x=x.drop(['hsdate_x_y','hsdate_x_x'],axis=1)
del car_data
gc.collect()
x_train,x_test,y_train,y_test=train_test_split(x[iv_df.Var.values.tolist()], y, test_size=0.3,stratify = y ,random_state=0)
bad=y_train[y_train==1]
good=x_train[y_train==0]
ori_good=good.sample(len(bad)*5,random_state=0)
ori_x_train=pd.concat([ori_good,x_train.loc[bad.index,:]])
ori_y_train=y_train.loc[ori_x_train.index]
del good,bad,ori_good
gc.collect()

params={'boosting_type':'gbdt','learning_rate':0.01,'objective':'binary',
        'metric':'auc','max_depth':2,'verbose':2,'random_state':0}
num_round=500
early_stopping_rounds=300
train_matrix=lgb.Dataset(ori_x_train,label=ori_y_train)
test_matrix=lgb.Dataset(x_test,label=y_test)
model=lgb.train(params,train_matrix,num_round,valid_sets=test_matrix)
final_cols=iv_df['Var'].values.tolist()
choose_cols=iv_df.loc[model.feature_importance()/model.feature_importance().sum()>0,'Var'].values.tolist()
columns_corrcoef=x[choose_cols].replace(-9999,np.nan).corr()
a=model.feature_importance()/model.feature_importance().sum().tolist()
feature_importance = pd.DataFrame({'var_name': final_cols, 'importance': a}).sort_values('importance', ascending=False)
importances=pd.DataFrame(np.array(a), index=final_cols,columns=['importance'])
del_columns=[]
del_desc={}
for onecol in choose_cols:
    meet_columns=columns_corrcoef.loc[columns_corrcoef[onecol].abs()>=0.7,:].index
    meet_importances=importances.loc[meet_columns,:]
    del_columns1=meet_importances.loc[meet_importances['importance']<meet_importances['importance'].max(),:].index.tolist()
    del_columns2=meet_importances.loc[meet_importances['importance']>=meet_importances['importance'].max(),:].index.tolist()
    del_desc2={}
    for cn in meet_importances.index:
        del_desc2[cn]=meet_importances.loc[cn,'importance']
    if len(del_columns1)!=0 and len(del_columns2)!=0:
        del_desc[del_columns2[0]]=del_desc2
    del_columns.extend(del_columns1)
choose_cols=[col for col in choose_cols if col not in del_columns]
print(len(choose_cols))
choose_cols=['i_ratio_cnt_partner_all_Imbank_365day',
 'i_ratio_freq_night_Loan_Offloan_365day',
 'i_max_length_record_Loan_P2pweb_365day',
 'i_length_first_Register_Imbank_365day',
 'i_max_length_record_Loan_con_365day',
 'i_max_length_record_Loan_P2pweb_90day',
 'i_ratio_freq_day_Login_P2pweb_365day',
 'i_length_first_lendingtime_Lending_Imbank_365day',
 'm_mean_length_event_all_Offloan_365day',
 'i_sum_loan_amount_Lending_all_90day',
 'i_get_node_rank_value_Loan_all_all',
 'm_freq_nightoutcall_MagicAuth_all_180day',
 'i_mean_length_event_Loan_Imbank_30day',
 'm_sum_duration_nightcall_MagicAuth_all_180day',
 'i_mean_freq_node_seq_partner_Loan_all_all',
 'm_cnt_node_dist2_Loan_all_all',
 'm_length_first_Register_all_180day',
 'i_mean_length_event_Login_Unconsumerfinance_180day',
 'i_max_length_record_Loan_Unconsumerfinance_15day',
 'm_ratio_cnt_grp_grey_Loan_all_all',
 'm_cnt_loanAPPinstlv2_Gen_Gen_365day',
 'm_mean_length_event_Login_finance_180day',
 'm_sum_loan_amount_Lending_Bank_180day',
 'm_length_first_all_O2O_365day',
 'm_ratio_freq_night_Register_Consumerfinance_90day']
#remove i_mean_length_event_Register_con_180day,m_ratio_freq_night_Register_Consumerfinance_90day,
#i_min_length_record_Loan_Consumerfinance_30day

final_choose_cols_fraud=choose_cols+feature_importance.loc[feature_importance.index[0:50],'var_name'].values.tolist()
final_choose_cols_fraud=list(set(final_choose_cols_fraud))
print(len(final_choose_cols_fraud))
pd.DataFrame(final_choose_cols_fraud).to_csv('final_choose_cols_fraud_20190716.csv')


## 筛选出来的变量确实率和均值情况
null_percents=x[final_choose_cols_fraud].replace({-9999:np.nan,-1111:np.nan,-999:np.nan}).isnull().sum()/x.shape[0]
null_percents=null_percents.reset_index().rename(columns={'index':'var_name',0:'null_percents'}).sort_values('null_percents',ascending=False)
car_data=pd.read_csv('car_data_apply_20190712.csv')
x['applymonth']=car_data.applydate.str.slice(0,7)
del car_data
gc.collect()
x=x[x['applymonth']!='2018-12']
null_matrix=x[final_choose_cols_fraud].replace({-9999:np.nan,-1111:np.nan,-999:np.nan}).isnull()
null_matrix['applymonth']=x['applymonth']
month_nullpercents=null_matrix.groupby(['applymonth'])[final_choose_cols_fraud].mean().T
month_nullpercents=month_nullpercents.reset_index().rename(columns={'index':'var_name'})
nullpercents=pd.merge(month_nullpercents,null_percents,on='var_name',how='left')
nullpercents.to_excel('nullpercents_fraud_20190716.xlsx',index=False)

mean_values=x[final_choose_cols_fraud].replace({-9999:np.nan,-1111:np.nan,-999:np.nan}).mean()
mean_values=mean_values.reset_index().rename(columns={'index':'var_name',0:'mean_value'})
mean_matrix=x[final_choose_cols_fraud].replace({-9999:np.nan,-1111:np.nan,-999:np.nan})
mean_matrix['applymonth']=x['applymonth']
month_meanvalues=mean_matrix.groupby('applymonth')[final_choose_cols_fraud].mean().T
month_meanvalues=month_meanvalues.reset_index().rename(columns={'index':'var_name'})
meanvalues=pd.merge(month_meanvalues,mean_values,on='var_name',how='left')
meanvalues.to_excel('meanvalues_fraud_20190716.xlsx',index=False)

## 带走

CarLoanData_BringToTounawang=combine_data
