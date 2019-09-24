# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:59:34 2019

@author: wj56740
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:02:54 2019

@author: wj56740
#针对客户：所有新增客户
#观察期时间窗：2017.07.01-2018.09.28
#表现期：2019年4月15日
## 坏客户定义：
#      1)  曾经逾期16+
## 好客户的定义： 
#      1） 已结清且未发生逾期   
#      2） 已还至少6期且未发生逾期


"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datetime
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.externals import joblib
import pickle
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)
from c_tools  import v2_cat_woe_iv,v2_equif_bin
from c_tools import ChiMerge,AssignBin, CalcWOE, BinBadRate
from itertools import *
from sklearn.linear_model import LogisticRegression
from  sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from scipy import stats
from sklearn.externals import joblib
from c_tools import Vif,score_situation,score_situation_test,auc_ruce
import statsmodels.api as sm
import time
import gc



warnings.filterwarnings("ignore")
rcParams['font.sans-serif'] = ['Microsoft YaHei']

## 修改路径到数据存储的文件夹
os.chdir('D:/llm/联合建模算法开发/数据/')


## 定义一些函数用于模型评估，计算KS
def r_p(y_test, answer_p, idx, low=0, high=150):
    a = answer_p
    b = y_test

    idx = idx[low:high]
    recall = (b.iloc[idx] == 1).sum() / (b == 1).sum()
    precision = (b.iloc[idx] == 1).sum() / (high - low)
    return (recall, precision)


def r_p_chart(y_test, answer_p, part=20):
    print('分段  平均分 最小分 最大分 客户数 逾期数 逾期数/客户数 KS值 GAIN_INDEX 累计召回率')
    a = answer_p
    b = y_test

    idx = sorted(range(len(a)), key=lambda k: a[k], reverse=True)

    total_label_RATE = (y_test == 1).sum() / len(y_test)

    ths = []
    cum = 0
    if len(np.unique(a)) < part:
        for unq_a in np.unique(a)[::-1]:
            ths.append(cum)
            cum = cum + (a == unq_a).sum()
        ths.append(cum)

    else:
        for i in np.arange(0, len(a), (len(a) / part)):
            ths.append(int(round(i)))
        ths.append(len(a))

    min_scores = []
    for idx_ths, _ in enumerate(ths):
        if idx_ths == 0:
            continue
        # idx_ths = 1
        low = ths[idx_ths - 1]
        high = ths[idx_ths]

        r, p = r_p(y_test, answer_p, idx, low, high)
        cum_r, cum_p = r_p(y_test, answer_p, idx, 0, high)

        max_score = answer_p[idx[low]]
        min_score = answer_p[idx[high - 1]]
        min_scores.append(min_score)
        mean_score = (max_score + min_score) / 2
        len_ = high - low
        idx_tmp = idx[low:high]
        bad_num = (b.iloc[idx_tmp] == 1).sum()
        INTERVAL_label_RATE = bad_num / len_
        idx_tmp = idx[0:high]

        tpr = (b.iloc[idx_tmp] == 1).sum() / (b == 1).sum()
        fpr = (b.iloc[idx_tmp] == 0).sum() / (b == 0).sum()
        ks = tpr - fpr
        gain_index = INTERVAL_label_RATE / total_label_RATE

        print('%d %10.3f %10.3f %10.3f %7d %7d %10.2f %10.2f %10.2f %10.2f'
              % (idx_ths, mean_score * 100, min_score * 100, max_score * 100, len_, bad_num, INTERVAL_label_RATE * 100,
                 ks * 100, gain_index, cum_r * 100))

    return min_scores


def r_p_chart2(y_test, answer_p, min_scores, part=20):
    print('分段  平均分 最小分 最大分 客户数 逾期数 逾期数/客户数 KS值 GAIN_INDEX 累计召回率')
    a = answer_p
    b = y_test

    idx = sorted(range(len(a)), key=lambda k: a[k], reverse=True)

    ths = []
    ths.append(0)
    min_scores_idx = 0
    for num, i in enumerate(idx):
        # print(a[i])
        if a[i] < min_scores[min_scores_idx]:
            ths.append(num)
            min_scores_idx = min_scores_idx + 1
    ths.append(len(idx))

    total_label_RATE = (y_test == 1).sum() / len(y_test)

    min_scores = []
    for idx_ths, _ in enumerate(ths):
        if idx_ths == 0:
            continue
        low = ths[idx_ths - 1]
        high = ths[idx_ths]

        r, p = r_p(y_test, answer_p, idx, low, high)
        cum_r, cum_p = r_p(y_test, answer_p, idx, 0, high)

        max_score = answer_p[idx[low]]
        min_score = answer_p[idx[high - 1]]

        min_scores.append(min_score)
        mean_score = (max_score + min_score) / 2
        len_ = high - low
        idx_tmp = idx[low:high]
        bad_num = (b.iloc[idx_tmp] == 1).sum()
        INTERVAL_label_RATE = bad_num / len_
        idx_tmp = idx[0:high]
        tpr = (b.iloc[idx_tmp] == 1).sum() / (b == 1).sum()
        fpr = (b.iloc[idx_tmp] == 0).sum() / (b == 0).sum()
        ks = tpr - fpr
        gain_index = INTERVAL_label_RATE / total_label_RATE

        print('%d %10.3f %10.3f %10.3f %7d %7d %10.2f %10.2f %10.2f %10.2f'
              % (idx_ths, mean_score * 100, min_score * 100, max_score * 100, len_, bad_num, INTERVAL_label_RATE * 100,
                 ks * 100, gain_index, cum_r * 100))
        
        

'''
##2.2 变量预处理--针对不同的数据类型进行预处理
'''

final_data=pd.read_excel('所有建模样本数据_20190802.xlsx')
touna_cd_score_for=pd.read_csv('touna_cd_score_for.csv')
touna_cd_score_for['applydate']=touna_cd_score_for['apply_date'].str.slice(0,10)
final_data=pd.merge(final_data,touna_cd_score_for,on=['credentials_no_md5', 'cust_name_md5', 'mobile_md5', 'applydate'],how='inner')
final_data=final_data.drop_duplicates(['credentials_no_md5', 'cust_name_md5', 'mobile_md5', 'applydate'])
columns_transform={'通用分':'rational_score','小额现金贷多期分':'small_creditmoney_multiplyterm_score','小额现金贷单期分':'small_creditmoney_singleterm_score',
                   '银行分':'bank_score','消费金融分':'consumerloan_score','大额现金贷分':'big_creditmoney_singleterm_score'}
final_data=final_data.rename(columns=columns_transform)

vars_count_table=pd.read_excel('D:/llm/联合建模算法开发/逻辑回归结果/de_dict_vars_20190722.xlsx')
choose_columns_table = vars_count_table[vars_count_table['是否选用'].isin(['是'])]
numeric_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['\ufeff数据型','数据型', '数字型']), 'var_name'].values.tolist()
str_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['字符型', '\ufeff字符型','字符型 ' ]), 'var_name'].values.tolist()
date_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['\ufeff日期型','日期型']), 'var_name'].values.tolist()
final_data=final_data.replace('\\N',np.nan)



'''
## 4.变量筛选
'''


"""
## 4.1 初步筛选
等频分箱：
类别大于10的变量进行等频分箱，然后计算IV
类别小于10 的变量直接进行计算IV
"""
model_data_new=final_data.loc[final_data.app_applydate<='2018-10-31 23:59:59',choose_columns_table.var_name.tolist()+
                               ['apply_y','rational_score','small_creditmoney_multiplyterm_score','small_creditmoney_singleterm_score',
                                'bank_score','consumerloan_score','big_creditmoney_singleterm_score','app_applydate']].copy()

'''
##  处理日期型变量，将日期变量转为距离申请日期的天数
'''
for col in date_columns:  # 去除异常的时间
    try:
        model_data_new.loc[model_data_new[col] >= '2030-01-01', col] = np.nan
    except:
        pass


def date_cal(x, app_applydate):  # 计算申请日期距离其他日期的天数
    days_dt = pd.to_datetime(app_applydate) - pd.to_datetime(x)
    return days_dt.dt.days


for col in date_columns:
    if col != 'app_applydate':
        try:
            if col not in ['vehicle_minput_drivinglicensevaliditydate']:
                model_data_new[col] = date_cal(model_data_new[col], model_data_new['app_applydate'])
            else:
                model_data_new[col] = date_cal(model_data_new['app_applydate'], model_data_new[col])  # 计算行驶证有效期限距离申请日期的天数
        except:
            pass

#x=model_data_new.drop('apply_y',axis=1)
#y=model_data_new['apply_y']
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
#test_data=pd.concat([x_test,y_test],axis=1)
#
#df = pd.concat([x_train,y_train],axis=1)
df=model_data_new.copy()
for cn in df.columns:
    try:
        df[cn]=df[cn].astype('float64')
        df[cn]=df[cn].fillna(-99)
    except:
        df[cn]=df[cn].fillna('-99')
        df[cn]=df[cn].astype('str')
        
        
def main(df):
    cols = df.columns.tolist()
    col_removed = ['apply_y']
    cols = list(set(cols)-set(col_removed))
    cat_over20 = []
    cat_less20 = []
    for col in cols:
        if len(df[col].value_counts())> 10:
            cat_over20.append(col)
        else:
            cat_less20.append(col)
    print("类别》10 的变量个数是：",len(cat_over20))
    print("类别《10的变量个数是：",len(cat_less20))

    target = "apply_y"
#    cat_less20 = ["deci_apt_facetrial_housetype","deci_cont02_relationship","deci_jxl_contact2_rel","deci_vehicle_minput_lastmortgagerinfo"]
    high_iv_name = v2_cat_woe_iv(df,cat_less20,target) # 计算woe 和IV值
    group = []
    for i in high_iv_name.index.tolist():
        group.append(len(df[i].value_counts()))
    less_iv_name = pd.DataFrame({"var_name":high_iv_name.index,"IV":high_iv_name.values,"group":group})
    print(less_iv_name)
    less_iv_name.to_excel("D:/llm/联合建模算法开发/逻辑回归结果/cat_less_iv_1.xlsx")
    df = v2_equif_bin(df, cat_over20, target)
    cols_name = [i+"_Bin" for i in cat_over20]
    over_iv_name = v2_cat_woe_iv(df, cols_name, target)  # 计算woe 和IV值
    over_iv_name.to_excel("D:/llm/联合建模算法开发/逻辑回归结果/cat_more_iv_2.xlsx")
    iv_name=less_iv_name.var_name.tolist()+[cn.split('_Bin')[0] for cn in over_iv_name.index]
    return iv_name

if __name__ == '__main__':
    iv_name=main(df)


"""
## 4.2 对连续型变量用卡方分箱
卡方分箱：
max_bin = 5
分箱后需要手动调节：badrate
"""
iv_st_name=[cn.split('_Bin')[0] for cn in iv_name]
init_choose_name_desc=choose_columns_table.set_index('var_name').loc[iv_st_name,:].reset_index()
#init_choose_name_desc.to_excel('D:/llm/联合建模算法开发/逻辑回归结果/init_choose_name_desc_20190805.xlsx',index=False)
init_choose_table=init_choose_name_desc.loc[init_choose_name_desc['是否选用'].isin(['是']),:]#手动去除一些不能用的变量
numeric_columns = init_choose_table.loc[init_choose_table.var_type.isin(['\ufeff数据型','数据型', '数字型','数值型']), 'var_name'].values.tolist()
str_columns = init_choose_table.loc[init_choose_table.var_type.isin(['字符型', '\ufeff字符型','字符型 ' ]), 'var_name'].values.tolist()
date_columns = init_choose_table.loc[init_choose_table.var_type.isin(['\ufeff日期型','日期型']), 'var_name'].values.tolist()
print(len(numeric_columns)+len(str_columns)+len(date_columns))

df_choose1=df[init_choose_table.var_name.tolist()+['apply_y']].copy()
df_choose1.to_csv('D:/llm/联合建模算法开发/逻辑回归结果/df_choose1.csv',index=False)
target ="apply_y"
max_interval = 5
num_var=numeric_columns+date_columns


def v3_Chimerge(df,target,max_interval,num_var):
    num_list=[]
    for col in num_var:
        print("{}开始进行分箱".format(col))
        if -99 not in set(df[col]):
            print("{}没有特殊取值".format(col))
            cutOff = ChiMerge(df, col, target, max_interval=max_interval, special_attribute=[])
            if len(cutOff)!=0:
               df[col + '_Bin'] = df[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
               dicts, regroup = BinBadRate(df, col + "_Bin", target, grantRateIndicator=0)
               regroup.columns = ["group", "total", "bad", "bad_rate"]
               regroup["var_name"] = col+"_Bin"
               regroup = regroup.sort_values(by="group")
               regroup.to_csv("D:/llm/联合建模算法开发/逻辑回归结果/regroup_Chi_cutbin__.csv", mode="a", header=True)
               print(regroup)
               num_list.append(col)

        else:
            max_interval = 5
            cutOff = ChiMerge(df, col, target, max_interval=max_interval, special_attribute=[-99])
            if len(cutOff)!=0:
               df[col + '_Bin'] = df[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-99]))
               dicts, regroup = BinBadRate(df, col + "_Bin", target, grantRateIndicator=0)
               regroup.columns = ["group", "total", "bad", "bad_rate"]
               regroup["var_name"] = col+"_Bin"
               regroup = regroup.sort_values(by="group")
               regroup.to_csv("D:/llm/联合建模算法开发/逻辑回归结果/regroup_Chi_cutbin__.csv", mode="a", header=True)
               print(regroup)
               num_list.append(col)
    return df,num_list
df_choose1,num_list = v3_Chimerge(df_choose1,target,max_interval, num_var)
df_choose1.to_csv("D:/llm/联合建模算法开发/逻辑回归结果/cut_bin_train.csv")
print(df_choose1.columns.tolist())
print(df_choose1.shape[0])


# 计算woe 和IV
cols_list = [i+"_Bin" for i in num_list]
df_hig_iv = v2_cat_woe_iv(df_choose1,cols_list,target)
df_hig_iv.to_excel("D:/llm/联合建模算法开发/逻辑回归结果/high_iv_chi.xlsx")

# 经卡方分箱完筛选后的最终变量
iv_name1=[cn.split('_Bin')[0] for cn in df_hig_iv.index.tolist()]+str_columns
init_choose_name_desc=choose_columns_table.set_index('var_name').loc[iv_name1,:].reset_index()
#init_choose_name_desc.to_excel('D:/llm/联合建模算法开发/逻辑回归结果/init_choose_name_desc_20190805v1.xlsx',index=False)


"""
## 4.3 手动调整分箱,依据相关性进一步筛选变量
手动调节不满足单调性的变量
使其满足单调性
并计算IV
转换为woe值
时间外样本转换为woe值
连续变量的相关性分析
"""
init_choose_name_desc=pd.read_excel('D:/llm/联合建模算法开发/逻辑回归结果/init_choose_name_desc_20190805v1.xlsx')
init_choose_table=init_choose_name_desc.loc[init_choose_name_desc['是否选用'].isin(['是']),:]#手动去除一些不能用的变量
numeric_columns = init_choose_table.loc[init_choose_table.var_type.isin(['\ufeff数据型','数据型', '数字型','数值型']), 'var_name'].values.tolist()
str_columns = init_choose_table.loc[init_choose_table.var_type.isin(['字符型', '\ufeff字符型','字符型 ' ]), 'var_name'].values.tolist()
date_columns = init_choose_table.loc[init_choose_table.var_type.isin(['\ufeff日期型','日期型']), 'var_name'].values.tolist()
print(len(numeric_columns)+len(str_columns)+len(date_columns))

target ="apply_y"
col=[i+"_Bin" for i in numeric_columns+date_columns]+str_columns
df_choose2=df_choose1.copy()
def func(x,dict_woe):
    return dict_woe[x]
""""把分箱好的变量转为woe值"""
def main(df,col,address):
    target = "apply_y"
    error_cols=[]
    for var in col:
        try:
          woe_iv,regroup  = CalcWOE(df, var, target)
          bin = regroup.iloc[:, 0].tolist()
          woe_value = regroup.iloc[:, -3].tolist()
          dict_woe = dict(zip(bin, woe_value))
          print(dict_woe)
          df[var] = df[var].map(lambda x: func(x, dict_woe))
          dicts, regroup_badrate = BinBadRate(df, var, target, grantRateIndicator=0)
          regroup_badrate=regroup_badrate.rename(columns={var:var+'_woe'})
          regroup=pd.merge(regroup,regroup_badrate,on=['total','bad'],how='left')
          regroup=regroup[[var,var+'_woe',"total","bad","good",'bad_rate',"bad_pcnt","good_pcnt","WOE","IV","var_name"]]
          regroup=regroup.rename(columns={var:'var_bin'})
          print(regroup)
          regroup.to_csv("D:/llm/联合建模算法开发/逻辑回归结果/num_bin_badrate.csv",mode="a",header=True)
        except:
            error_cols.append(var)
    df.to_csv(address+"data_woe_value_num.csv",header=True) # 包含分类变量和连续变量
    return error_cols,df

error_cols,df_choose3=main(df_choose2,col,'D:/llm/联合建模算法开发/逻辑回归结果/')

""" 相关性分析：大于0.5的变量进行筛选"""
#df_choose3= pd.read_csv("D:/llm/联合建模算法开发/逻辑回归结果/data_woe_value_num.csv")
df_choose4 = df_choose3[col].copy()

def corr_analysis(df):
        cor = df.corr()
        filter=0.5
        # 列出高于阈值的正相关或者负相关的系数矩阵
        cor.loc[:,:] = np.triu(cor, k=1)
        cor = cor.stack()
        high_cors = cor[(cor > filter) | (cor < -filter)]
#        df_cor = pd.DataFrame(cor)
        df_high_cor1 = pd.DataFrame(high_cors)
#        df_cor.to_excel('./data/savefile_v1.xlsx', header=True)
#        df_high_cor1.to_excel('./data/savefile_v2.xlsx', header=True)
        return df,df_high_cor1


df_choose4, df_high_cor1= corr_analysis(df_choose4)
df_high_cor1=df_high_cor1.reset_index()
df_high_cor1.columns=['var_name1','var_name2','corref']
df_high_cor1.to_excel("D:/llm/联合建模算法开发/逻辑回归结果/df_high_cor.xlsx")


## 相关性去除iv更低的变量
cat_less_iv=pd.read_excel('D:/llm/联合建模算法开发/逻辑回归结果/cat_less_iv_1.xlsx')
cat_less_iv=cat_less_iv.set_index('var_name').loc[str_columns,'IV'].reset_index().rename(columns={'IV':'iv'})
df_hig_iv=pd.read_excel("D:/llm/联合建模算法开发/逻辑回归结果/high_iv_chi.xlsx")
df_hig_iv=pd.concat([df_hig_iv,cat_less_iv],axis=0)
df_hig_iv.to_csv('D:/llm/联合建模算法开发/逻辑回归结果/df_hig_iv_all.csv',index=False)
df_high_cor1=pd.read_excel("D:/llm/联合建模算法开发/逻辑回归结果/df_high_cor.xlsx")

del_cor=[]
for i in range(len(df_high_cor1)):
    if df_hig_iv.loc[df_hig_iv.var_name.isin([df_high_cor1.var_name1.values[i]]),'iv'].values[0]>=\
                    df_hig_iv.loc[df_hig_iv.var_name.isin([df_high_cor1.var_name2.values[i]]),'iv'].values[0]:
            del_cor.append(df_high_cor1.var_name2.values[i])
del_cor=list(set(del_cor))
remain_col=[cn for cn in col if cn not in del_cor] ##剔除一些最好不能用的变量
remain_col.remove('apt_ec_currentoverdue')
remain_col.remove('tx_cardid')

#剩72个

## 经相关性后去除的分箱情况
target ="apply_y"
df_choose5=df_choose1.copy()

""""把分箱好的变量转为woe值"""
def main(df,col,address):
    target = "apply_y"
    for var in col:
        woe_iv,regroup  = CalcWOE(df, var, target)
        bin = regroup.iloc[:, 0].tolist()
        woe_value = regroup.iloc[:, -3].tolist()
        dict_woe = dict(zip(bin, woe_value))
        print(dict_woe)
        df[var] = df[var].map(lambda x: func(x, dict_woe))
        dicts, regroup_badrate = BinBadRate(df, var, target, grantRateIndicator=0)
        regroup_badrate=regroup_badrate.rename(columns={var:var+'_woe'})
        regroup=pd.merge(regroup,regroup_badrate,on=['total','bad'],how='left')
        regroup=regroup[[var,var+'_woe',"total","bad","good",'bad_rate',"bad_pcnt","good_pcnt","WOE","IV","var_name"]]
        regroup=regroup.rename(columns={var:'var_bin'})
        print(regroup)
        regroup.to_csv(address+".csv",mode="a",header=True)

main(df_choose5,remain_col,"D:/llm/联合建模算法开发/逻辑回归结果/final_test_num_bin_badrate1")


"""
## 4.4 进行交叉验证,进一步筛选变量
分箱之后：
调整单调性
调整分箱稳定性
"""

df_choose7=df_choose5[remain_col+['apply_y']].copy()
for cn in df_choose7.columns:
    print(cn,df_choose7[cn].unique())
remain_col.remove('m12_apply_platform_cnt_Bin')
remain_col.remove('vehicle_evtrpt_mileage_Bin')
remain_col.remove('jxl_120_record')
remain_col.remove('jxl_id_operator')
remain_col.remove('d1_id_relate_device_num_Bin')
remain_col.remove('contact_court_cnt_Bin')

x =  df_choose7[remain_col]
y = df["apply_y"]
rf = RandomForestClassifier()
rf.fit(x,y)
print(pd.DataFrame({"var_name": remain_col, "importance_": rf.feature_importances_}).sort_values("importance_",ascending=False))


"""
 交叉验证,挑选剔除后使模型性能下降的变量
"""
def cross_val(df,finally_columns_name):
    init_score=1
    del_cols=[]
    col = finally_columns_name.copy()
    for i in finally_columns_name:
        col.remove(i)
        if len(col)>=1:
           print(i)
           x = df[col]
           y = df["apply_y"]
           lr = LogisticRegression()
           scores = cross_val_score(lr,x,y,cv=10,scoring="roc_auc")
           score = scores.mean()
           if score>init_score:
               del_cols.append(i)
               break
           init_score=score
           print(score)
    return del_cols

## 迭代，使得不在剔除变量为止为停止准则，剩余48个变量
i=0
temp1=remain_col
while i<=100:
    del_cols=cross_val(df_choose7,remain_col)
    remain_col=[cn for cn in remain_col if cn not in del_cols ]
    if len(del_cols)==0:
        break
    print(del_cols)
    i=i+1
    
## 利用lasso回归继续变量筛选
def lasscv(df,finally_columns_name):
    col = []
    init_score=0
    del_cols=[]
    for i in finally_columns_name:
        col.append(i)
        x = df[col]
        x = np.matrix(x)
        y = df["apply_y"]
        la = LassoCV( cv=10)
        la.fit(x,y)
        print(i)
        print(la.score(x,y))
        score=la.score(x,y)
        if score<init_score:
            del_cols.append(i)
            break
        init_score=score
           
    return del_cols
        
## 迭代，使得不在增加变量为止为停止准则
remain_col=['accu_loan_amt_Bin',
 'agreement_month_repay_Bin',
 'apt_ec_overduephasetotallastyear_Bin',
 'avg_sms_cnt_l6m_Bin',
 'big_creditmoney_singleterm_score_Bin',
 'consumerloan_score_Bin',
 'contact_bank_call_cnt_Bin',
 'contact_car_contact_afternoon_Bin',
 'contact_unknown_contact_early_morning_Bin',
 'contacts_class1_cnt_Bin',
 'i_cnt_grp_partner_loan_all_all_Bin',
 'i_cnt_mobile_all_all_180day_Bin',
 'i_cv_cnt_30daypartner_all_all_360day_Bin',
 'i_freq_record_loan_thirdservice_365day_Bin',
 'i_max_length_record_loan_p2pweb_365day_Bin',
 'i_mean_freq_node_seq_partner_loan_all_all_Bin',
 'i_mean_length_event_loan_imbank_30day_Bin',
 'i_pctl_cnt_ic_partner_loan_insurance_60day_Bin',
 'i_pctl_cnt_ic_partner_loan_p2pweb_365day_Bin',
 'i_ratio_freq_day_loan_imbank_365day_Bin',
 'i_ratio_freq_day_login_p2pweb_365day_Bin',
 'i_ratio_freq_record_loan_consumerfinance_365day_Bin',
 'i_ratio_freq_record_loan_offloan_180day_Bin',
 'jxl_id_comb_othertel_num_Bin',
 'max_call_in_cnt_l6m_Bin',
 'max_overdue_amount_Bin',
 'max_overdue_terms_Bin',
 'max_total_amount_l6m_Bin',
 'other_org_count_Bin',
 'phone_used_time_Bin',
 'qtorg_query_orgcnt_Bin',
 'rational_score_Bin',
 'times_by_current_org_Bin',
 'vehicle_evtrpt_b2bprice_Bin',
 'cell_reg_time_Bin',
 'email_info_date_Bin',
 'prof_title_info_date_Bin',
 'vehicle_minput_lastreleasedate_Bin',
 'apt_lastloanmode',
 'vehicle_minput_lastmortgagerinfo',
 'high_acade_qua']

##lasso迭代，筛选变量,剩余38个
i=0
del_cols=[]
temp=remain_col
while i<=100:
    del_cols=lasscv(df_choose7,remain_col)
    if len(del_cols)==0:
        break
    remain_col=set(remain_col)-set(del_cols)
    remain_col=list(remain_col)
    print(del_cols)
    i=i+1

## 共线性和p值去除变量,剩余38个
df_choose6=df_choose1[remain_col+['apply_y']].copy()
main(df_choose6,remain_col,"D:/llm/联合建模算法开发/逻辑回归结果/final_test_num_bin_badrate2") 

while 1:
   df_choose7=df_choose6[remain_col+['apply_y']].copy()
   x =  df_choose7[remain_col]
   y = df_choose7["apply_y"]

   lgt=sm.Logit(y,x[remain_col])
   result=lgt.fit()
   print( result.summary2())
   p_value=pd.DataFrame(result.pvalues).reset_index().rename(columns={'index':'var_name',0:'p_value'})
   p_value_max=p_value['p_value'].max()
   p_value_max_cols=p_value.loc[p_value['p_value']>=p_value_max,'var_name'].values[0]
   print(p_value_max_cols)
   if p_value_max<=0.05:
       break
   remain_col.remove(p_value_max_cols)

while 1:
  max_vif ,vif_df= Vif(df_choose7, remain_col)
  print(vif_df)
  if max_vif['vif'].values[0]<=3:
      break
  remain_col.remove(max_vif['var_name'].values[0])
vif_df.to_excel('D:/llm/联合建模算法开发/逻辑回归结果/vif_df_20190729.xlsx')


df_choose7=df_choose6[remain_col+['apply_y']].copy()
x =  df_choose7[remain_col]
y = df_choose7["apply_y"]
x_train,x_test,y_train,y_test=train_test_split(x,y)
lr=LogisticRegression(penalty='l2',C=1,n_jobs=-1,verbose=0,random_state=0)
lr.fit(x_train,y_train)
columns_coef=pd.DataFrame(lr.coef_[0].tolist(),index=remain_col).reset_index().rename(columns={'index':'var_name',0:'coef'})
final_columns=columns_coef.loc[columns_coef['coef']>0,'var_name'].values.tolist()
while 1:
   lgt=sm.Logit(y_train,x_train[final_columns])
   result=lgt.fit()
   print( result.summary2())
   p_value=pd.DataFrame(result.pvalues).reset_index().rename(columns={'index':'var_name',0:'p_value'})
   p_value_max=p_value['p_value'].max()
   p_value_max_cols=p_value.loc[p_value['p_value']>=p_value_max,'var_name'].values[0]
   print(p_value_max_cols)
   if p_value_max<=0.05:
       break
   final_columns.remove(p_value_max_cols)

df_hig_iv = v2_cat_woe_iv(df_choose7,final_columns,'apply_y')
df_hig_iv=df_hig_iv.reset_index().rename(columns={'index':'var_name',0.0:'iv'}).sort_values('iv',ascending=False)

## 相关性去除
final_columns.remove('consumerloan_score_Bin')
final_columns.remove('big_creditmoney_singleterm_score_Bin')
final_columns.remove('phone_used_time_Bin')

##最终剩下的变量
final_columns=['vehicle_minput_lastmortgagerinfo',
 'i_mean_freq_node_seq_partner_loan_all_all_Bin',
 'rational_score_Bin',
 'contact_unknown_contact_early_morning_Bin',
 'vehicle_minput_lastreleasedate_Bin',
 'i_pctl_cnt_ic_partner_loan_insurance_60day_Bin',
 'cell_reg_time_Bin',
 'i_cnt_grp_partner_loan_all_all_Bin',
 'i_ratio_freq_record_loan_offloan_180day_Bin',
 'prof_title_info_date_Bin',
 'apt_ec_overduephasetotallastyear_Bin',
 'i_cnt_mobile_all_all_180day_Bin',
 'contact_bank_call_cnt_Bin',
 'apt_lastloanmode',
 'max_total_amount_l6m_Bin',
 'contact_car_contact_afternoon_Bin',
 'i_freq_record_loan_thirdservice_365day_Bin',
 'vehicle_evtrpt_b2bprice_Bin',
 'avg_sms_cnt_l6m_Bin',
 'times_by_current_org_Bin',
 'max_call_in_cnt_l6m_Bin',
 'max_overdue_terms_Bin']



"""
## 
分箱之后：
调整单调性
调整分箱稳定性
"""
df_choose6=df_choose1[final_columns+['apply_y']].copy()
df_choose6['vehicle_minput_lastmortgagerinfo']=df_choose6['vehicle_minput_lastmortgagerinfo'].replace({'-99':'NA&3,4'})

main(df_choose6,final_columns,"D:/llm/联合建模算法开发/逻辑回归结果/final_test_num_bin_badrate3")  
df_hig_iv = v2_cat_woe_iv(df_choose6,final_columns,'apply_y')
df_hig_iv=df_hig_iv.reset_index().rename(columns={'index':'var_name',0.0:'iv'})
choose_column=df_hig_iv.loc[df_hig_iv.iv>=0.02,'var_name'].values.tolist()


## 训练模型
df_choose7=df_choose6[choose_column+['apply_y']].copy()
#df_choose7.to_excel('D:/llm/联合建模算法开发/逻辑回归结果/训练集数据_20190806.xlsx')
x =  df_choose7[choose_column]
y = df_choose7["apply_y"]
x_train,x_test,y_train,y_test=train_test_split(x,y)
lgt=sm.Logit(y_train,x_train)
result=lgt.fit()
print( result.summary2())
choose_column.remove('apt_ec_overduephasetotallastyear_Bin')
choose_column.remove('apt_lastloanmode')

lr=LogisticRegression(penalty='l2',C=1,n_jobs=-1,verbose=0,random_state=0)
lr.fit(x_train[choose_column],y_train)
score=lr.predict_proba(x_train[choose_column])[:,1]
min_scores = r_p_chart(y_train, score, part=20)
min_scores = [round(i, 5) for i in min_scores]
min_scores[19] = 0
cuts = [round(min_scores[i] * 100.0, 3) for i in range(20)[::-1]] + [100.0]
joblib.dump(lr,'D:/llm/联合建模算法开发/逻辑回归结果/lr_alldata_20190806.pkl')
columns_coef=pd.DataFrame(lr.intercept_.tolist()+lr.coef_[0].tolist(),index=['intercept']+choose_column).reset_index().rename(columns={'index':'var_name',0:'coef'})
columns_coef.to_excel('D:/llm/联合建模算法开发/逻辑回归结果/columns_coef_20190806.xlsx',index=False)


print('训练集')
pred_p = lr.predict_proba(x_train[choose_column])[:, 1]
fpr, tpr, th = roc_curve(y_train, pred_p)
ks = tpr - fpr
print('train ks: ' + str(max(ks)))
print(roc_auc_score(y_train, pred_p))
r_p_chart2(y_train, pred_p, min_scores, part=20)

print('测试集')
pred_p2 = lr.predict_proba(x_test[choose_column])[:, 1]
fpr, tpr, th = roc_curve(y_test, pred_p2)
ks2 = tpr - fpr
print('test ks:  ' + str(max(ks2)))
print(roc_auc_score(y_test, pred_p2))
r_p_chart2(y_test, pred_p2, min_scores, part=20)

print('建模全集')
pred_p3 = lr.predict_proba(x[choose_column])[:, 1]
fpr, tpr, th = roc_curve(y, pred_p3)
ks3 = tpr - fpr
print('all ks:   ' + str(max(ks3)))
print(roc_auc_score(y, pred_p3))
r_p_chart2(y, pred_p3, min_scores, part=20)


## 入模变量分箱choose_column

def avg_sms_cnt_l6m(x):
    if x == -99:
        return  0.0463
    elif x <  5.833333333333332:
        return  -0.2544
    elif x < 21.166666666666668:
        return  -0.1303
    elif x < 68.66666666666667:
        return  0.2065
    elif x>= 68.66666666666667:
        return  0.0893

def cell_reg_time(x):
    if x == -99:
        return  0.0359
    elif x <  923.0:
        return  0.3336
    elif x< 3958.0:
        return  0.0005
    elif x < 5959.0:
        return  -0.353
    elif x>= 5959.0:
        return  -0.7457

def contact_bank_call_cnt(x):
    if x == -99:
        return 0.0463
    elif x <  7.0:
        return -0.1887
    elif x< 21.0:
        return  0.1478
    elif x < 36.0:
        return  0.2769
    elif x>= 36.0:
        return  0.5728

def contact_car_contact_afternoon(x):
    if x == -99:
        return  0.0463
    elif x < 5.0:
        return  -0.089
    elif x< 11.0:
        return   0.2716
    elif x>= 11.0:
        return  0.3853

def contact_unknown_contact_early_morning(x):
    if x == -99:
        return   0.0463
    elif x <  45.0:
        return  -0.1839
    elif x <  143.0:
        return  0.2056
    elif x< 253.0:
        return  0.3757
    elif x>= 253.0:
        return  0.4198

def i_cnt_grp_partner_loan_all_all(x):
    if x == -99:
        return  -1.6913
    elif x < 1.0:
        return   0.2569
    elif x< 3.0:
        return  -0.9927
    elif x< 63.0:
        return   -0.2717
    elif x>=  63.0:
        return   0.1975

def i_cnt_mobile_all_all_180day(x):
    if x == -99:
        return  -1.6913
    elif x< 2.0:
        return  -0.2916
    elif x< 3.0:
        return   0.3239
    elif x>= 3.0:
        return   0.789

def i_freq_record_loan_thirdservice_365day(x):
    if x == -99:
        return   -1.6913
    elif x <  1.0:
        return   -0.0709
    elif x< 2.0:
        return    0.6482
    elif x>= 2.0:
        return    1.0176

def i_mean_freq_node_seq_partner_loan_all_all(x):
    if x == -99:
        return   -1.6913
    elif x <  -999.0:
        return   -0.0222
    elif x<  1.3889:
        return    0.104
    elif x<  2.0:
        return    0.1767
    elif x>= 2.0:
        return   -0.326

def i_pctl_cnt_ic_partner_loan_insurance_60day(x):
    if x == -99:
        return  -1.6913
    elif x < 0.7752:
        return  -0.7926
    elif x < 0.9598:
        return   -0.147
    elif x>= 0.9598:
        return   0.4604


def i_ratio_freq_record_loan_offloan_180day(x):
    if x == -99:
        return  -1.6913
    elif x < -999.0:
        return  -0.3641
    elif x< 0.1316:
        return  -0.0927
    elif x< 0.2131:
        return  0.6413
    elif x>= 0.2131:
        return  0.3469

def max_call_in_cnt_l6m(x):
    if x == -99:
        return  0.0463
    elif x < 134.0:
        return  -0.2253
    elif x< 337.0:
        return  -0.1184 
    elif x< 570.0:
        return  0.2266 
    elif x>=570.0:
        return  0.5305


def max_overdue_terms(x):
    if x == -99:
        return -0.2221
    elif x < 1.0:
        return  0.162
    elif x< 2.0:
        return  1.2763 
    elif x>= 2.0:
        return  1.3173 

def max_total_amount_l6m(x):
    if x == -99:
        return   0.0463
    elif x < 110.39:
        return  -0.2687
    elif x< 298.8:
        return  -0.0599 
    elif x< 462.91:
        return   0.2699
    elif x>= 462.91:
        return   0.4125


def prof_title_info_date(x):
    if x == -99:
        return -0.188
    elif x < -29.0:
        return  3.2165
    elif x < -6.0:
        return  2.6882
    elif x < 145.0:
        return  0.3216
    elif x>=145.0:
        return  0.0506

def rational_score(x):
    if x < 424.0:
        return  -0.2805
    elif x< 467.0:
        return   1.0683
    elif x < 492.0:
        return   0.8485
    elif x< 601.0:
        return   0.0829
    elif x>= 601.0:
        return   -0.9137


def times_by_current_org(x):
    if x == -99:
        return   -0.1334
    elif x < 2.0:
        return   -0.0704
    elif x<  3.0:
        return    0.2697
    elif x < 4.0:
        return    0.5512
    elif x>= 4.0:
        return   1.5033

def vehicle_evtrpt_b2bprice(x):
    if x == -99:
        return  0.3172
    elif x < 3.33:
        return  0.3671
    elif x< 4.3:
        return  0.3085
    elif x < 10.6:
        return -0.0114
    elif x>= 10.6:
        return  -0.2644
 
def vehicle_minput_lastmortgagerinfo(x):
    if x == -99:
        return  -0.2394
    elif x==1:
        return   0.4462
    elif x ==2:
        return   0.2031
    elif x ==3:
        return   -0.6617
    elif x ==4:
        return   -0.225
    elif x ==5:
        return   0.2546


def vehicle_minput_lastreleasedate(x):
    if x == -99:
        return   -0.2392
    elif x< 1.0:
        return    0.1718
    elif x <6.0:
        return    0.6836
    elif x < 164.0:
        return    0.0425
    elif x>= 164.0:
        return   -0.2837

   

## 时间外验证进行重新评估
outtime_testdata=pd.read_table('outtime_test_20190806.txt',dtype={'app_applycode':str},sep='\u0001')
outtime_testdata=outtime_testdata.replace('\\N',np.nan)
ylabel_data=pd.read_table('contractno_ylabel_data.txt',sep='\u0001',dtype={'applycode':str})  #车贷申请表
outtime_testdata=pd.merge(outtime_testdata,ylabel_data,left_on=['app_applycode','contractno'],right_on=['applycode','contractno'],how='inner')
del ylabel_data
gc.collect()
outtime_testdata=outtime_testdata[outtime_testdata.apply_y.isin([0,1])].copy()
td_union_data=pd.read_excel('CarLoanData_BringToTounawang.xlsx')
outtime_testdata['applydate']=outtime_testdata['app_applydate'].str.slice(0,10)
outtime_testdata=pd.merge(outtime_testdata,td_union_data,left_on=['credentials_no_md5','cust_name_md5','mobile_md5','applydate'],right_on=
                     ['credentials_no_md5_x','cust_name_md5_x','mobile_md5_x','applydate'],how='inner')
del td_union_data
gc.collect()
outtime_testdata.columns=[cn.lower() for cn in outtime_testdata.columns]

touna_cd_score_for=pd.read_csv('touna_cd_score_for.csv')
touna_cd_score_for['applydate']=touna_cd_score_for['apply_date'].str.slice(0,10)
outtime_testdata=pd.merge(outtime_testdata,touna_cd_score_for,on=['credentials_no_md5', 'cust_name_md5', 'mobile_md5', 'applydate'],how='inner')
outtime_testdata=outtime_testdata.drop_duplicates(['credentials_no_md5', 'cust_name_md5', 'mobile_md5', 'applydate'])
columns_transform={'通用分':'rational_score','小额现金贷多期分':'small_creditmoney_multiplyterm_score','小额现金贷单期分':'small_creditmoney_singleterm_score',
                   '银行分':'bank_score','消费金融分':'consumerloan_score','大额现金贷分':'big_creditmoney_singleterm_score'}
outtime_testdata=outtime_testdata.rename(columns=columns_transform)


de_dict_var = pd.read_excel('de_dict_vars_20190722.xlsx')
for i, _ in de_dict_var.iterrows():
    name = de_dict_var.loc[i, 'var_name']
    default = de_dict_var.loc[i, 'default']
    if default != '""' and name in set(outtime_testdata.columns) and name!='app_applycode':
        try:
            outtime_testdata[name] = outtime_testdata[name].astype('float64')
            if (outtime_testdata[name] == float(default)).sum() > 0:
                outtime_testdata.loc[outtime_testdata[name] == float(default), name] = np.nan
        except:
            pass

    elif default == '""' and name in set(outtime_testdata.columns) and name!='app_applycode':
        try:
            outtime_testdata[name] = outtime_testdata[name].astype('float64')
            if (outtime_testdata[name] == float(-99)).sum() > 0:
                outtime_testdata.loc[outtime_testdata[name] == float(-99), name] = np.nan
            if (outtime_testdata[name] == '-99').sum() > 0:
                outtime_testdata.loc[outtime_testdata[name] == '-99', name] = np.nan
        except:
            pass


for col in ['vehicle_minput_lastreleasedate','prof_title_info_date','cell_reg_time']:  # 去除异常的时间
    try:
        outtime_testdata.loc[outtime_testdata[col] >= '2030-01-01', col] = np.nan
    except:
        pass

def date_cal(x, app_applydate):  # 计算申请日期距离其他日期的天数
    days_dt = pd.to_datetime(app_applydate) - pd.to_datetime(x)
    return days_dt.dt.days

for col in ['prof_title_info_date','vehicle_minput_lastreleasedate','cell_reg_time']:
    if col != 'app_applydate':
        try:
            if col not in ['vehicle_minput_drivinglicensevaliditydate']:
                outtime_testdata[col] = date_cal(outtime_testdata[col], outtime_testdata['app_applydate'])
            else:
                outtime_testdata[col] = date_cal(outtime_testdata['app_applydate'], outtime_testdata[col])  # 计算行驶证有效期限距离申请日期的天数
        except:
            pass


outtime_testdata=outtime_testdata.fillna(-99)
outtime_testdata["avg_sms_cnt_l6m"+'_Bin'] = outtime_testdata["avg_sms_cnt_l6m"].map(lambda x: avg_sms_cnt_l6m(x))
outtime_testdata["cell_reg_time"+'_Bin'] = outtime_testdata["cell_reg_time"].map(lambda x: cell_reg_time(x))
outtime_testdata["contact_bank_call_cnt"+'_Bin'] = outtime_testdata["contact_bank_call_cnt"].map(lambda x: contact_bank_call_cnt(x))
outtime_testdata["contact_car_contact_afternoon"+'_Bin'] = outtime_testdata["contact_car_contact_afternoon"].map(lambda x: contact_car_contact_afternoon(x))
outtime_testdata["contact_unknown_contact_early_morning"+'_Bin'] = outtime_testdata["contact_unknown_contact_early_morning"].map(lambda x: contact_unknown_contact_early_morning(x))
outtime_testdata["i_cnt_grp_partner_loan_all_all"+'_Bin'] = outtime_testdata["i_cnt_grp_partner_loan_all_all"].map(lambda x: i_cnt_grp_partner_loan_all_all(x))
outtime_testdata["i_cnt_mobile_all_all_180day"+'_Bin'] = outtime_testdata["i_cnt_mobile_all_all_180day"].map(lambda x: i_cnt_mobile_all_all_180day(x))
outtime_testdata["i_freq_record_loan_thirdservice_365day"+'_Bin'] = outtime_testdata["i_freq_record_loan_thirdservice_365day"].map(lambda x: i_freq_record_loan_thirdservice_365day(x))
outtime_testdata["i_mean_freq_node_seq_partner_loan_all_all"+'_Bin'] = outtime_testdata["i_mean_freq_node_seq_partner_loan_all_all"].map(lambda x: i_mean_freq_node_seq_partner_loan_all_all(x))
outtime_testdata["i_pctl_cnt_ic_partner_loan_insurance_60day"+'_Bin'] = outtime_testdata["i_pctl_cnt_ic_partner_loan_insurance_60day"].map(lambda x: i_pctl_cnt_ic_partner_loan_insurance_60day(x))
outtime_testdata["i_ratio_freq_record_loan_offloan_180day"+'_Bin'] = outtime_testdata["i_ratio_freq_record_loan_offloan_180day"].map(lambda x: i_ratio_freq_record_loan_offloan_180day(x))
outtime_testdata["max_call_in_cnt_l6m"+'_Bin'] = outtime_testdata["max_call_in_cnt_l6m"].map(lambda x: max_call_in_cnt_l6m(x))
outtime_testdata["max_overdue_terms"+'_Bin'] = outtime_testdata["max_overdue_terms"].map(lambda x: max_overdue_terms(x))
outtime_testdata["max_total_amount_l6m"+'_Bin'] = outtime_testdata["max_total_amount_l6m"].map(lambda x: max_total_amount_l6m(x))
outtime_testdata["prof_title_info_date"+'_Bin'] = outtime_testdata["prof_title_info_date"].map(lambda x: prof_title_info_date(x))
outtime_testdata["rational_score"+'_Bin'] = outtime_testdata["rational_score"].map(lambda x: rational_score(x))
outtime_testdata["times_by_current_org"+'_Bin'] = outtime_testdata["times_by_current_org"].map(lambda x: times_by_current_org(x))
outtime_testdata["vehicle_evtrpt_b2bprice"+'_Bin'] = outtime_testdata["vehicle_evtrpt_b2bprice"].map(lambda x: vehicle_evtrpt_b2bprice(x))
outtime_testdata["vehicle_minput_lastmortgagerinfo"] = outtime_testdata["vehicle_minput_lastmortgagerinfo"].map(lambda x: vehicle_minput_lastmortgagerinfo(x))
outtime_testdata["vehicle_minput_lastreleasedate"+'_Bin'] = outtime_testdata["vehicle_minput_lastreleasedate"].map(lambda x: vehicle_minput_lastreleasedate(x))


for cn in choose_column:
    print(cn,outtime_testdata[cn].unique())


print('时间外验证')
pred_p4= lr.predict_proba(outtime_testdata[choose_column])[:, 1]
fpr, tpr, th = roc_curve(outtime_testdata.apply_y, pred_p4)
ks3 = tpr - fpr
print('all ks:   ' + str(max(ks3)))
print(roc_auc_score(outtime_testdata.apply_y, pred_p4))
r_p_chart2(outtime_testdata.apply_y, pred_p4, min_scores, part=20)


## 入模变量稳定性评估

df_choose7['app_applydate']=model_data_new['app_applydate']
full_data=pd.concat([df_choose7[choose_column+['apply_y','app_applydate']],outtime_testdata[choose_column+['apply_y','app_applydate']]])
full_data['applymonth']=full_data['app_applydate'].str.slice(0,7)
full_data=full_data[full_data['applymonth']!='2018-12']
#full_data.to_csv('D:/llm/联合建模算法开发/逻辑回归结果/full_data_20190722.csv',index=False)

def judge_stable_analyze(data,col,address):
    num_counts=pd.crosstab(data['applymonth'],data[col],margins=True)
    num_counts_percents=num_counts.div(num_counts['All'],axis=0)
    num_counts_percents=num_counts_percents.drop('All',axis=1)
    num_counts_percents=num_counts_percents.drop('All',axis=0)
    
    bad_percents=pd.crosstab(index=data['applymonth'],columns=data[col],values=data['apply_y'],aggfunc='mean',margins=True)
    bad_percents=bad_percents.drop('All',axis=1)
    bad_percents=bad_percents.drop('All',axis=0)
    
    plt.cla()
    plt.figure(2)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for cn in num_counts_percents.columns:
       plt.sca(ax1)
       plt.plot(num_counts_percents[cn])
       plt.xticks(rotation=90)
       plt.legend()
       plt.title(col)
       plt.sca(ax2)
       plt.plot(bad_percents[cn])
       plt.xticks(rotation=90)
       plt.legend()
       plt.title(col)
    plt.savefig(address+col+'.jpg')
    plt.show()
    
    count_data=pd.concat([num_counts_percents,bad_percents],axis=0)
    count_data.to_csv(address+col+'.csv')

for cn in choose_column:
    judge_stable_analyze(full_data,cn,'D:/llm/联合建模算法开发/逻辑回归结果/逾期分析_加时间外验证/')



## 申请样本稳定性psi计算
psi_data=pd.read_table('psi_alldata_20190718.txt',dtype={'app_applycode':str},sep='\u0001')
psi_data=psi_data.replace('\\N',np.nan)
psi_data=pd.merge(psi_data,final_data[['app_applycode','y']],on='app_applycode',how='left')
newloan_label=pd.read_excel('./2019年3月7日之前的新增贷款客户申请编号.xlsx',converters={'applycode':str}) # 选择新增客户
psi_data=pd.merge(psi_data,newloan_label,left_on='app_applycode',right_on='applycode',how='inner')
del  newloan_label

cn=[cn.split('_Bin')[0] for cn in choose_column]
psi_data=psi_data[cn+['y','app_applydate']].copy()
psi_data.loc[psi_data.y.isnull(),'y']=0
de_dict_var = pd.read_excel('DE_Var_Select_20190613.xlsx')
for i, _ in de_dict.iterrows():
    name = de_dict_var.loc[i, 'var_name']
    default = de_dict_var.loc[i, 'default']
    if default != '""' and name in set(psi_data.columns) and name!='app_applycode':
        try:
            psi_data[name] = psi_data[name].astype('float64')
            if (psi_data[name] == float(default)).sum() > 0:
                psi_data.loc[psi_data[name] == float(default), name] = np.nan
        except:
            pass

    elif default == '""' and name in set(psi_data.columns) and name!='app_applycode':
        try:
            psi_data[name] = psi_data[name].astype('float64')
            if (psi_data[name] == float(-99)).sum() > 0:
                psi_data.loc[psi_data[name] == float(-99), name] = np.nan
            if (psi_data[name] == '-99').sum() > 0:
                psi_data.loc[psi_data[name] == '-99', name] = np.nan
        except:
            pass


for col in ['email_info_date','vehicle_minput_lastreleasedate']:  # 去除异常的时间
    try:
        psi_data.loc[psi_data[col] >= '2030-01-01', col] = np.nan
    except:
        pass

def date_cal(x, app_applydate):  # 计算申请日期距离其他日期的天数
    days_dt = pd.to_datetime(app_applydate) - pd.to_datetime(x)
    return days_dt.dt.days

for col in ['email_info_date','vehicle_minput_lastreleasedate']:
    if col != 'app_applydate':
        try:
            if col not in ['vehicle_minput_drivinglicensevaliditydate']:
                psi_data[col] = date_cal(psi_data[col], psi_data['app_applydate'])
            else:
                psi_data[col] = date_cal(psi_data['app_applydate'], psi_data[col])  # 计算行驶证有效期限距离申请日期的天数
        except:
            pass


psi_data=psi_data.fillna(-998)
psi_data["avg_sms_cnt_l6m"+'_Bin'] = psi_data["avg_sms_cnt_l6m"].map(lambda x: avg_sms_cnt_l6m(x))
psi_data["class2_black_cnt"+'_Bin'] = psi_data["class2_black_cnt"].map(lambda x: class2_black_cnt(x))
psi_data["coll_contact_total_sms_cnt"+'_Bin'] = psi_data["coll_contact_total_sms_cnt"].map(lambda x: coll_contact_total_sms_cnt(x))
psi_data["contact_bank_call_in_cnt"+'_Bin'] = psi_data["contact_bank_call_in_cnt"].map(lambda x: contact_bank_call_in_cnt(x))
psi_data["contact_bank_contact_weekday"+'_Bin'] = psi_data["contact_bank_contact_weekday"].map(lambda x: contact_bank_contact_weekday(x))
psi_data["contact_unknown_contact_early_morning"+'_Bin'] = psi_data["contact_unknown_contact_early_morning"].map(lambda x: contact_unknown_contact_early_morning(x))
psi_data["jxl_id_comb_othertel_num"+'_Bin'] = psi_data["jxl_id_comb_othertel_num"].map(lambda x: jxl_id_comb_othertel_num(x))
psi_data["m12_apply_platform_cnt"+'_Bin'] = psi_data["m12_apply_platform_cnt"].map(lambda x: m12_apply_platform_cnt(x))
psi_data["m3_id_relate_email_num"+'_Bin'] = psi_data["m3_id_relate_email_num"].map(lambda x: m3_id_relate_email_num(x))
#psi_data["m3_id_relate_homeaddress_num"+'_Bin'] = psi_data["m3_id_relate_homeaddress_num"].map(lambda x: m3_id_relate_homeaddress_num(x))
psi_data["max_call_cnt_l6m"+'_Bin'] = psi_data["max_call_cnt_l6m"].map(lambda x: max_call_cnt_l6m(x))
psi_data["max_overdue_terms"+'_Bin'] = psi_data["max_overdue_terms"].map(lambda x: max_overdue_terms(x))
psi_data["max_total_amount_l6m"+'_Bin'] = psi_data["max_total_amount_l6m"].map(lambda x: max_total_amount_l6m(x))
psi_data["phone_used_time"+'_Bin'] = psi_data["phone_used_time"].map(lambda x: phone_used_time(x))
psi_data["qtorg_query_orgcnt"+'_Bin'] = psi_data["qtorg_query_orgcnt"].map(lambda x: qtorg_query_orgcnt(x))
psi_data["times_by_current_org"+'_Bin'] = psi_data["times_by_current_org"].map(lambda x: times_by_current_org(x))
#psi_data["email_info_date"+'_Bin'] = psi_data["email_info_date"].map(lambda x: email_info_date(x))
psi_data["vehicle_minput_lastreleasedate"+'_Bin'] = psi_data["vehicle_minput_lastreleasedate"].map(lambda x: vehicle_minput_lastreleasedate(x))


for cn in choose_column:
    print(cn,psi_data[cn].unique())


pred_p5= lr.predict_proba(psi_data[choose_column])[:, 1]
psi_data['newloan_score3']=pred_p5*100

## 建模器申请样本集
modeltime_applydata=psi_data[(pd.to_datetime(psi_data.app_applydate)>=pd.to_datetime('2018-03-01 00:00:00')) & (pd.to_datetime(psi_data.app_applydate)<=pd.to_datetime('2018-10-31 23:59:59'))]
modeltime_applydata.loc[modeltime_applydata.y.isnull(),'y']=0
r_p_chart2(modeltime_applydata.y, (modeltime_applydata.newloan_score3.values/100).tolist(), min_scores, part=20)
fpr, tpr, th = roc_curve(modeltime_applydata.y, (modeltime_applydata.newloan_score3.values/100).tolist())
ks3 = tpr - fpr
print('all ks:   ' + str(max(ks3)))

## 时间外样本
outtime_psidata=modeltime_applydata=psi_data[(pd.to_datetime(psi_data.app_applydate)>=pd.to_datetime('2018-11-01 00:00:00')) & (pd.to_datetime(psi_data.app_applydate)<=pd.to_datetime('2018-11-30 23:59:59'))]
outtime_psidata['newloan_score3_segment']=pd.cut(outtime_psidata['newloan_score3'],cuts,right=False)
num_count=outtime_psidata['newloan_score3_segment'].value_counts().sort_index(ascending=False)
num_count_percents=num_count/num_count.sum()



