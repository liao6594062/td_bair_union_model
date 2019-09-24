# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:34:03 2019

@author: wj56740
"""

import os
import pandas as pd
import numpy as np
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
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
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
##1.1 导入相关表获取建模数据
'''

mdata_0 = pd.read_table('./data_20180702.txt', delimiter='\u0001', dtype={'app_applycode': str}) #读取决策引擎入参数据，从tbd中提取
mdata_0=mdata_0.replace('\\N',np.nan)
mdata_0=mdata_0[mdata_0.app_applycode.notnull()]
mdata_0 = mdata_0.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')

mdata_1 = pd.read_table('data_20180701-20190314.txt', delimiter='\u0001', dtype={'app_applycode': str}) #读取决策引擎入参数据，从tbd中提取
mdata_1=mdata_1.replace('\\N',np.nan)
mdata_1=mdata_1[mdata_1.app_applycode.notnull()]
mdata_1 = mdata_1.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')

mdata_2=pd.concat([mdata_0,mdata_1])
del mdata_0,mdata_1
gc.collect()
mdata_2 = mdata_2.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')


start_date = '2018-03-01 00:00:00'
end_date = '2018-11-30 23:59:59'
mdata_2 = mdata_2[(mdata_2.app_applydate >= start_date) & (mdata_2.app_applydate <= end_date)]
mdata_2 = mdata_2.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')


apply_contract_report = pd.read_table('applycode_20190416.txt',sep='\u0001',dtype={'applycode':str})  #车贷申请表
apply_contract_report=apply_contract_report.replace('\\N',np.nan)
apply_contract_report = apply_contract_report[(apply_contract_report.applycode.isnull() == False)].sort_values(['applycode','applydate']).drop_duplicates('applycode',keep='last')
mdata_3 = pd.merge(mdata_2, apply_contract_report, left_on='app_applycode', right_on='applycode', how='inner')
del apply_contract_report,mdata_2
gc.collect()
mdata_3 = mdata_3.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
mdata_3['applymonth'] = mdata_3.app_applydate.str.slice(0, 7)  # 生成申请月份
mdata_3['loantime_m']=mdata_3['loan_time'].copy()
mdata_3.loc[mdata_3.loantime_m.isin([1, 3, 6,12, 24, 36]) == False, 'loantime_m'] = 998  ##其余期限改为998


ylabel_data=pd.read_table('contractno_ylabel_data.txt',sep='\u0001',dtype={'applycode':str})  #车贷申请表
mdata_4=pd.merge(mdata_3,ylabel_data,on=['applycode','contractno'],how='inner')
del mdata_3,ylabel_data
gc.collect()
my_data=mdata_4[mdata_4.apply_y.isin([0,1])].copy()
my_data.to_excel('建模样本集—车贷决策引擎变量_20190722.xlsx',index=False)


'''
----------------------------
## 1.3  获取其他三方数据，主要是原始衍生变量数据，在这里合并主要是因为数据太大
----------------------------
'''
## 提取祥业聚信立报告
xiangye_jxl_report=pd.read_table('llm_jxl_report_20190506.txt', delimiter='\u0001', dtype={'apply_no': str}) #读取决策引擎入参数据，从tbd中提
xiangye_jxl_report=xiangye_jxl_report.replace('\\N',np.nan)
xiangye_jxl_report=xiangye_jxl_report[xiangye_jxl_report.apply_no.notnull()]
xiangye_jxl_report=xiangye_jxl_report.sort_values(['apply_no','query_time']).drop_duplicates('apply_no',keep='last')
jxl_report_dict=pd.read_excel('聚信立报告字典.xlsx')
xiangye_jxl_report=xiangye_jxl_report[jxl_report_dict['字段名'].values.tolist()]
jxl_columns_nullpercents=xiangye_jxl_report.isnull().sum()/xiangye_jxl_report.shape[0]
xiangye_jxl_report=xiangye_jxl_report[jxl_columns_nullpercents.loc[jxl_columns_nullpercents<0.95].index.tolist()]
del jxl_report_dict
final_data=pd.merge(my_data,xiangye_jxl_report,left_on='app_applycode',right_on='apply_no',how='left')
del xiangye_jxl_report,my_data
gc.collect()
final_data = final_data.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
final_data = final_data.drop(['apply_no'],axis=1)

#提取同盾数据源
td_data = pd.read_table('td_data_20190613.txt', delimiter='\u0001', dtype={'apply_no': str})
td_data=td_data.replace('\\N',np.nan)
td_data=td_data[td_data.apply_no.notnull()]
td_columns_nullpercents=td_data.isnull().sum()/td_data.shape[0]
#td_unique_num=td_data.apply(lambda x: len(x.unique())).reset_index().rename(columns={'index':'var_name',0.0:'unique_num'})
td_data=td_data[td_columns_nullpercents.loc[td_columns_nullpercents<0.95].index.tolist()]
final_data=pd.merge(final_data,td_data,left_on='app_applycode',right_on='apply_no',how='left')
del td_data
gc.collect()
final_data = final_data.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
final_data = final_data.drop(['apply_no'],axis=1)

#提取宜信数据源
yixin_data = pd.read_table('yixin_data_20190605.txt', delimiter='\u0001', dtype={'apply_no': str})
yixin_data=yixin_data.replace('\\N',np.nan)
yixin_data=yixin_data[yixin_data.apply_no.notnull()]
yixin_data=yixin_data.drop(['uniq_id','cr_no','query_time','dm_updatetime','id_card','mobile','queriedhistorycode','sharequerycode',
                  'querycreditscore','queryblacklistcode','queryloancode'],axis=1)
yixin_columns_nullpercents=yixin_data.isnull().sum()/yixin_data.shape[0]
yixin_data=yixin_data[yixin_columns_nullpercents.loc[yixin_columns_nullpercents<0.95].index.tolist()]
final_data=pd.merge(final_data,yixin_data,left_on='app_applycode',right_on='apply_no',how='left')
del yixin_data
gc.collect()
final_data = final_data.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
final_data = final_data.drop(['apply_no'],axis=1)


##提取上海资信数据
zixin_data = pd.read_table('zixin_data_20190527v1.txt', delimiter='\u0001', dtype={'applycode': str})
zixin_data=zixin_data.replace('\\N',np.nan)
zixin_data=zixin_data[zixin_data.applycode.notnull()]
zixin_columns_nullpercents=zixin_data.isnull().sum()/zixin_data.shape[0]
zixin_data=zixin_data[zixin_columns_nullpercents.loc[zixin_columns_nullpercents<0.95].index.tolist()]
final_data=pd.merge(final_data,zixin_data,left_on='app_applycode',right_on='applycode',how='left')
del zixin_data
gc.collect()
final_data = final_data.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
final_data = final_data.drop(['applycode_x','applycode_y'],axis=1)


## 整合同盾联合建模数据
td_union_data=pd.read_excel('CarLoanData_BringToTounawang.xlsx')
final_data['applydate']=final_data['app_applydate'].str.slice(0,10)
final_data=pd.merge(final_data,td_union_data,left_on=['credentials_no_md5','cust_name_md5','mobile_md5','applydate'],right_on=
                     ['credentials_no_md5_x','cust_name_md5_x','mobile_md5_x','applydate'],how='inner')
del td_union_data
gc.collect()
#final_data.to_excel('所有建模样本数据_20190722.xlsx',index=False)
#temp_6=final_data[(final_data.dkfs_x.isin(['押证'])) & (final_data.apply_y.isin([0,1])) ]
#count_stat_yz_fin_m = temp_6.groupby(['applymonth', 'loan_time']).app_applycode.agg('count').reset_index().rename(columns={'app_applycode': 'fin_count'})
#print(count_stat_yz_fin_m)
#count_sample_m2 = pd.merge(count_stat_yz_fin_m, final_data[final_data.apply_y.isin([0,1])].groupby(['applymonth', 'loan_time']).agg({'apply_y': ['count', 'sum', 'mean']}).reset_index(),
#                           on=['applymonth', 'loan_time'], how='left')


'''
## 2.变量预处理
'''

## 读入字典
tn_dict_1 = pd.read_excel('TouNa_DA-OAPPLIDT_20180814.xlsx')  ## 申请信息
tn_dict_2 = pd.read_excel('TouNa_DA-OBUREADT_20180814.xlsx')  ## 三方数据
tn_dict_1 = tn_dict_1[tn_dict_1.var_name.notnull()]
tn_dict_2 = tn_dict_2[tn_dict_2.var_name.notnull()]

tn_dict_3=pd.read_excel('上海资信接口文档.xlsx')
tn_dict_3=tn_dict_3[['字段名','类型','中文名','数据源']].rename(columns={'字段名':'var_name','类型':'var_type','中文名':'var_desc'})
tn_dict_3['length']=np.nan
tn_dict_3['var_code_comment']=np.nan
tn_dict_3['default']=np.nan
tn_dict_3.loc[tn_dict_3.var_type.isin(['varchar(100)','varchar(105)','varchar(110)', 'varchar(120)']),'var_type']='字符型'
tn_dict_3.loc[tn_dict_3.var_type.isin(['double']),'var_type']='数据型'
tn_dict_3.loc[tn_dict_3.var_type.isin(['timestamp','date']),'var_type']='日期型'
tn_dict_3.loc[tn_dict_3.var_type.isin(['字符型']),'default']=""
tn_dict_3.loc[tn_dict_3.var_type.isin(['数据型']),'default']=-99
tn_dict_3.loc[tn_dict_3.var_type.isin(['日期型']),'default']=99991231
tn_dict_3=tn_dict_3[['var_desc','var_type','length','var_name','var_code_comment','default','数据源']].drop_duplicates()


tn_dict_4=pd.read_excel('聚信立报告字典.xlsx')
tn_dict_4=tn_dict_4[['字段名','类型','中文名','数据源','逻辑说明']].rename(columns={'字段名':'var_name','类型':'var_type','中文名':'var_desc','逻辑说明':'var_code_comment'})
tn_dict_4['length']=np.nan
tn_dict_4['default']=np.nan
tn_dict_4.loc[tn_dict_4.var_type.isin(['varchar(100)','string']),'var_type']='字符型'
tn_dict_4.loc[tn_dict_4.var_type.isin(['int','double','bigint']),'var_type']='数据型'
tn_dict_4.loc[tn_dict_4.var_type.isin(['timestamp']),'var_type']='日期型'
tn_dict_4.loc[tn_dict_4.var_type.isin(['字符型']),'default']=""
tn_dict_4.loc[tn_dict_4.var_type.isin(['数据型']),'default']=-99
tn_dict_4.loc[tn_dict_4.var_type.isin(['日期型']),'default']=99991231
tn_dict_4=tn_dict_4[['var_desc','var_type','length','var_name','var_code_comment','default','数据源']].drop_duplicates()

tn_dict_5=pd.read_excel('宜信接口文档.xlsx')
tn_dict_5=tn_dict_5[['字段名','类型','中文名','数据源','逻辑说明']].rename(columns={'字段名':'var_name','类型':'var_type','中文名':'var_desc','逻辑说明':'var_code_comment'})
tn_dict_5['length']=np.nan
tn_dict_5['default']=np.nan
tn_dict_5.loc[tn_dict_5.var_type.isin(['varchar(100)','varchar(10)']),'var_type']='字符型'
tn_dict_5.loc[tn_dict_5.var_type.isin(['int','double','bigint']),'var_type']='数据型'
tn_dict_5.loc[tn_dict_5.var_type.isin(['timestamp']),'var_type']='日期型'
tn_dict_5.loc[tn_dict_5.var_type.isin(['字符型']),'default']=""
tn_dict_5.loc[tn_dict_5.var_type.isin(['数据型']),'default']=-99
tn_dict_5.loc[tn_dict_5.var_type.isin(['日期型']),'default']=99991231
tn_dict_5=tn_dict_5[['var_desc','var_type','length','var_name','var_code_comment','default','数据源']].drop_duplicates()


tn_dict_6=pd.read_excel('同盾接口文档.xlsx')
tn_dict_6=tn_dict_6[['字段名','类型','中文名','数据源','逻辑说明']].rename(columns={'字段名':'var_name','类型':'var_type','中文名':'var_desc','逻辑说明':'var_code_comment'})
tn_dict_6['length']=np.nan
tn_dict_6['default']=np.nan
tn_dict_6.loc[tn_dict_6.var_type.isin(['varchar(100)','varchar(10)']),'var_type']='字符型'
tn_dict_6.loc[tn_dict_6.var_type.isin(['int','double','bigint']),'var_type']='数据型'
tn_dict_6.loc[tn_dict_6.var_type.isin(['timestamp']),'var_type']='日期型'
tn_dict_6.loc[tn_dict_6.var_type.isin(['字符型']),'default']=""
tn_dict_6.loc[tn_dict_6.var_type.isin(['数据型']),'default']=-99
tn_dict_6.loc[tn_dict_6.var_type.isin(['日期型']),'default']=99991231
tn_dict_6=tn_dict_6[['var_desc','var_type','length','var_name','var_code_comment','default','数据源']].drop_duplicates()

tn_dict_7=pd.read_excel('同盾联合建模接口文档.xlsx')
tn_dict_7=tn_dict_7[['var_desc','var_type','length','var_name','var_code_comment','default','数据源']].drop_duplicates()


tn_dict = tn_dict_1.append(tn_dict_2).append(tn_dict_3).append(tn_dict_4).append(tn_dict_5).append(tn_dict_6).append(tn_dict_7)
tn_dict = tn_dict.drop_duplicates('var_name')

tn_dict['var_name'] = tn_dict.var_name.str.lower()
final_data.columns=[cn.lower() for cn in final_data.columns]
de_dict = pd.merge(tn_dict, pd.DataFrame(final_data.columns.tolist()), left_on='var_name', right_on=0, how='inner').drop(0,axis=1)
del tn_dict_1,tn_dict_2,tn_dict_3,tn_dict_4,tn_dict,tn_dict_5,tn_dict_6,tn_dict_7
#de_dict.to_excel('DE_Var_Select_20190722.xlsx',index=False)


'''
## 2.1 变量预处理---对变量进行质量评估，手动进行数据清洗，包括剔除缺失率非常高的变量、单一值变量以及其他明显无法使用的变量
'''

de_dict_var = pd.read_excel('DE_Var_Select_20190722.xlsx')
for i, _ in de_dict.iterrows():
    name = de_dict_var.loc[i, 'var_name']
    default = de_dict_var.loc[i, 'default']
    if default != '""' and name in set(final_data.columns) and name!='app_applycode':
        try:
            final_data[name] = final_data[name].astype('float64')
            if (final_data[name] == float(default)).sum() > 0:
                final_data.loc[final_data[name] == float(default), name] = np.nan
        except:
            pass

    elif default == '""' and name in set(final_data.columns) and name!='app_applycode':
        try:
            final_data[name] = final_data[name].astype('float64')
            if (final_data[name] == float(-99)).sum() > 0:
                final_data.loc[final_data[name] == float(-99), name] = np.nan
            if (final_data[name] == '-99').sum() > 0:
                final_data.loc[final_data[name] == '-99', name] = np.nan
        except:
            pass

## 统计各变量的质量情况,包括变量的总体缺失率、取不同值个数、按月统计变量的缺失率等

de_dict_var['null_percents'] = de_dict_var.var_name.map(lambda x: final_data[x].isnull().sum() / final_data.shape[0])  ##变量的总体缺失率
de_dict_var['different_values'] = de_dict_var.var_name.map(lambda x: len(final_data[x].unique()))  # 计算变量取值个数，包括缺失值
de_dict_var['whether_null'] = (de_dict_var['null_percents'] > 0).astype('int64')  # 变量是否缺失
de_dict_var['different_values'] = de_dict_var['different_values'] - de_dict_var['whether_null']  # 计算除了缺失值后的变量取值个数

applymonth = final_data.app_applydate.str.slice(0, 7)  # 取申请时间的月份
whether_null_matrix = final_data.isnull().astype('float64').groupby(applymonth)[final_data.columns.tolist()].mean()  # 按月统计变量的缺失率
whether_null_matrix = whether_null_matrix.T
whether_null_matrix['mean_null_percents'] = whether_null_matrix.mean(axis=1)  # 按月平均缺失率
whether_null_matrix['null_percents_std/mean'] = whether_null_matrix.std(axis=1) / whether_null_matrix.mean(axis=1)  # 按月缺失率标准差与均值的比值
whether_null_matrix = whether_null_matrix.reset_index().rename(columns={'index': 'var_name'})

de_dict_vars = pd.merge(de_dict_var, whether_null_matrix, on='var_name', how='inner')  # 变量评估，后台根据变量的质量指标进行手动清洗变量
del  whether_null_matrix,applymonth

de_dict_vars['是否选用']='是'
de_dict_vars['不选用的原因']=np.nan
de_dict_vars.loc[de_dict_vars['different_values']==1,'是否选用']='否'
de_dict_vars.loc[de_dict_vars['different_values']==1,'不选用的原因']='单一值'
de_dict_vars.loc[de_dict_vars['null_percents']>=0.95,'是否选用']='否'
de_dict_vars.loc[de_dict_vars['null_percents']>=0.95,'不选用的原因']='缺失率非常严重'
de_dict_vars.loc[de_dict_vars['null_percents']>=1,'是否选用']='否'
de_dict_vars.loc[de_dict_vars['null_percents']>=1,'不选用的原因']='完全缺失'
#de_dict_vars.to_excel('de_dict_vars_20190722.xlsx',index=False)
del de_dict_vars



'''
##2.2 变量预处理--针对不同的数据类型进行预处理
'''

vars_count_table=pd.read_excel('de_dict_vars_20190722.xlsx')
choose_columns_table = vars_count_table[vars_count_table['是否选用'].isin(['是'])]
choose_columns_table=choose_columns_table[choose_columns_table['数据源'].isin(['同盾联合建模'])==False]
numeric_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['\ufeff数据型','数据型', '数字型','数值型']), 'var_name'].values.tolist()
str_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['字符型', '\ufeff字符型','字符型 ' ]), 'var_name'].values.tolist()
date_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['\ufeff日期型','日期型']), 'var_name'].values.tolist()
final_data = final_data.replace('\\N',np.nan)



'''
##  处理数据型变量
'''

# 列出数据型变量异常值并对异常值进行处理 ，有l6m_avg_net_flow、l6m_avg_total_amount、net_flow、total_amount、jxl_call_num_aver_6months中出现负值
for col in numeric_columns: 
    try:
        final_data[col]=final_data[col].astype('float64')
        if final_data.loc[final_data[col] < 0, col].shape[0] > 0:
            print(col + ':', final_data.loc[final_data[col] < 0, col].unique(),
                  final_data.loc[final_data[col] < 0, col].shape[0]/final_data.shape[0])
            final_data.loc[final_data[col]==-1,col]=np.nan
        
    except: 
        pass

   
'''
##  处理日期型变量，将日期变量转为距离申请日期的天数
'''
for col in date_columns:  # 去除异常的时间
    try:
        final_data.loc[final_data[col] >= '2030-01-01', col] = np.nan
    except:
        pass


def date_cal(x, app_applydate):  # 计算申请日期距离其他日期的天数
    days_dt = pd.to_datetime(app_applydate) - pd.to_datetime(x)
    return days_dt.dt.days


for col in date_columns:
    if col != 'app_applydate':
        try:
            if col not in ['vehicle_minput_drivinglicensevaliditydate']:
                final_data[col] = date_cal(final_data[col], final_data['app_applydate'])
            else:
                final_data[col] = date_cal(final_data['app_applydate'], final_data[col])  # 计算行驶证有效期限距离申请日期的天数
        except:
            pass


'''
##  处理字符型变量，将一些不统计的取值统一
'''

for col in str_columns:  # 列出字符变量的不同取值
    print(col + ':', final_data[col].unique())

abnorm_str_columns = [ 'apt_facetrial_marry','jxl_110_record','jxl_120_record','jxl_macau_phone_record','jxl_court_phone_record',
                      'jxl_law_phone_record','jxl_id_operator','jxl_name_operator','apt_education','tx_cardid']


final_data.loc[final_data.jxl_110_record.isin(['偶尔通话(3次以内，包括3次)','多次通话(3次以上)']),'jxl_110_record']='有'
final_data.loc[final_data.jxl_110_record.isin(['无通话记录']),'jxl_110_record']='无'
final_data.loc[final_data.jxl_110_record.isin(['无数据']),'jxl_110_record']=np.nan

final_data.loc[final_data.jxl_120_record.isin(['偶尔通话(3次以内，包括3次)','多次通话(3次以上)']),'jxl_120_record']='有'
final_data.loc[final_data.jxl_120_record.isin(['无通话记录']),'jxl_120_record']='无'
final_data.loc[final_data.jxl_120_record.isin(['无数据']),'jxl_120_record']=np.nan

final_data.loc[final_data.jxl_macau_phone_record.isin(['偶尔通话(3次以内，包括3次)','多次通话(3次以上)']),'jxl_macau_phone_record']='有'
final_data.loc[final_data.jxl_macau_phone_record.isin(['无通话记录']),'jxl_macau_phone_record']='无'
final_data.loc[final_data.jxl_macau_phone_record.isin(['无数据']),'jxl_macau_phone_record']=np.nan

final_data.loc[final_data.jxl_court_phone_record.isin(['偶尔通话(3次以内，包括3次)','多次通话(3次以上)']),'jxl_court_phone_record']='有'
final_data.loc[final_data.jxl_court_phone_record.isin(['无通话记录']),'jxl_court_phone_record']='无'
final_data.loc[final_data.jxl_court_phone_record.isin(['无数据']),'jxl_court_phone_record']=np.nan

final_data.loc[final_data.jxl_law_phone_record.isin(['偶尔通话(3次以内，包括3次)','多次通话(3次以上)']),'jxl_law_phone_record']='有'
final_data.loc[final_data.jxl_law_phone_record.isin(['无通话记录']),'jxl_law_phone_record']='无'
final_data.loc[final_data.jxl_law_phone_record.isin(['无数据']),'jxl_law_phone_record']=np.nan

final_data.loc[final_data.jxl_id_operator.isin(['运营商未提供身份证号码']),'jxl_id_operator']='运营商未提供身份证号'
final_data.loc[final_data.jxl_id_operator.isin(['匹配']),'jxl_id_operator']='成功'
final_data.loc[final_data.jxl_id_operator.isin(['不匹配']),'jxl_id_operator']='失败'

final_data.loc[final_data.jxl_name_operator.isin(['匹配']),'jxl_name_operator']='成功'
final_data.loc[final_data.jxl_name_operator.isin(['不匹配']),'jxl_name_operator']='失败'

final_data.loc[final_data.tx_cardid.isin(['NO_DATA']),'tx_cardid']='NO_DA'
final_data.loc[final_data.tx_cardid.isin(['DIFFERENT']),'tx_cardid']='DIFFE'



'''
## 3.变量衍生,衍生了3个变量
'''

## 数值型变量衍生

final_data['age_and_gender']=1
final_data.loc[(final_data.apt_age>=42) & (final_data.apt_gender.isin(['男'])),'age_and_gender']=0  ##年龄与性别组合
final_data.loc[(final_data.apt_gender.isin(['女'])),'age_and_gender']=0



'''
## 4.将样本拆分成训练集、测试集以及验证集，利用逾期情况对字符型变量或者其他变量分箱操作
'''
model_data=final_data[final_data.app_applydate<='2018-10-31 23:59:59'].copy()
y = model_data['apply_y']
x = model_data.drop('apply_y',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3,stratify = y ,random_state=0)


## 根据训练集上的单变量分析，绘制变量的逾期分布图，确定将字符变量转成什么样的哑变量
def single_analyze(x, y, var_name):
    var_data = x[var_name].copy()
    try:
        var_data = var_data.astype('float64')
        var_data[var_data.isnull()] = -99
    except:
        var_data[var_data.isnull()] = '-99'

    var_name_overdue_analyze1 = pd.crosstab(var_data, y, margins=True).rename(columns={0.0: 'good_num', 1.0: 'bad_num', 'All': 'total_num'}).reset_index()
    var_name_overdue_analyze2 = var_name_overdue_analyze1[['good_num', 'bad_num', 'total_num']].div(var_name_overdue_analyze1['total_num'], axis=0).rename(columns={'good_num': 'good_percents', 'bad_num': 'bad_percents', 'total_num': 'total_percents'})
    var_name_overdue_analyze2[var_name] = var_name_overdue_analyze1[var_name].copy()
    var_name_overdue_analyze = pd.merge(var_name_overdue_analyze1, var_name_overdue_analyze2, on=var_name, how='inner')

    plt.cla()
    var_name_overdue_analyze3 = pd.crosstab(var_data, y, margins=True).rename(columns={0.0: 'good_num', 1.0: 'bad_num', 'All': 'total_num'})
    var_name_overdue_analyze3 = var_name_overdue_analyze3.div(var_name_overdue_analyze3['total_num'], axis=0).drop('All', axis=0)
    var_name_overdue_analyze3.bad_num.plot(kind='bar', color='k', alpha=0.7, title='overdue_percents')
    plt.savefig('D:/llm/联合建模算法开发/结果/字符变量单变量分析/' + var_name + '.png')

    var_name_overdue_analyze.to_excel('D:/llm/联合建模算法开发/结果/字符变量单变量分析/' + var_name + '.xlsx',index=False)

    return var_name_overdue_analyze


for var_name in str_columns:
    var_name_overdue_analyze = single_analyze(x_train[choose_columns_table['var_name'].values.tolist()],y_train, var_name)
    

## 哑变量衍生
model_data['final_decision_accept&-998'] = ((model_data.loc[:, 'final_decision'].isin(['Accept'])) | (model_data.loc[:, 'final_decision'].isnull())).astype('float64')
model_data['vehicle_minput_lastmortgagerinfo_3n4n-998'] = ((model_data.loc[:, 'vehicle_minput_lastmortgagerinfo'].isin([3,4])) | (model_data.loc[:, 'vehicle_minput_lastmortgagerinfo'].isnull())).astype('float64')
model_data['apt_gender_0'] = model_data.loc[:, 'apt_gender'].isin(['女']).astype('float64')
model_data['contact_bank_type_2'] = model_data.loc[:, 'contact_bank_type'].isin(['很少被联系','无该类号码记录']).astype('float64')
model_data['id_fin_black_arised_0']= model_data.loc[:, 'id_fin_black_arised'].isin([0]).astype('float64')
model_data['jxl_110_record_有']= model_data.loc[:, 'jxl_110_record'].isin(['有']).astype('float64')

dummy_columns = [ 'final_decision_accept&-998', 'vehicle_minput_lastmortgagerinfo_3n4n-998', 'apt_gender_0','contact_bank_type_2','id_fin_black_arised_0','jxl_110_record_有']

# 衍生后的变量间的相关性分析
 
columns_corrcoef = model_data.drop(str_columns, axis=1).corr()

for col in model_data:
    if model_data[col].isnull().sum() > 0 and col!='y':
        model_data.loc[model_data[col].isnull(), col] = -998


 #选择入模前的最终变量

numeric_columns.append('age_and_gender')


print(len(numeric_columns))  # 722
print(len(date_columns))  # 14
print(len(dummy_columns))  # 6


'''
## 汇集好坏样本，单变量分析 (用 R作图)
'''

full_data = model_data.copy()
sample_data=full_data[full_data.apply_y.notnull()].copy()
sample_data.to_csv('所有新增建模集数据_20190723_xgboost.csv',index=False)



'''
## 4. 模型训练
'''

## 拆分训练集和验证集
ori_columns = numeric_columns + date_columns + dummy_columns
ori_columns.remove('m3_apply_mobile_platform_cnt') #剔除不稳定的变量
ori_columns.remove('cur_overdue_amount')
#ori_columns.remove('apt_ec_lastoverduedaystotal')
ori_columns.remove('max_overdue_amount')
ori_columns.remove('m6_apply_onlinebk')
y = model_data['apply_y']
x = model_data[ori_columns]
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3,stratify = y ,random_state=0)



## 用lightgbm筛选变量,将相关性大于0.7以上的按重要性剔除重要性更低的变量
params={'boosting_type':'gbdt','learning_rate':0.05,'objective':'binary',
 'metric':'auc','max_depth':5,'verbose':2,'random_state':0}
num_round=500
early_stopping_rounds=300
train_matrix=lgb.Dataset(x_train,label=y_train)
test_matrix=lgb.Dataset(x_test,label=y_test)
model=lgb.train(params,train_matrix,num_round,valid_sets=test_matrix,early_stopping_rounds=early_stopping_rounds)
importances=model.feature_importance()/model.feature_importance().sum()
feature_importances=pd.DataFrame(importances,index=model.feature_name()).reset_index().rename(columns={'index':'var_name',0.0:'importances'})
feature_importances=feature_importances[feature_importances.importances>=1/x_train.shape[0]]
a = model.feature_importance()/model.feature_importance().sum().tolist()
importances=pd.DataFrame(np.array(a), index=model.feature_name(),columns=['importance'])
del_columns=[]
del_desc={}
for onecol in model.feature_name():
    meet_columns=columns_corrcoef.loc[columns_corrcoef[onecol].abs()>=0.70,:].index
    meet_importances=importances.loc[meet_columns,:]
    del_columns1=meet_importances.loc[meet_importances['importance']<meet_importances['importance'].max(),:].index.tolist()
    del_columns2=meet_importances.loc[meet_importances['importance']>=meet_importances['importance'].max(),:].index.tolist()
    del_desc2={}
    for cn in meet_importances.index:
        del_desc2[cn]=meet_importances.loc[cn,'importance']
    if len(del_columns1)!=0 and len(del_columns2)!=0:
        del_desc[del_columns2[0]]=del_desc2
    del_columns.extend(del_columns1)
final_columns=[col for col in model.feature_name() if col not in del_columns]
print(len(final_columns))



##  选取在模型中重要性不小于给定阈值的变量
a=range(len(final_columns)+1)
tol=0.03
clf=XGBClassifier(learning_rate=0.05,
                  min_child_weight=0.8,
                  max_depth=5,
                  gamma=20,
                  n_estimators=300,
                   random_state=0,objective='binary:logistic')

while (len(a)>len(final_columns)):    
    ## 训练模型并估计参数    
    clf.fit(x_train[final_columns], y_train)     
    pred_p = clf.predict_proba(x_train[final_columns])[:,1]
    fpr, tpr, th = roc_curve(y_train, pred_p)
    ks = tpr - fpr
    pred_p2 = clf.predict_proba(x_test[final_columns])[:,1]
    fpr, tpr, th = roc_curve(y_test, pred_p2)
    ks2 = tpr - fpr
    a = clf.feature_importances_.tolist()
    print(final_columns)
    print('len(final_columns)= ', len(final_columns))
    print('minimum feature importance = ', min(a))
    print('train ks: ' + str(max(ks)))
    print('test ks:  ' + str(max(ks2)))

    if( min(a) < tol ):
        b = (clf.feature_importances_ == min(a))
        final_columns = [final_columns[i] for i in range(len(final_columns)) if b[i]==False]


'''
# 最终模型
'''

columns=['times_by_current_org',
 'vehicle_evtrpt_b2bprice',
 'query_orgcnt',
 'm3_apply_onlinebk',
 'm6_apply_thirdservcprvd',
 'm3_id_relate_homeaddress_num',
 'avg_call_in_time_rank1',
 'phone_used_time',
 'contact_bank_contact_1m',
 'contact_unknown_contact_early_morning',
 'phone_gray_score',
 'query_cnt_l3m',
 'max_overdue_terms',
 'jxl_id_comb_othertel_num',
 'apt_ec_overduedaystotallastyear',
 'apt_ec_lastloansettleddate',
 'vehicle_minput_lastreleasedate',
 'vehicle_minput_lastmortgagerinfo_3n4n-998']
print(len(columns))
clf=XGBClassifier(learning_rate=0.1,
                  min_child_weight=0.8,
                  max_depth=5,
                  gamma=20,
                  n_estimators=300,
                  random_state=0,objective='binary:logistic')
clf.fit(x_train[columns], y_train)
a = clf.feature_importances_.tolist()
pred_p = clf.predict_proba(x_train[columns])[:, 1]
fpr, tpr, th = roc_curve(y_train, pred_p)
ks = tpr - fpr
pred_p2 = clf.predict_proba(x_test[columns])[:, 1]
fpr, tpr, th = roc_curve(y_test, pred_p2)
ks2 = tpr - fpr

feature_importance = pd.DataFrame({'var_name': columns, 'importance': a}).sort_values('importance', ascending=False)
feature_importance = pd.merge(feature_importance, choose_columns_table[['var_name', 'var_desc','数据源']], on='var_name', how='left')[['var_name', 'var_desc','数据源', 'importance']]
print(feature_importance)
print('train ks: ' + str(max(ks)))
print('test ks:  ' + str(max(ks2)))

#pickle.dump(clf, open('D:/llm/联合建模算法开发/结果/训练结果/newcar_xgboost_owndata_2019090725v1.pkl', 'wb'))
#feature_importance.to_excel('D:/llm/联合建模算法开发/结果/训练结果/feature_importance_owndata_20190725v1.xlsx',index=False)



'''
## 模型表现 
'''

pred_p = clf.predict_proba(x_train[columns])[:, 1]
min_scores = r_p_chart(y_train, pred_p, part=20)
min_scores = [round(i, 5) for i in min_scores]
min_scores[19] = 0
cuts = [round(min_scores[i] * 100.0, 3) for i in range(20)[::-1]] + [100.0]

print('训练集')
pred_p = clf.predict_proba(x_train[columns])[:, 1]
fpr, tpr, th = roc_curve(y_train, pred_p)
ks = tpr - fpr
print('train ks: ' + str(max(ks)))
print(roc_auc_score(y_train, pred_p))
r_p_chart2(y_train, pred_p, min_scores, part=20)

print('测试集')
pred_p2 = clf.predict_proba(x_test[columns])[:, 1]
fpr, tpr, th = roc_curve(y_test, pred_p2)
ks2 = tpr - fpr
print('test ks:  ' + str(max(ks2)))
print(roc_auc_score(y_test, pred_p2))
r_p_chart2(y_test, pred_p2, min_scores, part=20)

print('建模全集')
pred_p3 = clf.predict_proba(x[columns])[:, 1]
fpr, tpr, th = roc_curve(y, pred_p3)
ks3 = tpr - fpr
print('all ks:   ' + str(max(ks3)))
print(roc_auc_score(y, pred_p3))
r_p_chart2(y, pred_p3, min_scores, part=20)


'''
## 入模变量稳定性分析,要包含取值分布、逾期分布，均值分布、缺失分布、iv和ks的计算等
'''

## 统计均值
xx_data_s=model_data[['applydate']+columns]
xx_data_s[xx_data_s==-998]=np.nan
xx_data_s['applymonth']=xx_data_s['applydate'].str.slice(0,7)
vars_mean_count=xx_data_s.groupby(xx_data_s['applymonth'])[xx_data_s.columns.tolist()].mean().T.reset_index().rename(columns={'index':'var_name'})
vars_mean_count=pd.merge(vars_mean_count,choose_columns_table[['var_name','var_desc']],on='var_name',how='left')
vars_mean_count.set_index('var_name').ix[columns,:].reset_index().to_excel('D:/llm/联合建模算法开发\结果\训练结果.xlsx',index=False)
del xx_data_s,vars_mean_count

## 统计缺失率
xx_data_s=model_data[['applydate']+columns]
xx_data_s[xx_data_s==-998]=np.nan
applymonth=xx_data_s['applydate'].str.slice(0,7)
xx_data_s=xx_data_s.drop('applydate',axis=1).isnull().astype('float64')
null_percents_count = xx_data_s.groupby(applymonth)[xx_data_s.columns.tolist()].mean().T.reset_index().rename(columns={'index': 'var_name', 'applymonth': '申请月份'})
null_percents_count=pd.merge(null_percents_count,choose_columns_table[['var_name','var_desc']],on='var_name',how='left')
null_percents_count.set_index('var_name').ix[columns,:].reset_index().to_excel('所有入模变量的缺失率分布_20190703.xlsx',index=False)
del xx_data_s,null_percents_count


## 统计图
def judge_stable_analyze(xx_data_s,col,address):
    xx_data_s[xx_data_s==-998]=np.nan
    applymonth=xx_data_s['applydate'].str.slice(0,7)
    xx_data_1=xx_data_s.drop('applydate',axis=1).isnull().astype('float64')
    null_percents=xx_data_1.groupby(applymonth)[col].mean()
    mean_value=xx_data_s.groupby(applymonth)[col].mean()

    plt.cla()
    plt.figure(2)
    plt.figure(figsize=(10,5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.sca(ax1)
    plt.plot(null_percents)
    plt.xticks(rotation=90)
    plt.title(col+'_nullpercents')
    plt.sca(ax2)
    plt.plot(mean_value)
    plt.xticks(rotation=90)
    plt.title(col+'_meanvalue')
    plt.savefig(address+col+'.jpg')
    plt.show()

xx_data_s=model_data[['applydate']+columns]  
for cn in columns:
    judge_stable_analyze(xx_data_s,cn,'D:/llm/联合建模算法开发/结果/训练结果/入模变量的缺失率情况_自有数据_自有数据/')





'''
时间外验证
'''
outtime_testdata=pd.read_table('tdunion_outtime_data_20190725.txt',dtype={'app_applycode':str},sep='\u0001')
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

de_dict_var = pd.read_excel('de_dict_vars_20190722.xlsx')
for i, _ in de_dict.iterrows():
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


for col in ['vehicle_minput_lastreleasedate','prof_title_info_date','cell_reg_time','apt_ec_lastloansettleddate']:  # 去除异常的时间
    try:
        outtime_testdata.loc[outtime_testdata[col] >= '2030-01-01', col] = np.nan
    except:
        pass

def date_cal(x, app_applydate):  # 计算申请日期距离其他日期的天数
    days_dt = pd.to_datetime(app_applydate) - pd.to_datetime(x)
    return days_dt.dt.days

for col in ['prof_title_info_date','vehicle_minput_lastreleasedate','cell_reg_time','apt_ec_lastloansettleddate']:
    if col != 'app_applydate':
        try:
            if col not in ['vehicle_minput_drivinglicensevaliditydate']:
                outtime_testdata[col] = date_cal(outtime_testdata[col], outtime_testdata['app_applydate'])
            else:
                outtime_testdata[col] = date_cal(outtime_testdata['app_applydate'], outtime_testdata[col])  # 计算行驶证有效期限距离申请日期的天数
        except:
            pass

##变量衍生
outtime_testdata['vehicle_minput_lastmortgagerinfo_3n4n-998'] = ((outtime_testdata.loc[:, 'vehicle_minput_lastmortgagerinfo'].isin([3,4])) | (outtime_testdata.loc[:, 'vehicle_minput_lastmortgagerinfo'].isnull())).astype('float64')
numerical_cols=['times_by_current_org',
 'vehicle_evtrpt_b2bprice',
 'query_orgcnt',
 'm3_apply_onlinebk',
 'm6_apply_thirdservcprvd',
 'd7_id_apply_num',
 'm3_id_relate_homeaddress_num',
 'd7_mobile_apply_num',
 'm3_id_relate_mobile_num',
 'avg_call_in_time_rank1',
 'avg_call_out_time_rank1',
 'phone_used_time',
 'contact_bank_contact_1m',
 'contact_unknown_contact_early_morning',
 'searched_org_cnt',
 'phone_gray_score',
 'query_cnt_l3m',
 'max_overdue_terms',
 'jxl_id_comb_othertel_num',
 'apt_ec_overduedaystotallastyear']
for cn in numerical_cols:
    outtime_testdata.loc[outtime_testdata[cn]<0,cn]=-998
## 时间外验证
print('时间外验证')
pred_p4= clf.predict_proba(outtime_testdata[columns])[:, 1]
fpr, tpr, th = roc_curve(outtime_testdata.apply_y, pred_p4)
ks3 = tpr - fpr
print('all ks:   ' + str(max(ks3)))
print(roc_auc_score(outtime_testdata.apply_y, pred_p4))
r_p_chart2(outtime_testdata.apply_y, pred_p4, min_scores, part=20)


'''
## 入模变量稳定性分析,要包含取值分布、逾期分布，均值分布、缺失分布、iv和ks的计算等
'''
##统计分数-ks值和auc值
xx_data_s=pd.concat([model_data[['app_applydate','apply_y','contractno','app_applycode']+columns],outtime_testdata[['app_applydate','apply_y','contractno','app_applycode']+columns]])
xx_data_s['applymonth']=xx_data_s['app_applydate'].str.slice(0,7)
pred_p5= clf.predict_proba(xx_data_s[columns])[:, 1]
xx_data_s['own_scores']=pred_p5*100
num_counts=pd.crosstab(xx_data_s['applymonth'],xx_data_s['apply_y'],margins=True).reset_index()
ks_list={}
auc_list={}
for cn in xx_data_s['applymonth'].unique():
    temp=xx_data_s.loc[xx_data_s['applymonth']==cn,['apply_y','own_scores']].copy()
    fpr, tpr, th = roc_curve(temp['apply_y'], temp['own_scores'].values/100)
    ks2 = tpr - fpr
    ks_list[cn]=max(ks2)
    auc_list[cn]=roc_auc_score(temp['apply_y'], temp['own_scores'].values/100)
    
ks_pd=pd.Series(ks_list)
ks_pd=ks_pd.reset_index().rename(columns={'index':'applymonth',0:'ks'})
auc_pd=pd.Series(auc_list)
auc_pd=auc_pd.reset_index().rename(columns={'index':'applymonth',0:'auc'})
auc_ks=pd.merge(ks_pd,auc_pd,on='applymonth',how='inner')
print(auc_ks)
xx_data_s[['applymonth','contractno','app_applycode','own_scores']].to_excel('D:/llm/联合建模算法开发/文档/自有数据建模分数_20190726.xlsx',index=False)







## 统计均值
xx_data_s=pd.concat([model_data[['app_applydate']+columns],outtime_testdata[['app_applydate']+columns]])
xx_data_s[xx_data_s==-998]=np.nan
xx_data_s['applymonth']=xx_data_s['app_applydate'].str.slice(0,7)
vars_mean_count=xx_data_s.groupby(xx_data_s['applymonth'])[xx_data_s.columns.tolist()].mean().T.reset_index().rename(columns={'index':'var_name'})
vars_mean_count=pd.merge(vars_mean_count,choose_columns_table[['var_name','var_desc']],on='var_name',how='left')
vars_mean_count.set_index('var_name').ix[columns,:].reset_index().to_excel('所有入模变量的均值分布_20190703v1.xlsx',index=False)
del xx_data_s,vars_mean_count

## 统计缺失率
xx_data_s=pd.concat([model_data[['app_applydate']+columns],outtime_testdata[['app_applydate']+columns]])
xx_data_s[xx_data_s==-998]=np.nan
applymonth=xx_data_s['app_applydate'].str.slice(0,7)
xx_data_s=xx_data_s.drop('app_applydate',axis=1).isnull().astype('float64')
null_percents_count = xx_data_s.groupby(applymonth)[xx_data_s.columns.tolist()].mean().T.reset_index().rename(columns={'index': 'var_name', 'applymonth': '申请月份'})
null_percents_count=pd.merge(null_percents_count,choose_columns_table[['var_name','var_desc']],on='var_name',how='left')
null_percents_count.set_index('var_name').ix[columns,:].reset_index().to_excel('所有入模变量的缺失率分布_20190703v1.xlsx',index=False)
del xx_data_s,null_percents_count

## 取值分布，主要针对字符型变量
request_cols=['app_applydate','apt_education', 'apt_facetrial_householdregister', 'apt_facetrial_housetype',
                 'apt_facetrial_marry', 'apt_facetrial_residertogether','apt_gender', 'id_fin_black_arised',
                 'vehicle_buymode','vehicle_minput_lastmortgagerinfo','vehicle_minput_registcertflag']
xx_data_s=pd.concat([model_data[request_cols],outtime_testdata[request_cols]])
xx_data_s['applymonth']=xx_data_s['app_applydate'].str.slice(0,7)
dummy_mean_count=pd.crosstab(xx_data_s['applymonth'],xx_data_s['apt_education'],margins=True)
dummy_mean_count=dummy_mean_count.div(dummy_mean_count['All'],axis=0).drop('All',axis=1).reset_index()

request_cols2=['apt_facetrial_householdregister', 'apt_facetrial_housetype',
                 'apt_facetrial_marry', 'apt_facetrial_residertogether','apt_gender', 'id_fin_black_arised',
                 'vehicle_buymode','vehicle_minput_lastmortgagerinfo','vehicle_minput_registcertflag']
for cn in request_cols2:
    try:
       dummy_mean_count_1 = pd.crosstab(xx_data_s['applymonth'], xx_data_s[cn], margins=True)
    except:
       xx_data_s[cn]=xx_data_s[cn].astype('str')
       dummy_mean_count_1 = pd.crosstab(xx_data_s['applymonth'], xx_data_s[cn], margins=True)

    dummy_mean_count_1 = dummy_mean_count_1.div(dummy_mean_count_1['All'], axis=0).drop('All', axis=1).reset_index()
    dummy_mean_count=pd.merge(dummy_mean_count,dummy_mean_count_1,on='applymonth',how='inner')

dummy_mean_count.to_excel('哑变量均值按月统计_20190703v1.xlsx',index=False)
del request_cols,xx_data_s,dummy_mean_count,request_cols2,dummy_mean_count_1

## 逾期分布
def achieve_score_ks(ja_analyzedata,col,times_term,segment_num=10,cuts_min_scores=None):
   gd_apply_termdata = ja_analyzedata[[col,times_term]].copy()
   gd_apply_termdata['applymonth']=ja_analyzedata['app_applydate'].str.slice(0,7)
   
   try:
        gd_apply_termdata[col] = gd_apply_termdata[col].astype('float64')
        gd_apply_termdata.loc[gd_apply_termdata[col].isnull(),col] = -99
   except:
        gd_apply_termdata.loc[gd_apply_termdata[col].isnull(),col]  = '-99'

   whether_str=1
   try:
      gd_apply_termdata[col]=gd_apply_termdata[col].astype('float64')
      whether_str=0
   except:
      gd_apply_termdata[col] = gd_apply_termdata[col].astype('str')

   if   whether_str==1:
       cuts=gd_apply_termdata[col].unique().tolist()
   elif  len(gd_apply_termdata[col].unique())<=segment_num:
       cuts=gd_apply_termdata[col].unique().tolist()
   else:
       min_value=gd_apply_termdata[col].min()
       if  min_value<0:
           middle_cuts=np.unique(gd_apply_termdata.loc[gd_apply_termdata[col].isin([-99])==False,col].quantile(np.arange(0,1+1/segment_num,1/segment_num)).values).tolist()
           middle_cuts[0]=0
           middle_cuts.append(-998)
           cuts=np.unique(np.array(middle_cuts)).tolist()
           cuts[len(cuts)-1]=np.inf
       else:
           cuts = np.unique(gd_apply_termdata[col].quantile(np.arange(0, 1 + 1 / segment_num, 1/segment_num)).values).tolist()
           cuts[len(cuts) - 1] = np.inf
       if cuts_min_scores==None:
          gd_apply_termdata[col]=pd.cut(gd_apply_termdata[col],cuts,right=False)
       else:
          gd_apply_termdata[col] = pd.cut(gd_apply_termdata[col], cuts_min_scores, right=False)
   gd_apply_termdata['applymonth']=gd_apply_termdata['applymonth'].astype('str')
   col_overdue_crosstab=gd_apply_termdata.groupby(['applymonth']+[col])[times_term].agg(['count','sum','mean']).reset_index()
   
   
   return  col_overdue_crosstab


xx_data_s=pd.concat([model_data[['app_applydate']+columns+['y']],outtime_testdata[['app_applydate']+columns+['y']]])
for cn  in  columns:
        col_overdue_crosstab=achieve_score_ks(xx_data_s,cn,'y',5)
        col_overdue_crosstab.to_csv('D:/llm/车贷新增客户三期模型开发/结果/入模变量逾期分布/'+cn+'.csv',index=False)
del  xx_data_s,col_overdue_crosstab



 
'''
## 对全量申请样本进行评分，可选择申请月份
'''
vars_count_table=pd.read_excel('de_dict_vars_20190305v1.xlsx')
choose_columns_table = vars_count_table[vars_count_table.whether_choose == 1]
numeric_columns_s = choose_columns_table.ix[choose_columns_table.var_type == '\ufeff数据型', 'var_name'].values.tolist()
str_columns_s = choose_columns_table.ix[choose_columns_table.var_type == '\ufeff字符型', 'var_name'].values.tolist()
date_columns_s = choose_columns_table.ix[choose_columns_table.var_type == '\ufeff日期型', 'var_name'].values.tolist()


'''
计算新增客户申请评分二期模型分数
'''
psi_data=pd.read_table('car_comput_psi_data_20190704.txt',dtype={'app_applycode':str},sep='\u0001')
paymentdata_phases_counts_1=pd.read_csv('./paymentdata_phases_counts_20190605.csv',encoding='utf-8') #坏客户标签
psi_data= pd.merge(psi_data, paymentdata_phases_counts_1, on='contractno', how='left')
del paymentdata_phases_counts_1
psi_data=psi_data.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
psi_data=psi_data.drop('applycode',axis=1)

newloan_label=pd.read_excel('./2019年3月7日之前的新增贷款客户申请编号.xlsx',converters={'applycode':str}) # 选择新增客户
psi_data=pd.merge(psi_data,newloan_label,left_on='app_applycode',right_on='applycode',how='inner')
del  newloan_label
psi_data['fin_phases']=psi_data[['fin_totalphases','realtotalphases']].max(axis=1)
psi_data['TotalPhases_m'] = psi_data['fin_phases'].copy() ##实际贷款期限，与申请期限略有差别
psi_data.loc[psi_data.TotalPhases_m.isin([1, 3, 6,12, 24, 36]) == False, 'TotalPhases_m'] = 998  ##其余期限改为998
psi_data=psi_data.drop('applycode',axis=1)


de_dict_var = pd.read_excel('DE_Var_Select_20190613.xlsx')  # 处理默认值为np.nan
for i, _ in de_dict_var.iterrows():
    name = de_dict_var.loc[i, 'var_name']
    default = de_dict_var.loc[i, 'default']
    if default != '""' and name in set(psi_data.columns) and name!='app_applycode':
        try:
            psi_data[name] = psi_data[name].astype('float64')
            if (psi_data[name] == psi_data(default)).sum() > 0:
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
        
        
for col in columns: 
    try:
        psi_data[col]=psi_data[col].astype('float64')
        if psi_data.loc[psi_data[col] < 0, col].shape[0] > 0:
            print(col + ':', psi_data.loc[psi_data[col] < 0, col].unique(),
                  psi_data.loc[psi_data[col] < 0, col].shape[0]/psi_data.shape[0])
            psi_data.loc[final_data[col] < 0, col] = np.nan
    except: 
        pass


for col in ['marry_info_date','vehicle_minput_lastreleasedate','prof_title_info_date']:  # 去除异常的时间
    try:
        psi_data.loc[psi_data[col] >= '2030-01-01', col] = np.nan
    except:
        pass


def date_cal(x, app_applydate):  # 计算申请日期距离其他日期的天数
    days_dt = pd.to_datetime(app_applydate) - pd.to_datetime(x)
    return days_dt.dt.days


for col in ['marry_info_date','vehicle_minput_lastreleasedate','prof_title_info_date']:
    if col != 'app_applydate':
        try:
            if col != 'vehicle_minput_drivinglicensevaliditydate':
                psi_data[col] = date_cal(psi_data[col], psi_data['app_applydate'])
                psi_data.loc[psi_data[col] < 0, col] = np.nan
            else:
                psi_data[col] = date_cal(psi_data['app_applydate'], psi_data[col])  # 计算行驶证有效期限距离申请日期的天数
        except:
            pass


## 统计变量的均值等
xx_data_s=psi_data[columns+['app_applydate']].copy()
xx_data_s['applymonth']=xx_data_s['app_applydate'].str.slice(0,7)
vars_mean_count=xx_data_s.groupby(xx_data_s['applymonth'])[xx_data_s.columns.tolist()].mean().T.reset_index().rename(columns={'index':'var_name'})
vars_mean_count=pd.merge(vars_mean_count,choose_columns_table[['var_name','var_desc']],on='var_name',how='left')
vars_mean_count.set_index('var_name').loc[columns,:].reset_index().to_excel('所有入模变量的申请数据均值分布_20190514v1.xlsx',index=False)

## 统计缺失率
de_dict_var=de_dict_var.loc[de_dict_var.var_name.map(lambda x: x in xx_data_s.columns.tolist()),:]
de_dict_var['null_percents'] = de_dict_var.var_name.map(lambda x: xx_data_s[x].isnull().sum() / xx_data_s.shape[0])  ##变量的总体缺失率
de_dict_var['different_values'] = de_dict_var.var_name.map(lambda x: len(xx_data_s[x].unique()))  # 计算变量取值个数，包括缺失值
de_dict_var['whether_null'] = (de_dict_var['null_percents'] > 0).astype('int64')  # 变量是否缺失
de_dict_var['different_values'] = de_dict_var['different_values'] - de_dict_var['whether_null']  # 计算除了缺失值后的变量取值个数

applymonth = xx_data_s.app_applydate.str.slice(0, 7)  # 取申请时间的月份
whether_null_matrix = xx_data_s.isnull().astype('float64').groupby(applymonth)[xx_data_s.columns.tolist()].mean()  # 按月统计变量的缺失率
whether_null_matrix = whether_null_matrix.T
whether_null_matrix['mean_null_percents'] = whether_null_matrix.mean(axis=1)  # 按月平均缺失率
whether_null_matrix['null_percents_std/mean'] = whether_null_matrix.std(axis=1) / whether_null_matrix.mean(axis=1)  # 按月缺失率标准差与均值的比值
whether_null_matrix = whether_null_matrix.reset_index().rename(columns={'index': 'var_name'})

de_dict_vars = pd.merge(de_dict_var, whether_null_matrix, on='var_name', how='inner')  # 变量评估，后台根据变量的质量指标进行手动清洗变量
de_dict_vars.to_excel('所有入选变量的申请数据缺失率情况_20190514v1.xlsx',index=False)
del de_dict_var,applymonth,whether_null_matrix,de_dict_vars


## 新老接口默认值设置
xx_data_s=psi_data[columns+['app_applydate']].copy()
xx_data_s['applymonth']=xx_data_s['app_applydate'].str.slice(0,7)
xx_data_s.loc[(xx_data_s.app_applydate>='2018-12-10') & (xx_data_s['jxl_id_comb_othertel_num']==0),'jxl_id_comb_othertel_num']=np.nan

model_data_s = xx_data_s[columns].copy()
model_data_s[model_data_s.isnull()] = -998
p = clf.predict_proba(model_data_s)[:, 1]

## 获取所有申请押证样本的新增二期分数
psi_data['new_scores2']=p*100
model_data_s[['app_applydate','app_applycode','contractno','new_scores2','last_update_date','app_callpoint','hkfs','dkfs','loantype']]=\
psi_data[['app_applydate','app_applycode','contractno','new_scores2','last_update_date','app_callpoint','hkfs','dkfs','loantype_y']].copy()

## 计算建模期全部申请样本客户各分数段占比
temp_1=model_data_s[(model_data_s.app_applydate >= '2017-08-01 00:00:00') & (model_data_s.app_applydate <=  '2018-12-10 23:59:59')].copy()
temp_1=temp_1.loc[(temp_1.dkfs.isin(['押证'])) & (temp_1.hkfs.isin(['2','4']))]
temp_1=temp_1[temp_1.loantype.isin([4,'4'])==False]
temp_1=pd.merge(temp_1,final_data[['app_applycode','y','loan_time']],on='app_applycode',how='left')
temp_1.loc[temp_1.y.isnull(),'y']=0
#占比
temp_1['grp'] = pd.cut(temp_1['new_scores2'], cuts, right=False)
score_dist  = temp_1.groupby('grp').grp.count()
rev = sorted(list(np.arange(score_dist.shape[0])), reverse=True)
print(score_dist[rev])
score_dist = score_dist[rev]/score_dist[rev].sum()
print(score_dist)
#区分能力
r_p_chart2(temp_1.y, temp_1.new_scores2.values / 100, min_scores, part=20)
del temp_1

##按月份统计-时间外申请样本客户各分数段占比
temp_2=model_data_s.copy()
temp_2=temp_2.loc[(temp_2.dkfs.isin(['押证'])) & (temp_2.hkfs.isin(['2','4']))]
temp_2=temp_2[temp_2.loantype.isin([4,'4'])==False]
apply_time_data_2 = temp_2[(temp_2.app_applydate >= '2018-08-01 00:00:00') & (temp_2.app_applydate <=  '2018-12-10 23:59:59')].copy()
apply_time_data_2=apply_time_data_2[((apply_time_data_2.app_applydate >= '2018-10-08 00:00:00') & (apply_time_data_2.app_applydate <=  '2018-10-09 23:59:59'))==False].copy()
apply_time_data_2=apply_time_data_2[apply_time_data_2.app_callpoint.isin(['APP0'])==False]
apply_time_data_2['applymonth']=apply_time_data_2['app_applydate'].str.slice(0,7)
apply_time_data_2['grp']=pd.cut(apply_time_data_2['new_scores2'],cuts,right=False)
applynum_bymonth=pd.crosstab(apply_time_data_2['grp'],apply_time_data_2['applymonth'],margins=True)
applynum_bymonth.reset_index().to_csv('时间外样本各分数段客户数_20190523v1.csv',index=False)
applynum_bymonth=applynum_bymonth.div(applynum_bymonth.ix['All',:],axis=1).reset_index()
applynum_bymonth.to_csv('时间外样本各分数段占比_20190523v1.csv',index=False)
del temp_2,model_data_s

## 为客户打标签

psi_data['overdue_flag'] = (psi_data.maxoverduedays >= 16)
psi_data['bad'] = (psi_data.overdue_flag == True)  # 占比约为4.6%
psi_data['chargeoff_flag'] = (psi_data.maxoverduedays==0) & (psi_data.returnstatus.isin(['已结清']))  # 结清里面大概有75%没有逾期
psi_data['r6_good_flag'] = (psi_data.returnphases >= 6) & (psi_data.maxoverduedays==0)
psi_data['good'] = psi_data.chargeoff_flag | psi_data.r6_good_flag
psi_data['y1'] = np.nan
psi_data.loc[psi_data.loandate.notnull(),'y1']=2
psi_data.loc[(psi_data.bad == True) & (psi_data.good == False), 'y1'] = 1
psi_data.loc[(psi_data.bad == False) & (psi_data.good == True), 'y1'] = 0

#占比
temp_3=psi_data[(psi_data.app_applydate >= '2018-08-01 00:00:00') & (psi_data.app_applydate <=  '2018-10-31 23:59:59')].copy()
temp_3=temp_3.loc[(temp_3.dkfs.isin(['押证'])) & (temp_3.hkfs.isin(['2','4']))]
temp_3=temp_3[temp_3.y1.isin([0,1])]
temp_3['grp'] = pd.cut(temp_3['new_scores2'], cuts, right=False)
score_dist  = temp_3.groupby('grp').grp.count()
rev = sorted(list(np.arange(score_dist.shape[0])), reverse=True)
print(score_dist[rev])
score_dist = score_dist[rev]/score_dist[rev].sum()
print(score_dist)
#区分能力
r_p_chart2(temp_3.y1, temp_3.new_scores2.values / 100, min_scores, part=20)
del temp_3





'''
为到表现期(固定6个表现期）的押证放款样本打客户标签，主要用于本模型的时间外验证 
'''

## 计算历史最大逾期天数
paymentdata0 = pd.read_table('payment_20190226.txt', sep='\u0001') #导入还款表)
paymentdata = paymentdata0[paymentdata0.loandate >= '2017-05-16 00:00:00'].copy()
paymentdata=paymentdata[(paymentdata.payphases<=6)].copy()
del paymentdata0

shouldpaydate=paymentdata.groupby(['contractno']).shouldpaydate.agg(['min','max']).reset_index().reset_index().rename(columns={'min':'shouldpaydate_0','max':'shouldpaydate_6'})
shouldpaydate['performence_day']=pd.to_datetime(shouldpaydate['shouldpaydate_6'])+datetime.timedelta(days=16)
paymentdata=pd.merge(paymentdata,shouldpaydate[['contractno','performence_day']],on='contractno',how='inner')

late_index=(pd.to_datetime(paymentdata.paydate) >= paymentdata['performence_day']) & (pd.to_datetime(paymentdata.shouldpaydate) < paymentdata['performence_day'])
paymentdata.loc[late_index, 'paydate']=paymentdata.loc[late_index, 'performence_day']  ## 将表现窗口设为截止到2018-03-31
paymentdata.loc[pd.to_datetime(paymentdata.shouldpaydate) >= paymentdata['performence_day'], 'paydate'] = paymentdata.loc[pd.to_datetime(paymentdata.shouldpaydate) >= paymentdata['performence_day'], 'shouldpaydate']

paymentdata=paymentdata[pd.to_datetime(paymentdata.paydate)>=pd.to_datetime('1970-12-03 00:00:00')].copy() #异常值处理
paymentdata['overdue_days'] = (pd.to_datetime(paymentdata.paydate) - pd.to_datetime(paymentdata.shouldpaydate)).dt.days
paymentdata.loc[(paymentdata[['shouldcapital', 'shouldgpsf', 'shouldint', 'shouldmanage']] == 0).all(axis=1), 'overdue_days'] = 0  ##计算历史最大逾期天数
paymentdata.loc[paymentdata['overdue_days'] < 0, 'overdue_days'] = 0
paymentdata_maxdue = paymentdata.groupby(['contractno']).overdue_days.max().reset_index().rename(columns={'overdue_days': 'maxoverduedays'})
del paymentdata

paymentdata_maxdue['overdue_flag'] = (paymentdata_maxdue.maxoverduedays >= 16)
paymentdata_maxdue['bad'] = (paymentdata_maxdue.overdue_flag == True)  # 占比约为4.6%
paymentdata_maxdue['chargeoff_flag'] = (paymentdata_maxdue.maxoverduedays == 0)
paymentdata_maxdue['good'] = paymentdata_maxdue.chargeoff_flag
paymentdata_maxdue['y'] = np.nan
paymentdata_maxdue.loc[(paymentdata_maxdue.bad == True) & (paymentdata_maxdue.good == False), 'y'] = 1
paymentdata_maxdue.loc[(paymentdata_maxdue.bad == False) & (paymentdata_maxdue.good == True), 'y'] = 0

outof_test1= pd.merge(temp_1[['app_applycode','app_applydate','contractno','newcar_model2_notcutof_score','last_update_date','hkfs','loan_time']], paymentdata_maxdue, on='contractno', how='inner')
outof_test1=outof_test1.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')


##计算某一个时间段的新增押证放款样本的模型区分能力

start_date = '2018-04-16 00:00:00'
end_date = '2018-06-20 23:59:59'
fin_time_data = outof_test[(outof_test.app_applydate >= start_date) & (outof_test.app_applydate <= end_date)]
fin_time_data=fin_time_data[fin_time_data.y.notnull()]
fpr, tpr, th = roc_curve(fin_time_data['y'], fin_time_data.new_scores2.values/100)
ks4 = tpr - fpr
print('all ks:   ' + str(max(ks4)))
print(roc_auc_score(fin_time_data['y'], fin_time_data.new_scores2.values/100))
r_p_chart2(fin_time_data['y'], fin_time_data.new_scores2.values/100, min_scores, part=20)


## 按月统计模型ks值,份12期先息后本以及非12期先息后本
start_date = '2017-05-16 00:00:00'
end_date = '2018-06-20 23:59:59'
fin_time_data = outof_test[(outof_test.app_applydate >= start_date) & (outof_test.app_applydate <= end_date)]
fin_time_data['whether_12']=((fin_time_data.loan_time==12) & (fin_time_data.hkfs.isin(['1']))).astype('float64')
testdata=fin_time_data[fin_time_data['whether_12']==0].copy()
testdata['applymonth']=testdata['app_applydate'].str.slice(0,7)
testdata=testdata[testdata['y'].isin([0,1])]
num_counts=pd.crosstab(testdata['applymonth'],testdata['y'],margins=True).reset_index()
ks_list={}
gini_list={}
for cn in testdata['applymonth'].unique():
    temp=testdata.ix[testdata['applymonth']==cn,['y','new_scores2']].copy()
    fpr, tpr, th = roc_curve(temp['y'], temp['new_scores2'].values/100)
    ks2 = tpr - fpr
    ks_list[cn]=max(ks2)
    try:
        gini_cn=2*roc_auc_score(temp['y'], temp['new_scores2'].values/100)-1
    except:
        gini_cn=np.nan
    gini_list[cn]=gini_cn

ks_pd=pd.Series(ks_list)
ks_pd=ks_pd.reset_index()

gini_pd=pd.Series(gini_list)
gini_pd=gini_pd.reset_index()

fpr, tpr, th = roc_curve(testdata['y'], testdata['new_scores2'].values / 100)
ks2 = tpr - fpr
print(max(ks2))
print(roc_auc_score(testdata['y'], testdata['new_scores2'].values/100))
r_p_chart2(testdata['y'], testdata['new_scores2'].values/100, min_scores, part=20)



'''
统计表现期时间截止日期（2019年2月11日）的模型表现
'''
paymentdata = pd.read_table('payment_20190212.txt', delimiter='\u0001')
paymentdata= paymentdata.replace('\\N', np.nan)

paymentdata=paymentdata.ix[paymentdata.shouldpaydate < '2019-02-12 00:00:00',['contractno','paydate','shouldpaydate','shouldcapital', 'shouldgpsf', 'shouldint', 'shouldmanage','update_time','totalphases','payphases','phases']]
paymentdata.loc[(paymentdata.paydate >= '2019-02-12 00:00:00'), 'paydate'] = '2019-02-11 23:59:59'  ## 将表现窗口设为截止到2019-01-06

paymentdata=paymentdata[paymentdata.paydate>='1970-12-03 00:00:00'].copy() #异常值处理
paymentdata['overdue_days'] = (pd.to_datetime(paymentdata.paydate) - pd.to_datetime(paymentdata.shouldpaydate)).dt.days
paymentdata.loc[(paymentdata[['shouldcapital', 'shouldgpsf', 'shouldint', 'shouldmanage']] == 0).all(axis=1), 'overdue_days'] = 0  ##计算历史最大逾期天数
paymentdata.loc[paymentdata['overdue_days'] < 0, 'overdue_days'] = 0
paymentdata_maxdue = paymentdata.groupby(['contractno']).overdue_days.max().reset_index().rename(columns={'overdue_days': 'maxoverduedays'})

paymentdata[['totalphases', 'payphases', 'phases']] = paymentdata[['totalphases', 'payphases', 'phases']].astype('int64')  # 将一些字段转成整型数据

paymentdata_totalphases = paymentdata.groupby(['contractno']).totalphases.max().reset_index()  # 计算贷款总期限,不包括展期
paymentdata_realtotalphases = paymentdata[paymentdata.update_time< '2019-02-12 00:00:00'].groupby(['contractno']).payphases.max().reset_index().rename(columns={'payphases': 'max_payphases'})  # 包括是否展期
paymentdata_totalphases = pd.merge(paymentdata_totalphases, paymentdata_realtotalphases, on='contractno', how='inner')
paymentdata_totalphases['realtotalphases'] = paymentdata_totalphases[['totalphases', 'max_payphases']].max(axis=1)  # 在实际贷款期限与是否展期合并获得总贷款期限

paymentdata_returnphases = paymentdata.groupby(['contractno']).payphases.max().reset_index().rename(columns={'payphases': 'returnphases'})  # 计算已还期数
paymentdata_phases_counts = pd.merge(paymentdata_returnphases, paymentdata_totalphases, on='contractno',how='inner')  # 合并贷款期限与已还期数
paymentdata_phases_counts = pd.merge(paymentdata_phases_counts, paymentdata_maxdue, on='contractno',how='inner')  # 合并最大逾期与贷款期限
del paymentdata,paymentdata_totalphases,paymentdata_returnphases,paymentdata_maxdue

findata0 = pd.read_table('fnsche_over20190211.txt',encoding='gbk',dtype={'contractno': str})#导入进度逾期表
findata = findata0.ix[ findata0['loandate'].notnull(), ['loandate', 'contractno', 'currentduedays', 'returnstatus']].copy()
findata['loandate'] = findata['loandate'].map(lambda x: datetime.datetime.strptime(str(x), "%d%b%Y")).copy()  # pd.to_datetime(findata0['loandate']).astype('str')
findata = findata[(findata['loandate'] >= '2017-05-16 00:00:00')].copy()  # .rename(columns={'ContractNo': 'contractno'})
findata = findata.sort_values(['contractno', 'loandate']).drop_duplicates('contractno', keep='last')
findata = pd.merge(findata, paymentdata_phases_counts, on='contractno', how='left')
fin_data = pd.merge(temp_1[['app_applycode','app_applydate','contractno','new_scores2','last_update_date','hkfs','loan_time']], findata, on='contractno', how='inner')
del findata0

fin_data['overdue_flag'] = (fin_data.maxoverduedays >= 16)
fin_data['bad'] = (fin_data.overdue_flag == True)  # 占比约为4.6%
fin_data['chargeoff_flag'] = (fin_data.maxoverduedays == 0) & (fin_data.returnstatus.isin(['已结清']))  # 结清里面大概有75%没有逾期
fin_data['r6_good_flag'] = (fin_data.returnphases >= 8) & (fin_data.maxoverduedays == 0)
fin_data['good'] = fin_data.chargeoff_flag | fin_data.r6_good_flag
fin_data['y'] = 2
fin_data.loc[(fin_data.bad == True) & (fin_data.good == False), 'y'] = 1
fin_data.loc[(fin_data.bad == False) & (fin_data.good == True), 'y'] = 0

## 按月统计模型ks值,份12期先息后本以及非12期先息后本
start_date = '2018-04-16 00:00:00'
end_date = '2018-07-20 23:59:59'
fin_time_data = fin_data[(fin_data.app_applydate >= start_date) & (fin_data.app_applydate <= end_date)]
testdata=fin_time_data[(fin_time_data.loan_time==12) & (fin_time_data.hkfs.isin(['1']))]
testdata['applymonth']=testdata['app_applydate'].str.slice(0,7)
testdata=testdata[testdata['y'].isin([0,1])]
num_counts=pd.crosstab(testdata['applymonth'],testdata['y'],margins=True).reset_index()
ks_list={}
for cn in testdata['applymonth'].unique():
    temp=testdata.ix[testdata['applymonth']==cn,['y','new_scores2']].copy()
    fpr, tpr, th = roc_curve(temp['y'], temp['new_scores2'].values/100)
    ks2 = tpr - fpr
    ks_list[cn]=max(ks2)

ks_pd=pd.Series(ks_list)
ks_pd=ks_pd.reset_index()

print(roc_auc_score(testdata['y'], testdata['new_scores2'].values/100))
r_p_chart2(testdata['y'], testdata['new_scores2'].values/100, min_scores, part=20)