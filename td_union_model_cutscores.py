# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:32:50 2019

@author: wj56740
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
my_data.to_excel('申请评分模型建模样本集—车贷决策引擎变量_20190722.xlsx',index=False)

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

## 附加同盾联合建模数据-评分
touna_cd_score_for=pd.read_csv('touna_cd_score_for.csv')
touna_cd_score_for['applydate']=touna_cd_score_for['apply_date'].str.slice(0,10)
final_data=pd.merge(final_data,touna_cd_score_for,on=['credentials_no_md5', 'cust_name_md5', 'mobile_md5', 'applydate'],how='inner')
final_data=final_data.drop_duplicates(['credentials_no_md5', 'cust_name_md5', 'mobile_md5', 'applydate'])
columns_transform={'通用分':'rational_score','小额现金贷多期分':'small_creditmoney_multiplyterm_score','小额现金贷单期分':'small_creditmoney_singleterm_score',
                   '银行分':'bank_score','消费金融分':'consumerloan_score','大额现金贷分':'big_creditmoney_singleterm_score'}
final_data=final_data.rename(columns=columns_transform)
#final_data.to_excel('所有建模样本数据_20190802.xlsx',index=False)

## 整合百融的联合建模数据
smxx_test=pd.read_excel('./百融详细匹配数据/01实名信息验证.xlsx',sheetname='实名信息验证匹配数据')
smxx_test=smxx_test.replace('\\N',np.nan)
smxx_test_columns_nullpercents=smxx_test.isnull().sum()/smxx_test.shape[0]
smxx_test=smxx_test[smxx_test_columns_nullpercents.loc[smxx_test_columns_nullpercents<0.95].index.tolist()]

spresult_test=pd.read_excel('./百融详细匹配数据/1.投哪儿_20190801_审批结果.xlsx',sheetname='审批结果匹配数据')
spresult_test=spresult_test.replace('\\N',np.nan)
spresult_test_columns_nullpercents=spresult_test.isnull().sum()/spresult_test.shape[0]
spresult_test=spresult_test[spresult_test_columns_nullpercents.loc[spresult_test_columns_nullpercents<0.95].index.tolist()]
spresult_test=spresult_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(smxx_test,spresult_test,on='cus_num',how='inner')
del smxx_test,spresult_test
gc.collect()

spresult_test=pd.read_excel('./百融详细匹配数据/1.投哪儿_20190806_审批结果.xlsx',sheetname='审批结果匹配数据')
spresult_test=spresult_test.replace('\\N',np.nan)
spresult_test_columns_nullpercents=spresult_test.isnull().sum()/spresult_test.shape[0]
spresult_test=spresult_test[spresult_test_columns_nullpercents.loc[spresult_test_columns_nullpercents<0.95].index.tolist()]
spresult_test=spresult_test.drop(['id', 'name', 'cell', 'user_date', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult_test,on='cus_num',how='inner')
del spresult_test
gc.collect()

spresult2_test=pd.read_excel('./百融详细匹配数据/02借贷意向验证.xlsx',sheetname='借贷意向验证匹配数据')
spresult2_test=spresult2_test.replace('\\N',np.nan)
spresult2_test_columns_nullpercents=spresult2_test.isnull().sum()/spresult2_test.shape[0]
spresult2_test=spresult2_test[spresult2_test_columns_nullpercents.loc[spresult2_test_columns_nullpercents<0.95].index.tolist()]
spresult2_test=spresult2_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult2_test,on='cus_num',how='inner')
del spresult2_test
gc.collect()

spresult3_test=pd.read_excel('./百融详细匹配数据/04借贷意向验证—月度版.xlsx',sheetname='借贷意向验证—月度版匹配数据')
spresult3_test=spresult3_test.replace('\\N',np.nan)
spresult3_test_columns_nullpercents=spresult3_test.isnull().sum()/spresult3_test.shape[0]
spresult3_test=spresult3_test[spresult3_test_columns_nullpercents.loc[spresult3_test_columns_nullpercents<0.95].index.tolist()]
spresult3_test=spresult3_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult3_test,on='cus_num',how='inner')
del spresult3_test
gc.collect()

spresult4_test=pd.read_excel('./百融详细匹配数据/05借贷意向衍生特征.xlsx',sheetname='借贷意向衍生特征匹配数据')
spresult4_test=spresult4_test.replace('\\N',np.nan)
spresult4_test_columns_nullpercents=spresult4_test.isnull().sum()/spresult4_test.shape[0]
spresult4_test=spresult4_test[spresult4_test_columns_nullpercents.loc[spresult4_test_columns_nullpercents<0.95].index.tolist()]
spresult4_test=spresult4_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult4_test,on='cus_num',how='inner')
del spresult4_test
gc.collect()

spresult5_test=pd.read_excel('./百融详细匹配数据/06商品消费衍生.xlsx',sheetname='商品消费衍生产品匹配数据')
spresult5_test=spresult5_test.replace('\\N',np.nan)
spresult5_test_columns_nullpercents=spresult5_test.isnull().sum()/spresult5_test.shape[0]
spresult5_test=spresult5_test[spresult5_test_columns_nullpercents.loc[spresult5_test_columns_nullpercents<0.95].index.tolist()]
spresult5_test=spresult5_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult5_test,on='cus_num',how='inner')
del spresult5_test
gc.collect()

spresult6_test=pd.read_excel('./百融详细匹配数据/07特殊名单验证.xlsx',sheetname='特殊名单验证匹配数据')
spresult6_test=spresult6_test.replace('\\N',np.nan)
spresult6_test_columns_nullpercents=spresult6_test.isnull().sum()/spresult6_test.shape[0]
spresult6_test=spresult6_test[spresult6_test_columns_nullpercents.loc[spresult6_test_columns_nullpercents<0.95].index.tolist()]
spresult6_test=spresult6_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult6_test,on='cus_num',how='inner')
del spresult6_test
gc.collect()

spresult7_test=pd.read_excel('./百融详细匹配数据/08借贷意向验证—当日版.xlsx',sheetname='借贷意向验证—当日版匹配数据')
spresult7_test=spresult7_test.replace('\\N',np.nan)
spresult7_test_columns_nullpercents=spresult7_test.isnull().sum()/spresult7_test.shape[0]
spresult7_test=spresult7_test[spresult7_test_columns_nullpercents.loc[spresult7_test_columns_nullpercents<0.95].index.tolist()]
spresult7_test=spresult7_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult7_test,on='cus_num',how='inner')
del spresult7_test
gc.collect()

spresult8_test=pd.read_excel('./百融详细匹配数据/09地址信息验证.xlsx',sheetname='地址信息验证匹配数据')
spresult8_test=spresult8_test.replace('\\N',np.nan)
spresult8_test_columns_nullpercents=spresult8_test.isnull().sum()/spresult8_test.shape[0]
spresult8_test=spresult8_test[spresult8_test_columns_nullpercents.loc[spresult8_test_columns_nullpercents<0.95].index.tolist()]
spresult8_test=spresult8_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult8_test,on='cus_num',how='inner')
del spresult8_test
gc.collect()

spresult9_test=pd.read_excel('./百融详细匹配数据/011团伙欺诈排查通用版.xlsx',sheetname='团伙欺诈排查通用版匹配数据')
spresult9_test=spresult9_test.replace('\\N',np.nan)
spresult9_test_columns_nullpercents=spresult9_test.isnull().sum()/spresult9_test.shape[0]
spresult9_test=spresult9_test[spresult9_test_columns_nullpercents.loc[spresult9_test_columns_nullpercents<0.95].index.tolist()]
spresult9_test=spresult9_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult9_test,on='cus_num',how='inner')
del spresult9_test
gc.collect()

spresult10_test=pd.read_excel('./百融详细匹配数据/012高风险借贷意向验证.xlsx',sheetname='高风险借贷意向验证匹配数据')
spresult10_test=spresult10_test.replace('\\N',np.nan)
spresult10_test_columns_nullpercents=spresult10_test.isnull().sum()/spresult10_test.shape[0]
spresult10_test=spresult10_test[spresult10_test_columns_nullpercents.loc[spresult10_test_columns_nullpercents<0.95].index.tolist()]
spresult10_test=spresult10_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult10_test,on='cus_num',how='inner')
del spresult10_test
gc.collect()

spresult11_test=pd.read_excel('./百融详细匹配数据/013消费指数.xlsx',sheetname='消费指数匹配数据')
spresult11_test=spresult11_test.replace('\\N',np.nan)
spresult11_test_columns_nullpercents=spresult11_test.isnull().sum()/spresult11_test.shape[0]
spresult11_test=spresult11_test[spresult11_test_columns_nullpercents.loc[spresult11_test_columns_nullpercents<0.95].index.tolist()]
spresult11_test=spresult11_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult11_test,on='cus_num',how='inner')
del spresult11_test
gc.collect()

spresult12_test=pd.read_excel('./百融详细匹配数据/014商品消费指数.xlsx',sheetname='商品消费指数匹配数据')
spresult12_test=spresult12_test.replace('\\N',np.nan)
spresult12_test_columns_nullpercents=spresult12_test.isnull().sum()/spresult12_test.shape[0]
spresult12_test=spresult12_test[spresult12_test_columns_nullpercents.loc[spresult12_test_columns_nullpercents<0.95].index.tolist()]
spresult12_test=spresult12_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult12_test,on='cus_num',how='inner')
del spresult12_test
gc.collect()


spresult13_test=pd.read_excel('./百融详细匹配数据/015稳定性指数.xlsx',sheetname='稳定性指数匹配数据')
spresult13_test=spresult13_test.replace('\\N',np.nan)
spresult13_test_columns_nullpercents=spresult13_test.isnull().sum()/spresult13_test.shape[0]
spresult13_test=spresult13_test[spresult13_test_columns_nullpercents.loc[spresult13_test_columns_nullpercents<0.95].index.tolist()]
spresult13_test=spresult13_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult13_test,on='cus_num',how='inner')
del spresult13_test
gc.collect()


spresult14_test=pd.read_excel('./百融详细匹配数据/016媒体阅览指数.xlsx',sheetname='媒体阅览指数匹配数据')
spresult14_test=spresult14_test.replace('\\N',np.nan)
spresult14_test_columns_nullpercents=spresult14_test.isnull().sum()/spresult14_test.shape[0]
spresult14_test=spresult14_test[spresult14_test_columns_nullpercents.loc[spresult14_test_columns_nullpercents<0.95].index.tolist()]
spresult14_test=spresult14_test.drop(['id', 'name', 'cell', 'user_time', 'swift_number'],axis=1)
bair_data=pd.merge(bair_data,spresult14_test,on='cus_num',how='inner')
del spresult14_test
gc.collect()
#bair_data.to_excel('bair_data_20190807.xlsx',index=False)
bair_data=bair_data[(bair_data.user_time>='2018-03-01 00:00:00') & (bair_data.user_time<='2018-11-30 23:59:59')]
bair_data['user_time_day']=bair_data['user_time'].map(str).str.slice(0,10)
combine_data=pd.merge(final_data,bair_data,left_on=['credentials_no_md5','mobile_md5','applydate'],right_on=
                     ['id','cell','user_time_day'],how='inner')
combine_data=combine_data.drop_duplicates(['credentials_no_md5', 'mobile_md5', 'applydate'])
#combine_data.to_excel('百融+同盾+自有建模数据整合_20190808.xlsx',index=False)
del final_data,bair_data
gc.collect()


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

tn_dict_8=pd.read_excel('百融联合建模接口文档.xlsx')
tn_dict_8=tn_dict_8[['var_desc','var_type','length','var_name','var_code_comment','default','数据源']].drop_duplicates()


tn_dict = tn_dict_1.append(tn_dict_2).append(tn_dict_3).append(tn_dict_4).append(tn_dict_5).append(tn_dict_6).append(tn_dict_7)
tn_dict=tn_dict.append(tn_dict_8)
tn_dict = tn_dict.drop_duplicates('var_name')

tn_dict['var_name'] = tn_dict.var_name.str.lower()
combine_data.columns=[cn.lower() for cn in combine_data.columns]
de_dict = pd.merge(tn_dict, pd.DataFrame(combine_data.columns.tolist()), left_on='var_name', right_on=0, how='inner').drop(0,axis=1)
del tn_dict_1,tn_dict_2,tn_dict_3,tn_dict_4,tn_dict,tn_dict_5,tn_dict_6,tn_dict_7,tn_dict_8
#de_dict.to_excel('D:/llm/联合建模算法开发/百融联合建模_逻辑回归/DE_Var_Select_20190808.xlsx',index=False)


'''
## 2.1 变量预处理---对变量进行质量评估，手动进行数据清洗，包括剔除缺失率非常高的变量、单一值变量以及其他明显无法使用的变量
'''

de_dict_var = pd.read_excel('D:/llm/联合建模算法开发/百融联合建模_逻辑回归/DE_Var_Select_20190808.xlsx')
for i, _ in de_dict.iterrows():
    name = de_dict_var.loc[i, 'var_name']
    default = de_dict_var.loc[i, 'default']
    if default != '""' and name in set(combine_data.columns) and name!='app_applycode':
        try:
            combine_data[name] = combine_data[name].astype('float64')
            if (combine_data[name] == float(default)).sum() > 0:
                combine_data.loc[combine_data[name] == float(default), name] = np.nan
        except:
            pass

    elif default == '""' and name in set(combine_data.columns) and name!='app_applycode':
        try:
            combine_data[name] = combine_data[name].astype('float64')
            if (combine_data[name] == float(-99)).sum() > 0:
                combine_data.loc[combine_data[name] == float(-99), name] = np.nan
            if (combine_data[name] == '-99').sum() > 0:
                combine_data.loc[combine_data[name] == '-99', name] = np.nan
        except:
            pass

## 统计各变量的质量情况,包括变量的总体缺失率、取不同值个数、按月统计变量的缺失率等

de_dict_var['null_percents'] = de_dict_var.var_name.map(lambda x: combine_data[x].isnull().sum() / combine_data.shape[0])  ##变量的总体缺失率
de_dict_var['different_values'] = de_dict_var.var_name.map(lambda x: len(combine_data[x].unique()))  # 计算变量取值个数，包括缺失值
de_dict_var['whether_null'] = (de_dict_var['null_percents'] > 0).astype('int64')  # 变量是否缺失
de_dict_var['different_values'] = de_dict_var['different_values'] - de_dict_var['whether_null']  # 计算除了缺失值后的变量取值个数

applymonth = combine_data.app_applydate.str.slice(0, 7)  # 取申请时间的月份
whether_null_matrix = combine_data.isnull().astype('float64').groupby(applymonth)[combine_data.columns.tolist()].mean()  # 按月统计变量的缺失率
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
#de_dict_vars.to_excel('D:/llm/联合建模算法开发/百融联合建模_逻辑回归/de_dict_vars_20190808.xlsx',index=False)
del de_dict_vars



'''
##2.2 变量预处理--针对不同的数据类型进行预处理
#'''

vars_count_table=pd.read_excel('D:/llm/联合建模算法开发/百融联合建模_逻辑回归/de_dict_vars_20190808.xlsx') #手动增加了同盾外部评分
choose_columns_table = vars_count_table[vars_count_table['是否选用'].isin(['是'])]
numeric_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['\ufeff数据型','数据型', '数字型','数值型']), 'var_name'].values.tolist()
str_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['字符型', '\ufeff字符型','字符型 ' ]), 'var_name'].values.tolist()
date_columns = choose_columns_table.loc[choose_columns_table.var_type.isin(['\ufeff日期型','日期型']), 'var_name'].values.tolist()
print(len(numeric_columns)+len(str_columns)+len(date_columns))
combine_data = combine_data.replace('\\N',np.nan)



'''
## 4.变量筛选
'''


"""
## 4.1 初步筛选
等频分箱：
类别大于10的变量进行等频分箱，然后计算IV
类别小于10 的变量直接进行计算IV
"""
model_data_new=combine_data.loc[combine_data.app_applydate<='2018-10-31 23:59:59',choose_columns_table.var_name.tolist()+
                               ['apply_y','app_applydate']].copy()

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
    less_iv_name.to_excel("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/cat_less_iv_1.xlsx")
    df = v2_equif_bin(df, cat_over20, target)
    cols_name = [i+"_Bin" for i in cat_over20]
    over_iv_name = v2_cat_woe_iv(df, cols_name, target,0.02,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/")  # 计算woe 和IV值
    over_iv_name.to_excel("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/cat_more_iv_2.xlsx")
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
#init_choose_name_desc.to_excel('D:/llm/联合建模算法开发/百融联合建模_逻辑回归/init_choose_name_desc_20190808.xlsx',index=False)
init_choose_table=init_choose_name_desc.loc[init_choose_name_desc['是否选用'].isin(['是']),:]#手动去除一些不能用的变量
numeric_columns = init_choose_table.loc[init_choose_table.var_type.isin(['\ufeff数据型','数据型', '数字型','数值型']), 'var_name'].values.tolist()
str_columns = init_choose_table.loc[init_choose_table.var_type.isin(['字符型', '\ufeff字符型','字符型 ' ]), 'var_name'].values.tolist()
date_columns = init_choose_table.loc[init_choose_table.var_type.isin(['\ufeff日期型','日期型']), 'var_name'].values.tolist()
print(len(numeric_columns)+len(str_columns)+len(date_columns))

df_choose1=df[init_choose_table.var_name.tolist()+['apply_y']].copy()
df_choose1.to_excel('D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/df_choose1.xlsx',index=False)
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
               regroup.to_csv("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/regroup_Chi_cutbin__.csv", mode="a", header=True)
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
               regroup.to_csv("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/regroup_Chi_cutbin__.csv", mode="a", header=True)
               print(regroup)
               num_list.append(col)
    return df,num_list
df_choose1,num_list = v3_Chimerge(df_choose1,target,max_interval, num_var)
df_choose1.to_excel("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/cut_bin_train.xlsx")
print(df_choose1.columns.tolist())
print(df_choose1.shape[0])


# 计算woe 和IV
cols_list = [i+"_Bin" for i in num_list]
df_hig_iv = v2_cat_woe_iv(df_choose1,cols_list,target,0.02,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/")
df_hig_iv.to_excel("D:/llm/联合建模算法开发/百融联合建模_逻辑回归/high_iv_chi.xlsx")
df_choose1['als_d15_cell_oth_orgnum_Bin']=df_choose1['als_d15_cell_oth_orgnum_Bin'].replace({'b2[1.0,2.0)':'b2[1.0,inf)','b3[2.0,3.0)':'b2[1.0,inf)','b5[3.0,inf)':'b2[1.0,inf)'})
df_choose1['m12_apply_platform_cnt_Bin']=df_choose1['m12_apply_platform_cnt_Bin'].replace({'b0[-inf, 2.0)':'b0[-inf, 34.0)','b1[2.0,34.0)':'b0[-inf, 34.0)'})
df_choose1['vehicle_evtrpt_mileage_Bin']=df_choose1['vehicle_evtrpt_mileage_Bin'].replace({'b-1NA':'b0[-inf, 16663.0)&NA','b0[-inf, 16663.0)':'b0[-inf, 16663.0)&NA'})
df_choose1['contact_court_cnt_Bin']=df_choose1['contact_court_cnt_Bin'].replace({'b3[1.0,3.0)':'b3[1.0,inf)','b5[3.0,inf)':'b3[1.0,inf)'})
df_choose1['d1_id_relate_device_num_Bin']=df_choose1['d1_id_relate_device_num_Bin'].replace({'b2[0.0,1.0)':'b2[0.0,inf)','b4[1.0,inf)':'b2[0.0,inf)'})
df_hig_iv = v2_cat_woe_iv(df_choose1,cols_list+str_columns,target,0.02,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/")
df_hig_iv.to_excel("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/high_iv_chi.xlsx")

# 经卡方分箱完筛选后的最终变量
iv_name1=[cn.split('_Bin')[0] for cn in df_hig_iv.index.tolist()]
init_choose_name_desc=choose_columns_table.set_index('var_name').loc[iv_name1,:].reset_index()
init_choose_name_desc.to_excel('D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/init_choose_name_desc_20190809.xlsx',index=False)



"""
## 4.3 手动调整分箱,依据相关性进一步筛选变量
手动调节不满足单调性的变量
使其满足单调性
并计算IV
转换为woe值
时间外样本转换为woe值
连续变量的相关性分析
"""
init_choose_name_desc=pd.read_excel('D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/init_choose_name_desc_20190826.xlsx')
init_choose_table=init_choose_name_desc.loc[init_choose_name_desc['是否选用'].isin(['是']),:]#手动去除一些不能用的变量
init_choose_table=init_choose_table[init_choose_table['数据源'].isin(['百融联合建模'])==False]#去掉百融联合建模变量
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
          regroup.to_csv(address+"num_bin_badrate.csv",mode="a",header=True)
        except:
            error_cols.append(var)
    df.to_csv(address+"data_woe_value_num.csv",header=True) # 包含分类变量和连续变量
    return error_cols,df

error_cols,df_choose3=main(df_choose2,col,'D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/')

""" 相关性分析：大于0.5的变量进行筛选"""
#df_choose3= pd.read_csv("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/data_woe_value_num.csv")
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
df_high_cor1.to_excel("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/df_high_cor.xlsx")


## 相关性去除iv更低的变量
cat_less_iv=pd.read_excel('D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/cat_less_iv_1.xlsx')
cat_less_iv=cat_less_iv.set_index('var_name').loc[str_columns,'IV'].reset_index().rename(columns={'IV':'iv'})
df_hig_iv=pd.read_excel("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/high_iv_chi.xlsx")
df_hig_iv=pd.concat([df_hig_iv,cat_less_iv],axis=0)
df_hig_iv.to_csv('D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/df_hig_iv_all.csv',index=False)
df_high_cor1=pd.read_excel("D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/df_high_cor.xlsx")

del_cor=[]
for i in range(len(df_high_cor1)):
    if df_hig_iv.loc[df_hig_iv.var_name.isin([df_high_cor1.var_name1.values[i]]),'iv'].values[0]>=\
                    df_hig_iv.loc[df_hig_iv.var_name.isin([df_high_cor1.var_name2.values[i]]),'iv'].values[0]:
            del_cor.append(df_high_cor1.var_name2.values[i])
    else:
          del_cor.append(df_high_cor1.var_name1.values[i])
del_cor=list(set(del_cor))
remain_col=[cn for cn in col if cn not in del_cor] ##剔除一些最好不能用的变量
#剩44个变量

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

main(df_choose5,remain_col,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/final_test_num_bin_badrate1")
remain_col=['apt_ec_lastoverduedaystotal_Bin',
 'apt_ec_overduephasetotallastyear_Bin',
 'avg_sms_cnt_l6m_Bin',
 'call_cnt_rank2_Bin',
 'contact_bank_call_cnt_Bin',
 'contact_car_contact_afternoon_Bin',
 'contact_loan_contact_3m_Bin',
 'contact_unknown_contact_early_morning_Bin',
 'contacts_class1_cnt_Bin',
 'contacts_router_ratio_Bin',
 'i_cnt_grp_partner_loan_all_all_Bin',
 'i_cnt_node_dist1_mobile_loan_all_all_Bin',
 'i_mean_freq_node_seq_partner_loan_all_all_Bin',
 'i_pctl_cnt_ic_partner_loan_p2pweb_365day_Bin',
 'jxl_id_comb_othertel_num_Bin',
 'm3_id_relate_homeaddress_num_Bin',
 'm_cnt_node_dist1_risk_loan_all_all_Bin',
 'm_cnt_node_dist2_loan_all_all_Bin',
 'm_cnt_shoppingappinstlv2_gen_gen_365day_Bin',
 'm_length_first_register_all_180day_Bin',
 'm_mean_length_event_all_consumerfinance_180day_Bin',
 'm_mean_length_event_all_offloan_365day_Bin',
 'm_mean_length_event_register_unconsumerfinance_365day_Bin',
 'm_mom_cnt_90daypartner_loan_imbank_180day_Bin',
 'm_pctl_cnt_ic_partner_loan_unconsumerfinance_180day_Bin',
 'm_ratio_cnt_grp_max_partner_loan_all_all_Bin',
 'm_ratio_cnt_partner_all_imbank_365day_Bin',
 'm_ratio_cnt_partner_all_o2o_365day_Bin',
 'm_ratio_cnt_partner_loan_inconsumerfinance_365day_Bin',
 'm_ratio_freq_night_loan_all_365day_Bin',
 'm_ratio_freq_night_register_offloan_180day_Bin',
 'm_ratio_freq_record_register_all_180day_Bin',
 'm_ratio_freq_weekday_all_finance_365day_Bin',
 'max_call_in_cnt_l6m_Bin',
 'max_overdue_terms_Bin',
 'max_total_amount_l6m_Bin',
 'qtorg_query_orgcnt_Bin',
 'times_by_current_org_Bin',
 'cell_reg_time_Bin',
 'mate_mobile_info_date_Bin',
 'prof_title_info_date_Bin',
 'vehicle_minput_lastreleasedate_Bin',
 'apt_lastloanmode']


"""
## 4.4 进行交叉验证,进一步筛选变量
分箱之后：
调整单调性
调整分箱稳定性
"""

df_choose7=df_choose5[remain_col+['apply_y']].copy()
for cn in df_choose7.columns:
    print(cn,df_choose7[cn].unique())

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

## 迭代，使得不在剔除变量为止为停止准则，剩余35个变量
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
remain_col=['apt_ec_lastoverduedaystotal_Bin',
 'apt_ec_overduephasetotallastyear_Bin',
 'avg_sms_cnt_l6m_Bin',
 'contact_bank_call_cnt_Bin',
 'contact_car_contact_afternoon_Bin',
 'contact_unknown_contact_early_morning_Bin',
 'contacts_class1_cnt_Bin',
 'contacts_router_ratio_Bin',
 'i_cnt_grp_partner_loan_all_all_Bin',
 'i_mean_freq_node_seq_partner_loan_all_all_Bin',
 'i_pctl_cnt_ic_partner_loan_p2pweb_365day_Bin',
 'jxl_id_comb_othertel_num_Bin',
 'm3_id_relate_homeaddress_num_Bin',
 'm_cnt_node_dist2_loan_all_all_Bin',
 'm_cnt_shoppingappinstlv2_gen_gen_365day_Bin',
 'm_mean_length_event_register_unconsumerfinance_365day_Bin',
 'm_pctl_cnt_ic_partner_loan_unconsumerfinance_180day_Bin',
 'm_ratio_cnt_grp_max_partner_loan_all_all_Bin',
 'm_ratio_cnt_partner_all_imbank_365day_Bin',
 'm_ratio_cnt_partner_all_o2o_365day_Bin',
 'm_ratio_cnt_partner_loan_inconsumerfinance_365day_Bin',
 'm_ratio_freq_night_loan_all_365day_Bin',
 'm_ratio_freq_night_register_offloan_180day_Bin',
 'm_ratio_freq_record_register_all_180day_Bin',
 'm_ratio_freq_weekday_all_finance_365day_Bin',
 'max_call_in_cnt_l6m_Bin',
 'max_overdue_terms_Bin',
 'max_total_amount_l6m_Bin',
 'qtorg_query_orgcnt_Bin',
 'times_by_current_org_Bin',
 'cell_reg_time_Bin',
 'mate_mobile_info_date_Bin',
 'prof_title_info_date_Bin',
 'vehicle_minput_lastreleasedate_Bin',
 'apt_lastloanmode']

##lasso迭代，筛选变量,剩余34个
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
main(df_choose6,remain_col,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/final_test_num_bin_badrate2") 

temp2=remain_col
while 1:
  max_vif ,vif_df= Vif(df_choose7, remain_col)
  print(vif_df)
  if max_vif['vif'].values[0]<=3:
      break
  remain_col.remove(max_vif['var_name'].values[0])
vif_df.to_excel('D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/vif_df_20190814.xlsx')


df_choose7=df_choose6[remain_col+['apply_y']].copy()
x =  df_choose7[remain_col]
y = df_choose7["apply_y"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
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

df_hig_iv = v2_cat_woe_iv(df_choose7,final_columns,'apply_y',0.02,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/")
df_hig_iv=df_hig_iv.reset_index().rename(columns={'index':'var_name',0.0:'iv'}).sort_values('iv',ascending=False)


##最终剩下的变量
final_columns=['i_pctl_cnt_ic_partner_loan_p2pweb_365day_Bin',
 'm_cnt_node_dist2_loan_all_all_Bin',
 'm_cnt_shoppingappinstlv2_gen_gen_365day_Bin',
 'm_ratio_freq_record_register_all_180day_Bin',
 'i_mean_freq_node_seq_partner_loan_all_all_Bin',
 'm_ratio_cnt_grp_max_partner_loan_all_all_Bin',
 'contact_unknown_contact_early_morning_Bin',
 'm_ratio_cnt_partner_all_imbank_365day_Bin',
 'vehicle_minput_lastreleasedate_Bin',
 'apt_ec_lastoverduedaystotal_Bin',
 'jxl_id_comb_othertel_num_Bin',
 'qtorg_query_orgcnt_Bin',
 'cell_reg_time_Bin',
 'prof_title_info_date_Bin',
 'm_ratio_cnt_partner_all_o2o_365day_Bin',
 'apt_ec_overduephasetotallastyear_Bin',
 'contacts_router_ratio_Bin',
 'contact_bank_call_cnt_Bin',
 'apt_lastloanmode',
 'max_total_amount_l6m_Bin',
 'm3_id_relate_homeaddress_num_Bin',
 'avg_sms_cnt_l6m_Bin',
 'max_call_in_cnt_l6m_Bin',
 'times_by_current_org_Bin',
 'max_overdue_terms_Bin']

"""
## 
分箱之后：
调整单调性
调整分箱稳定性
"""
df_choose6=df_choose1[final_columns+['apply_y']].copy()
df_choose6['jxl_id_comb_othertel_num_Bin']=np.nan
df_choose6.loc[df_choose1['jxl_id_comb_othertel_num']<=0,'jxl_id_comb_othertel_num_Bin']='NA&=0'
df_choose6.loc[df_choose1['jxl_id_comb_othertel_num']>=1,'jxl_id_comb_othertel_num_Bin']='[1,inf)'
df_choose6['apt_ec_lastoverduedaystotal_Bin']=df_choose6['apt_ec_lastoverduedaystotal_Bin'].replace(
        {'b1[5.0,7.0)':'b1[5.0,inf)','b2[7.0,9.0)':'b1[5.0,inf)','b4[9.0,inf)':'b1[5.0,inf)','b-1NA':'NA&b0[-inf, 5.0)','b0[-inf, 5.0)':'NA&b0[-inf, 5.0)'})
df_choose6['apt_ec_overduephasetotallastyear_Bin']=df_choose6['apt_ec_overduephasetotallastyear_Bin'].replace(
        {'b1[1.0,2.0)':'b1[1.0,inf)','b2[2.0,3.0)':'b1[1.0,inf)','b3[3.0,4.0)':'b1[1.0,inf)','b4[4.0,inf)':'b1[1.0,inf)'})
df_choose6['apt_lastloanmode']=df_choose6['apt_lastloanmode'].astype('str').replace({'2.0':'NA&123','3.0':'NA&23','-99.0':'NA&23'})
df_choose6['avg_sms_cnt_l6m_Bin']=df_choose6['avg_sms_cnt_l6m_Bin'].replace({'b2[21.166666666666668,68.66666666666667)':'NA&>21.17','b4[68.66666666666667,inf)':'NA&>21.17','b-1NA':'NA&>21.17',
    'b0[-inf, 5.833333333333332)':'b0[-inf, 21.166666666666668)','b1[5.833333333333332,21.166666666666668)':'b0[-inf, 21.166666666666668)'})
df_choose6['contact_bank_call_cnt_Bin']=df_choose6['contact_bank_call_cnt_Bin'].replace({'b1[7.0,21.0)':'NA&b1[7.0,inf)',
    'b2[21.0,36.0)':'NA&b1[7.0,inf)','b4[36.0,inf)':'NA&b1[7.0,inf)','b-1NA':'NA&b1[7.0,inf)'})
df_choose6['max_call_in_cnt_l6m_Bin']=df_choose6['max_call_in_cnt_l6m_Bin'].replace({'b0[-inf, 134.0)':'NA&b0[-inf, 337.0)',
    'b1[134.0,337.0)':'NA&b0[-inf, 337.0)','b-1NA':'NA&b0[-inf, 337.0)'})
df_choose6['max_overdue_terms_Bin']=df_choose6['max_overdue_terms_Bin'].replace({'b2[1.0,2.0)':'b2[1.0,inf)',
    'b4[2.0,inf)':'b2[1.0,inf)'})    
df_choose6['prof_title_info_date_Bin']=df_choose6['prof_title_info_date_Bin'].replace({'b0[-inf, -29.0)':'b0[-inf, 145.0)',
    'b1[-29.0,-6.0)':'b0[-inf, 145.0)','b2[-6.0,145.0)':'b0[-inf, 145.0)'})        
df_choose6['times_by_current_org_Bin']=df_choose6['times_by_current_org_Bin'].replace({'b-1NA':'NA&b0[-inf, 2.0)',
    'b0[-inf, 2.0)':'NA&b0[-inf, 2.0)','b2[3.0,4.0)':'b2[2.0,inf)','b4[4.0,inf)':'b2[2.0,inf)','b1[2.0,3.0)':'b2[2.0,inf)'})
df_choose6['vehicle_minput_lastreleasedate_Bin']=df_choose6['vehicle_minput_lastreleasedate_Bin'].replace({'b-1NA':'NA&b4[164.0,inf)',
    'b4[164.0,inf)':'NA&b4[164.0,inf)','b0[-inf, 1.0)':'b2[6.0,164.0)&<1','b2[6.0,164.0)':'b2[6.0,164.0)&<1'})
df_choose6['contact_unknown_contact_early_morning_Bin']=df_choose6['contact_unknown_contact_early_morning_Bin'].replace({'b1[45.0,143.0)':'b1[45.0,inf)',
    'b2[143.0,253.0)':'b1[45.0,inf)','b4[253.0,inf)':'b1[45.0,inf)','b-1NA':'NA&b0[-inf, 45.0)','b0[-inf, 45.0)':'NA&b0[-inf, 45.0)'})
df_choose6['cell_reg_time_Bin']=df_choose6['cell_reg_time_Bin'].replace({'b1[923.0,3958.0)':'NA&b1[923.0,3958.0)',
    'b1[923.0,3958.0)':'NA&b1[923.0,3958.0)','b2[3958.0,5959.0)':'b2[3958.inf)','b4[5959.0,inf)':'b2[3958.inf)','b-1NA':'NA&b1[923.0,3958.0)'})
df_choose6['max_total_amount_l6m_Bin']=df_choose6['max_total_amount_l6m_Bin'].replace({'b2[298.8,462.91)':'NA&b2[298.8,inf)',
    'b-1NA':'NA&b2[298.8,inf)','b4[462.91,inf)':'NA&b2[298.8,inf)','b0[-inf, 110.39)':'b0[-inf, 298.8)','b1[110.39,298.8)':'b0[-inf, 298.8)'})
df_choose6['m3_id_relate_homeaddress_num_Bin']=df_choose6['m3_id_relate_homeaddress_num_Bin'].replace({'b2[1.0,3.0)':'b2[1.0,inf)',
    'b4[3.0,inf)':'b2[1.0,inf)'})
df_choose6['i_pctl_cnt_ic_partner_loan_p2pweb_365day_Bin']=df_choose6['i_pctl_cnt_ic_partner_loan_p2pweb_365day_Bin'].replace(
        {'b2[0.6106,0.6306)':'b2[0.3315,inf)','b4[0.6306,inf)':'b2[0.3315,inf)','b1[0.3315,0.6106)':'b2[0.3315,inf)','b-1NA':'NA&<0.3315',
      'b0[-inf, 0.3315)':'NA&<0.3315'})
df_choose6['m_cnt_node_dist2_loan_all_all_Bin']=df_choose6['m_cnt_node_dist2_loan_all_all_Bin'].replace(
        {'b0[-inf, 1.0)':'b0[-inf, 9.0)','b1[1.0,9.0)':'b0[-inf, 9.0)','b-1NA':'NA&>=9','b2[9.0,17.0)':'NA&>=9','b4[17.0,inf)':'NA&>=9'})
df_choose6['m_cnt_shoppingappinstlv2_gen_gen_365day_Bin']=df_choose6['m_cnt_shoppingappinstlv2_gen_gen_365day_Bin'].replace(
        {'b-1NA':'NA&>=8','b1[8.0,13.0)':'NA&>=8','b2[13.0,14.0)':'NA&>=8','b4[14.0,inf)':'NA&>=8'})
df_choose6['m_ratio_cnt_grp_max_partner_loan_all_all_Bin']=df_choose6['m_ratio_cnt_grp_max_partner_loan_all_all_Bin'].replace(
        {'b-1NA':'NA&<0.496','b0[-inf, 0.2736766294562478)':'NA&<0.496','b1[0.2736766294562478,0.4962264150943396)':'NA&<0.496',
      'b2[0.4962264150943396,0.6254)':'b2[0.496,inf)','b4[0.6254,inf)':'b2[0.496,inf)'}) 
df_choose6['m_ratio_cnt_partner_all_imbank_365day_Bin']=df_choose6['m_ratio_cnt_partner_all_imbank_365day_Bin'].replace(
        {'b-1NA':'NA&>=0.8333','b2[0.8333,0.9016)':'NA&>=0.8333','b4[0.9016,inf)':'NA&>=0.8333','b0[-inf, 0.5)':'b0[-inf, 0.8333)',
    'b1[0.5,0.8333)':'b0[-inf, 0.8333)'})
df_choose6['m_ratio_freq_record_register_all_180day_Bin']=df_choose6['m_ratio_freq_record_register_all_180day_Bin'].replace(
        {'b0[-inf, 0.0)':'b0[-inf, 0.0714)','b1[0.0,0.0714)':'b0[-inf, 0.0714)','b-1NA':'NA&>=0.0714','b2[0.0714,0.1154)':'NA&>=0.0714',
         'b4[0.1154,inf)':'NA&>=0.0714'})
df_choose6['i_mean_freq_node_seq_partner_loan_all_all_Bin']=df_choose6['i_mean_freq_node_seq_partner_loan_all_all_Bin'].replace(
        {'b0[-inf, 1.2)':'<1.2&>=1.9375','b4[1.9375,inf)':'<1.2&>=1.9375','b-1NA':'NA&b1[1.2,1.937)','b1[1.2,1.4444)':'NA&b1[1.2,1.937)',
         'b2[1.4444,1.9375)':'NA&b1[1.2,1.937)'})
df_choose6['qtorg_query_orgcnt_Bin']=df_choose6['qtorg_query_orgcnt_Bin'].replace({'b-1NA':'NA&<1','b1[0.0,1.0)':'NA&<1'})
df_choose6['m_ratio_cnt_partner_all_o2o_365day_Bin']=df_choose6['m_ratio_cnt_partner_all_o2o_365day_Bin'].replace(
        {'b-1NA':'NA&<0.0313','b0[-inf, 0.0313)':'NA&<0.0313','b1[0.0313,0.1429)':'b1[0.0313,inf)','b2[0.1429,0.3333)':'b1[0.0313,inf)',
         'b4[0.3333,inf)':'b1[0.0313,inf)'})
df_choose6['contacts_router_ratio_Bin']=df_choose6['contacts_router_ratio_Bin'].replace({'b-1NA':'NA&b0[-inf, 0.5297297297297298)',
    'b0[-inf, 0.45454545454545453)':'NA&b0[-inf, 0.5297297297297298)','b1[0.45454545454545453,0.5297297297297298)':'NA&b0[-inf, 0.5297297297297298)',
    'b2[0.5297297297297298,0.782608695652174)':'>=0.530','b4[0.782608695652174,inf)':'>=0.530'})


main(df_choose6,final_columns,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/final_test_num_bin_badrate4")  
df_hig_iv = v2_cat_woe_iv(df_choose6,final_columns,'apply_y',0.01,"D:/llm/联合建模算法开发/同盾联合建模_逻辑回归_剔除分数/")
df_hig_iv=df_hig_iv.reset_index().rename(columns={'index':'var_name',0.0:'iv'})
choose_column=df_hig_iv.loc[df_hig_iv.iv>=0.01,'var_name'].values.tolist()


## 训练模型
df_choose7=df_choose6[choose_column+['apply_y']].copy()
#df_choose7.to_excel('D:/llm/联合建模算法开发/逻辑回归结果/训练集数据_20190806.xlsx')
x =  df_choose7[choose_column]
y = df_choose7["apply_y"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
lgt=sm.Logit(y_train,x_train)
result=lgt.fit()
print(result.summary2())
choose_column.remove('prof_title_info_date_Bin')
choose_column.remove('contacts_router_ratio_Bin')
choose_column.remove('i_mean_freq_node_seq_partner_loan_all_all_Bin')
choose_column.remove('cell_reg_time_Bin')
choose_column.remove('i_pctl_cnt_ic_partner_loan_p2pweb_365day_Bin')
choose_column.remove('m_cnt_shoppingappinstlv2_gen_gen_365day_Bin')
choose_column.remove('m_ratio_freq_record_register_all_180day_Bin')
choose_column.remove('max_total_amount_l6m_Bin')
choose_column.remove('contact_bank_call_cnt_Bin')


lr=LogisticRegression(penalty='l1',C=3,n_jobs=-1,verbose=0,random_state=0)
lr.fit(x_train[choose_column],y_train)
score=lr.predict_proba(x_train[choose_column])[:,1]
min_scores = r_p_chart(y_train, score, part=20)
min_scores = [round(i, 5) for i in min_scores]
min_scores[19] = 0
cuts = [round(min_scores[i] * 100.0, 3) for i in range(20)[::-1]] + [100.0]
#joblib.dump(lr,'D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/lr_tdunion_20190821.pkl')
#columns_coef=pd.DataFrame(lr.intercept_.tolist()+lr.coef_[0].tolist(),index=['intercept']+choose_column).reset_index().rename(columns={'index':'var_name',0:'coef'})
#columns_coef['var_name']=columns_coef['var_name'].map(lambda x: x.split('_Bin')[0])
#columns_coef=pd.merge(columns_coef,init_choose_table,on='var_name',how='left')
#columns_coef.to_excel('D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/tdunion_columns_coef_20190821v1.xlsx',index=False)


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

def apt_ec_lastoverduedaystotal(x):
    if x == -99:
        return  -0.0175
    elif x < 5.0:
        return  -0.0175
    elif x< 7.0:
        return   1.2596
    elif x< 9.0:
        return   1.2596
    elif x>= 9.0:
        return   1.2596

def apt_ec_overduephasetotallastyear(x):
    if x == -99:
        return   -0.0311
    elif x <   1.0:
        return  -0.0311
    elif x <  2.0:
        return  0.7648
    elif x< 3.0:
        return   0.7648
    elif x< 4.0:
        return  0.7648
    elif x>= 4.0:
        return  0.7648

def apt_lastloanmode(x):
    if x == -99:
        return  0.0586
    elif x ==1.0:
        return   -0.2159
    elif x==2.0:
        return  0.0586
    elif x==3.0:
        return  0.0586

def cell_reg_time(x):
    if x == -99:
        return  0.0132
    elif x< 923.0:
        return  0.3336
    elif x< 3958.0:
        return   0.0132
    elif x< 5959.0:
        return  -0.4091
    elif x>=5959.0:
        return  -0.4091

def contact_bank_call_cnt(x):
    if x == -99:
        return   0.1512
    elif x <  7.0:
        return   -0.1887
    elif x< 21.0:
        return    0.1512
    elif x< 36.0:
        return    0.1512
    elif x>= 36.0:
        return    0.1512

def contact_unknown_contact_early_morning(x):
    if x == -99:
        return  -0.1128
    elif x < 45.0:
        return  -0.1128
    elif x < 143.0:
        return   0.2587
    elif x < 253.0:
        return   0.2587
    elif x>= 253.0:
        return   0.2587

def i_pctl_cnt_ic_partner_loan_p2pweb_365day(x):
    if x == -99:
        return   -0.4173
    elif x < 0.3315:
        return   -0.4173
    elif x < 0.6106:
        return   0.4409
    elif x < 0.6306:
        return   0.4409
    elif x>= 0.6306:
        return   0.4409

def avg_sms_cnt_l6m(x):
    if x == -99:
        return  0.107
    elif x < 5.833333333333332:
        return   -0.1821
    elif x < 21.166666666666668:
        return   -0.1821
    elif x < 68.66666666666667:
        return   0.107
    elif x>= 68.66666666666667:
        return   0.107

def max_call_in_cnt_l6m(x):
    if x == -99:
        return -0.0982
    elif x < 134.0:
        return  -0.0982
    elif x< 337.0:
        return  -0.0982
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
        return  1.2906 
    elif x>= 2.0:
        return  1.2906 

def max_total_amount_l6m(x):
    if x == -99:
        return   0.171
    elif x < 110.39:
        return  -0.114
    elif x< 298.8:
        return  -0.114 
    elif x< 462.91:
        return   0.171
    elif x>= 462.91:
        return  0.171

def times_by_current_org(x):
    if x == -99:
        return   -0.0714
    elif x < 2.0:
        return   -0.0714
    elif x<  3.0:
        return    0.3617
    elif x < 4.0:
        return    0.3617
    elif x>= 4.0:
        return   0.3617

def vehicle_minput_lastreleasedate(x):
    if x == -99:
        return   -0.2507
    elif x< 1.0:
        return   0.1051
    elif x <6.0:
        return    0.6836
    elif x < 164.0:
        return    0.1051
    elif x>= 164.0:
        return    -0.2507

def m_ratio_cnt_grp_max_partner_loan_all_all(x):
    if x == -99:
        return   0.0974
    elif x < 0.2736766294562478:
        return   0.0974
    elif x< 0.4962264150943396:
        return   0.0974
    elif x< 0.6254:
        return   -0.3519
    elif x>= 0.6254:
        return   -0.3519

def qtorg_query_orgcnt(x):
    if x == -99:
        return  -0.1248
    elif x < 1.0:
        return  -0.1248
    elif x< 3.0:
        return   0.4587
    elif x>= 3.0:
        return   1.0691

def m_cnt_node_dist2_loan_all_all(x):
    if x == -99:
        return  0.3287
    elif x < 1.0:
        return   -0.18
    elif x< 9.0:
        return   -0.18
    elif x< 17.0:
        return   0.3287
    elif x>= 17.0:
        return   0.3287

def m3_id_relate_homeaddress_num(x):
    if x == -99:
        return  -0.1747
    elif x < 1.0:
        return   0.1176
    elif x< 3.0:
        return   0.6186
    elif x>= 3.0:
        return   0.6186

def jxl_id_comb_othertel_num(x):
    if x == -99:
        return  -0.1013
    elif x < 1.0:
        return  -0.1013
    elif x < 2.0:
        return   0.4983
    elif x< 3.0:
        return   0.4983
    elif x>= 3.0:
        return   0.4983


def m_ratio_cnt_partner_all_imbank_365day(x):
    if x == -99:
        return  0.2388
    elif x < 0.5:
        return   -0.2024
    elif x< 0.8333:
        return   -0.2024
    elif x< 0.9016:
        return   0.2388
    elif x>= 0.9016:
        return   0.2388

def m_ratio_freq_record_register_all_180day(x):
    if x == -99:
        return  0.1595
    elif x < 0.0:
        return   -0.121
    elif x< 0.0714:
        return   0.1595
    elif x< 0.1154:
        return   0.1595
    elif x>= 0.1154:
        return   0.1595

def i_mean_freq_node_seq_partner_loan_all_all(x):
    if x == -99:
        return  0.2297
    elif x < 1.2:
        return   -0.2742
    elif x< 1.4444:
        return   0.2297
    elif x< 1.9375:
        return   0.2297
    elif x>= 1.9375:
        return  -0.2742

def m_ratio_cnt_partner_all_o2o_365day(x):
    if x == -99:
        return   0.0735
    elif x < 0.0313:
        return   0.0735
    elif x< 0.1429:
        return   -0.3147
    elif x< 0.3333:
        return   -0.3147
    elif x>= 0.3333:
        return   -0.3147

def contacts_router_ratio(x):
    if x == -99:
        return   0.0948
    elif x < 0.45454545454545453:
        return   0.0948
    elif x < 0.5297297297297298:
        return   0.0948
    elif x < 0.782608695652174:
        return   -0.1844
    elif x>= 0.782608695652174:
        return   -0.1844

def m_cnt_shoppingappinstlv2_gen_gen_365day(x):
    if x == -99:
        return   -0.1377
    elif x < 8.0:
        return   0.2047
    elif x < 13.0:
        return   -0.1377
    elif x < 14.0:
        return   -0.1377
    elif x>= 14.0:
        return   -0.1377


## 时间外验证进行重新评估
outtime_testdata=pd.read_table('outtime_test_20190820.txt',dtype={'app_applycode':str},sep='\u0001')
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


de_dict_var = pd.read_excel('D:/llm/联合建模算法开发/百融联合建模_逻辑回归/de_dict_vars_20190808.xlsx')
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

outtime_testdata["apt_ec_lastoverduedaystotal"+'_Bin'] = outtime_testdata["apt_ec_lastoverduedaystotal"].map(lambda x: apt_ec_lastoverduedaystotal(x))
outtime_testdata["apt_ec_overduephasetotallastyear"+'_Bin'] = outtime_testdata["apt_ec_overduephasetotallastyear"].map(lambda x: apt_ec_overduephasetotallastyear(x))
outtime_testdata['apt_lastloanmode'] = outtime_testdata["apt_lastloanmode"].map(lambda x: apt_lastloanmode(x))
outtime_testdata["cell_reg_time"+'_Bin'] = outtime_testdata["cell_reg_time"].map(lambda x: cell_reg_time(x))
outtime_testdata["contact_bank_call_cnt"+'_Bin'] = outtime_testdata["contact_bank_call_cnt"].map(lambda x: contact_bank_call_cnt(x))
outtime_testdata["contact_unknown_contact_early_morning"+'_Bin'] = outtime_testdata["contact_unknown_contact_early_morning"].map(lambda x: contact_unknown_contact_early_morning(x))
outtime_testdata["max_call_in_cnt_l6m"+'_Bin'] = outtime_testdata["max_call_in_cnt_l6m"].map(lambda x: max_call_in_cnt_l6m(x))
outtime_testdata["max_overdue_terms"+'_Bin'] = outtime_testdata["max_overdue_terms"].map(lambda x: max_overdue_terms(x))
outtime_testdata["max_total_amount_l6m"+'_Bin'] = outtime_testdata["max_total_amount_l6m"].map(lambda x: max_total_amount_l6m(x))
outtime_testdata["times_by_current_org"+'_Bin'] = outtime_testdata["times_by_current_org"].map(lambda x: times_by_current_org(x))
outtime_testdata["vehicle_minput_lastreleasedate"+'_Bin'] = outtime_testdata["vehicle_minput_lastreleasedate"].map(lambda x: vehicle_minput_lastreleasedate(x))
outtime_testdata["avg_sms_cnt_l6m"+'_Bin'] = outtime_testdata["avg_sms_cnt_l6m"].map(lambda x: avg_sms_cnt_l6m(x))
outtime_testdata["m3_id_relate_homeaddress_num"+'_Bin'] = outtime_testdata["m3_id_relate_homeaddress_num"].map(lambda x: m3_id_relate_homeaddress_num(x))
outtime_testdata["jxl_id_comb_othertel_num"+'_Bin'] = outtime_testdata["jxl_id_comb_othertel_num"].map(lambda x: jxl_id_comb_othertel_num(x))
outtime_testdata["i_pctl_cnt_ic_partner_loan_p2pweb_365day"+'_Bin'] = outtime_testdata["i_pctl_cnt_ic_partner_loan_p2pweb_365day"].map(lambda x: i_pctl_cnt_ic_partner_loan_p2pweb_365day(x))
outtime_testdata["m_cnt_node_dist2_loan_all_all"+'_Bin'] = outtime_testdata["m_cnt_node_dist2_loan_all_all"].map(lambda x: m_cnt_node_dist2_loan_all_all(x))
outtime_testdata["m_ratio_cnt_grp_max_partner_loan_all_all"+'_Bin'] = outtime_testdata["m_ratio_cnt_grp_max_partner_loan_all_all"].map(lambda x: m_ratio_cnt_grp_max_partner_loan_all_all(x))
outtime_testdata["m_ratio_cnt_partner_all_imbank_365day"+'_Bin'] = outtime_testdata["m_ratio_cnt_partner_all_imbank_365day"].map(lambda x: m_ratio_cnt_partner_all_imbank_365day(x))
outtime_testdata["qtorg_query_orgcnt"+'_Bin'] = outtime_testdata["qtorg_query_orgcnt"].map(lambda x: qtorg_query_orgcnt(x))
outtime_testdata["m_ratio_freq_record_register_all_180day"+'_Bin'] = outtime_testdata["m_ratio_freq_record_register_all_180day"].map(lambda x: m_ratio_freq_record_register_all_180day(x))
outtime_testdata["i_mean_freq_node_seq_partner_loan_all_all"+'_Bin'] = outtime_testdata["i_mean_freq_node_seq_partner_loan_all_all"].map(lambda x: i_mean_freq_node_seq_partner_loan_all_all(x))
outtime_testdata["m_ratio_cnt_partner_all_o2o_365day"+'_Bin'] = outtime_testdata["m_ratio_cnt_partner_all_o2o_365day"].map(lambda x: m_ratio_cnt_partner_all_o2o_365day(x))
outtime_testdata["contacts_router_ratio"+'_Bin'] = outtime_testdata["contacts_router_ratio"].map(lambda x: contacts_router_ratio(x))
outtime_testdata["m_cnt_shoppingappinstlv2_gen_gen_365day"+'_Bin'] = outtime_testdata["m_cnt_shoppingappinstlv2_gen_gen_365day"].map(lambda x: m_cnt_shoppingappinstlv2_gen_gen_365day(x))




for cn in choose_column:
    print(cn,outtime_testdata[cn].unique())


print('时间外验证')
pred_p4= lr.predict_proba(outtime_testdata[choose_column])[:, 1]
fpr, tpr, th = roc_curve(outtime_testdata.apply_y, pred_p4)
ks3 = tpr - fpr
print('all ks:   ' + str(max(ks3)))
print(roc_auc_score(outtime_testdata.apply_y, pred_p4))
r_p_chart2(outtime_testdata.apply_y, pred_p4, min_scores, part=10)


## 入模变量稳定性评估

df_choose7['app_applydate']=model_data_new['app_applydate']
full_data=pd.concat([df_choose7[choose_column+['apply_y','app_applydate']],outtime_testdata[choose_column+['apply_y','app_applydate']]])
full_data['applymonth']=full_data['app_applydate'].str.slice(0,7)
full_data=full_data[full_data['applymonth']!='2018-12']
#full_data.to_csv('D:/llm/联合建模算法开发/百融联合建模_逻辑回归/full_data_20190722.csv',index=False)

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
    judge_stable_analyze(full_data,cn,'D:/llm/联合建模算法开发/同盾联合建模_逻辑回归/同盾联合建模_逻辑回归_剔除分数/')


##统计分数-ks值和auc值
pred_p5= lr.predict_proba(full_data[choose_column])[:, 1]
full_data['bair_union_scores']=pred_p5*100
num_counts=pd.crosstab(full_data['applymonth'],full_data['apply_y'],margins=True).reset_index()
print(num_counts)
ks_list={}
auc_list={}
for cn in full_data['applymonth'].unique():
    temp=full_data.loc[full_data['applymonth']==cn,['apply_y','bair_union_scores']].copy()
    fpr, tpr, th = roc_curve(temp['apply_y'], temp['bair_union_scores'].values/100)
    ks2 = tpr - fpr
    ks_list[cn]=max(ks2)
    auc_list[cn]=roc_auc_score(temp['apply_y'], temp['bair_union_scores'].values/100)
    
ks_pd=pd.Series(ks_list)
ks_pd=ks_pd.reset_index().rename(columns={'index':'applymonth',0:'ks'})
auc_pd=pd.Series(auc_list)
auc_pd=auc_pd.reset_index().rename(columns={'index':'applymonth',0:'auc'})
auc_ks=pd.merge(ks_pd,auc_pd,on='applymonth',how='inner')
print(auc_ks)
#full_data[['applymonth','contractno','app_applycode','bair_union_scores']].to_excel('D:/llm/联合建模算法开发/文档/百融联合建模分数_20190726.xlsx',index=False)



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