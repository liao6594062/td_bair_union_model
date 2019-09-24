import os
import pandas as pd
import numpy as np
import warnings
from sklearn.externals import joblib
import datetime
from impala.dbapi import connect
from impala.util import as_pandas
from sklearn.metrics  import roc_curve,roc_auc_score
warnings.filterwarnings("ignore")

## 修改路径到数据存储的文件夹
os.chdir('D:/llm/联合建模算法开发/数据/')

##加载模型
clf=joblib.load('gbdt3_v5.pkl')


##python直连hive代码函数

def qurry_hdp(sql,flag=1):
    conn = connect(host='172.30.17.197', port=10000,user='liaoliming', auth_mechanism='PLAIN',password='2osh1ofq')
    cur = conn.cursor()
    cur.execute(sql)
    temp=as_pandas(cur)
    cur.close()
    conn.close()
    if flag != 1:
        temp.columns = [x.split('.')[1] for x in temp.columns]
    return temp


'''
## 导入决策引擎数据，合同信息
'''
## 所需的所有变量列表
model_para1=['jxl_id_comb_othertel_num','jxl_tel_loan_call_sumnum','vehicle_illegal_num','jxl_tel_length','jxl_eachphone_num',
 'jxl_call_num_aver_6months','jxl_nocturnal_ratio','jxl_black_dir_cont_num','yx_underly_record_num','yx_otherorgan_times',
 'apt_currentaddrresideyears','vehicle_evtrpt_mileage','apt_ec_historyloantimes','apt_ec_overduedaystotallastyear','apt_age',
 'vehicle_evtrpt_b2bprice','vehicle_evtrpt_c2bprice', 'vehicle_evtrpt_evalprice2','vehicle_evtrpt_evalprice3','apt_ec_lastloansettleddate',
 'vehicle_minput_lastreleasedate','vehicle_minput_drivinglicensevaliditydate','vehicle_minput_obtaindate','apt_facetrial_creditcardfalg','apt_gender',
 'apt_telecom_phoneattribution','vehicle_minput_lastmortgagerinfo','apt_facetrial_housetype','app_applydate']

dates_columns=['apt_ec_lastloansettleddate','vehicle_minput_lastreleasedate','vehicle_minput_drivinglicensevaliditydate','vehicle_minput_obtaindate']
model_para=['app_applycode']+model_para1
other_require_para=['loan_product','loan_mode','app_callpoint','cd_antifraud_score3','last_update_date','app_applycode','vehicle_evtrpt_evaluateprice']


## 导入决策引擎入参表、出参表、客户申请编号与合同编号关联表
combine_data=pd.read_excel('百融+同盾+自有建模数据整合_20190808.xlsx')
outtime_mdata3=combine_data[(combine_data.app_applydate>='2018-10-01 00:00:00') & (combine_data.app_applydate<='2018-11-30 23:59:59')]


'''
线下计算反欺诈分
'''
m11_data=outtime_mdata3.copy()
m11_data['jxl_id_comb_othertel_num']=-99
m11_data['jxl_tel_loan_call_sumnum']=-99
m11_data['jxl_tel_length']=85  # -998 80
m11_data['jxl_eachphone_num']=20  #-998 10 20 30
m11_data['jxl_call_num_aver_6months']=85  #-998  135  
m11_data['jxl_nocturnal_ratio']=30  #-998 20 越高越低
m11_data['jxl_black_dir_cont_num']=70  #-998 70 100   越高


## 增加两个变量applyday、applymonth
m11_data['applyday'] = m11_data.app_applydate.str.slice(0, 10)  # 生成申请日期（以月为单位）
m11_data['applymonth'] = m11_data.app_applydate.str.slice(0, 7)  # 生成申请日期（以天为单位）

##　异常值处理
m11_data.loc[m11_data.vehicle_minput_obtaindate>='2030-10-01','vehicle_minput_obtaindate']=np.nan  #将异常的车辆获得时间设为缺失值np.nan

## 提取所用到的模型入参数据
model_data=m11_data[model_para1].copy()

for col in dates_columns:  #去除异常的时间
       try:
           model_data.ix[model_data[col]>='2030-01-01',col]=np.nan
       except:
           pass

##  处理日期型变量
def date_cal(x, app_applydate):
    days_dt = pd.to_datetime(app_applydate) - pd.to_datetime(x)
    return days_dt.dt.days

for col in dates_columns:
    if  col!='vehicle_minput_drivinglicensevaliditydate':
      model_data[col] = date_cal(model_data[col], model_data['app_applydate'])
    else:
        model_data[col] = date_cal( model_data['app_applydate'],model_data[col])

model_data=model_data.drop('app_applydate',axis=1) #去除申请日期这个变量

## 将变量的默认值设为缺失值np.nan,并将部分数值的字符型变量转成数值型变量
de_dict_var = pd.read_excel('DE_Var_Select_0.xlsx')  # 处理默认值为np.nan
for i, _ in de_dict_var.iterrows():
    name = de_dict_var.loc[i, 'var_name']
    default = de_dict_var.loc[i, 'default']
    if default != '""' and name in set(model_data.columns):
        try:
            model_data[name] = model_data[name].astype('float64')
            if (model_data[name] == float(default)).sum() > 0:
                model_data.loc[model_data[name] == float(default), name] = np.nan
        except:
            pass

    elif default == '""' and name in set(model_data.columns):
        try:
            model_data[name] = model_data[name].astype('float64')
            if (model_data[name] == float(-99)).sum() > 0:
                model_data.loc[model_data[name] == float(-99), name] = np.nan
            if (model_data[name] == '-99').sum() > 0:
                model_data.loc[model_data[name] == '-99', name] = np.nan
        except:
            pass

## 变量的缺失率统计
#null_data = model_data.isnull().astype('float64')
#null_percents_count = null_data.groupby(m11_data['applymonth']).agg(['mean']).reset_index().rename(columns={'mean': '缺失率', 'applymonth': '申请日期（以月为单位）'})
#null_percents_count.to_excel('./三期监控输出文件/入模变量缺失率按月统计'+str(datetime.datetime.now().date())+'.xlsx')

## 变量衍生逻辑
model_data['vehicle_evtrpt_evalprice_trend'] = model_data['vehicle_evtrpt_evalprice3'] / model_data['vehicle_evtrpt_evalprice2']
model_data.loc[model_data['vehicle_evtrpt_evalprice_trend']==np.inf,'vehicle_evtrpt_evalprice_trend']=np.nan
model_data['vehicle_evtrpt_2bprice_gap'] = abs(model_data['vehicle_evtrpt_b2bprice'] - model_data['vehicle_evtrpt_c2bprice']) / model_data['vehicle_evtrpt_b2bprice']
model_data.loc[model_data['vehicle_evtrpt_2bprice_gap']==np.inf,'vehicle_evtrpt_2bprice_gap']=np.nan
model_data['apt_facetrial_creditcardfalg_2'] = model_data.loc[:,'apt_facetrial_creditcardfalg'].isin([2]).astype('float64')
model_data['apt_telecom_phoneattribution_2n3'] =  model_data.loc[:,'apt_telecom_phoneattribution'].isin([2,3]).astype('float64')
model_data['apt_gender_0'] = model_data.loc[:,'apt_gender'].isin(['女']).astype('float64')
model_data['vehicle_minput_lastmortgagerinfo_1n2'] = model_data.loc[:,'vehicle_minput_lastmortgagerinfo'].isin([1,2]).astype('float64')
model_data['apt_facetrial_housetype_1']=model_data.loc[:,'apt_facetrial_housetype'].isin([1]).astype('float64')


## 提取最终模型入参数据
columns=model_para1[0:15]+['vehicle_evtrpt_2bprice_gap','vehicle_evtrpt_evalprice_trend']+dates_columns+['apt_facetrial_creditcardfalg_2','apt_gender_0','apt_telecom_phoneattribution_2n3','vehicle_minput_lastmortgagerinfo_1n2',
        'apt_facetrial_housetype_1']
final_data=model_data[columns].copy()

## 数值型变量的均值分析
#vars_mean_count=final_data.groupby(m11_data['applymonth']).mean().reset_index().rename(columns={'mean':'均值','applymonth':'申请日期（以月为单位）'})
#vars_mean_count.to_excel('./三期监控输出文件/入模变量均值按月统计'+str(datetime.datetime.now().date())+'.xlsx',index=False)

## 哑变量的各个不同取值对应的客户数分布情况
dummy_vars=['apt_facetrial_creditcardfalg','vehicle_minput_lastmortgagerinfo','apt_facetrial_housetype','apt_telecom_phoneattribution','apt_gender']
dummy_data=model_data[dummy_vars].copy()
dummy_data['applymonth']=m11_data['applymonth'].copy()
for cn in dummy_data:
    if dummy_data[cn].isnull().sum()>0:  #哑变量缺失设为默认值-998
        dummy_data.ix[dummy_data[cn].isnull(),cn]=-998

dummy_mean_count=pd.crosstab(dummy_data['applymonth'],dummy_data['apt_facetrial_creditcardfalg'],margins=True)
dummy_mean_count=dummy_mean_count.div(dummy_mean_count['All'],axis=0).drop('All',axis=1).reset_index()

for cn in ['vehicle_minput_lastmortgagerinfo','apt_facetrial_housetype','apt_telecom_phoneattribution','apt_gender']:
    try:
       dummy_mean_count_1 = pd.crosstab(dummy_data['applymonth'], dummy_data[cn], margins=True)
    except:
       dummy_data[cn]=dummy_data[cn].astype('str')
       dummy_mean_count_1 = pd.crosstab(dummy_data['applymonth'], dummy_data[cn], margins=True)

    dummy_mean_count_1 = dummy_mean_count_1.div(dummy_mean_count_1['All'], axis=0).drop('All', axis=1).reset_index()

    dummy_mean_count=pd.merge(dummy_mean_count,dummy_mean_count_1,on='applymonth',how='inner')

dummy_mean_count.to_excel('./三期监控输出文件/哑变量均值按月统计'+str(datetime.datetime.now().date())+'.xlsx',index=False)

## 将缺失值np.nan设为-998
for cn in final_data:
    if final_data[cn].isnull().sum()>0:
        final_data.loc[final_data[cn].isnull(),cn]=-998

## 计算反欺诈分
p = clf.predict_proba(final_data)[:, 1]
m11_data['scores3']=p*100

'''
变量稳定性分析
'''
columns_data_analyze=final_data.copy()
columns_data_analyze[columns_data_analyze==-998]=np.nan
columns_data_analyze['app_applydate']=m11_data['app_applydate'].copy()
columns_data_analyze['applymonth']=columns_data_analyze['app_applydate'].str.slice(0,7)
columns_data_analyze.loc[columns_data_analyze['applymonth'].isin(['2017-05','2017-06','2017-07','2017-08','2017-09','2017-10']),'applymonth']='2017-05'
columns_data_analyze['scores3']=m11_data['scores3'].copy()
columns_data_analyze.loc[m11_data.app_applydate>='2018-12-10 00:00:00','scores3']=m11_data.loc[m11_data.app_applydate>='2018-12-10 00:00:00','cd_antifraud_score3']
columns_data_analyze=columns_data_analyze.loc[((columns_data_analyze.app_applydate>='2018-10-08 00:00:00') & (columns_data_analyze.app_applydate<='2018-10-09 23:59:59'))==False]
columns_data_analyze.scores2 = columns_data_analyze.scores2.astype('float64')
cuts = [0.0,6.595,6.994,7.354,7.692,8.036,8.384,8.736,9.117,9.563,10.051,10.629,11.313,12.107,13.058,14.240,15.728,17.644,20.455,25.720,100]
columns_data_analyze['newgrp'] = pd.cut(columns_data_analyze.scores3_new, cuts, right=False)
columns_data_analyze.loc[(columns_data_analyze.app_applydate>='2018-12-15') & (columns_data_analyze['jxl_id_comb_othertel_num']==0),'jxl_id_comb_othertel_num']=np.nan
columns_data_analyze.loc[(columns_data_analyze.app_applydate>='2018-12-15') & (columns_data_analyze['jxl_tel_loan_call_sumnum']==0),'jxl_tel_loan_call_sumnum']=np.nan

for cn in  columns:
    cn_mean=pd.crosstab(index=columns_data_analyze['newgrp'],columns=columns_data_analyze['applymonth'],values=columns_data_analyze[cn],aggfunc='mean',margins=True)
    cn_isnullpercents=pd.crosstab(index=columns_data_analyze['newgrp'],columns=columns_data_analyze['applymonth'],values=columns_data_analyze[cn].isnull().astype('float64'),aggfunc='mean',margins=True)
    cn_mean.to_csv('./三期监控输出文件/变量稳定性_单个变量/'+cn+'均值按分数段以及按月统计'+str(end_date.date())+'.csv')
    cn_isnullpercents.to_csv('./三期监控输出文件/变量稳定性_单个变量/' + cn + '缺失率按分数段以及按月统计' + str(end_date.date()) + '.csv')

columns_data_analyze.loc[columns_data_analyze['apt_comp_monthlysalary']>=5000000,'apt_comp_monthlysalary']=np.nan
columns_data_analyze.loc[columns_data_analyze['vehicle_evtrpt_mileage']>=10000000,'vehicle_evtrpt_mileage']=np.nan
columns_mean=columns_data_analyze.groupby('applymonth')[columns].mean().T
columns_mean.to_csv('./三期监控输出文件/变量稳定性_所有变量/'+cn+'均值按分数段以及按月统计'+str(end_date.date())+'.csv')
columns_data_analyze_null=columns_data_analyze.isnull().astype('float64')
columns_isnullpercents=columns_data_analyze_null.groupby(columns_data_analyze['applymonth'])[columns].mean().T
columns_isnullpercents.to_csv('./三期监控输出文件/变量稳定性_所有变量/'+cn+'缺失率按分数段以及按月统计'+str(end_date.date())+'.csv')

## 剔除异常的样本
m11_data=m11_data.loc[((m11_data.app_applydate>='2018-10-08 00:00:00') & (m11_data.app_applydate<='2018-10-09 23:59:59'))==False]
m11_data.loc[m11_data.app_applydate>='2018-12-10 00:00:00','scores3']=m11_data.loc[m11_data.app_applydate>='2018-12-10 00:00:00','cd_antifraud_score3']
m11_data['scores3']=m11_data['scores3'].astype('float64')

## 分数均值或者中位数按月统计
scores3_mean_bymonth=m11_data.groupby('applymonth')['scores3'].mean().reset_index().rename(columns={'scores3':'三期模型分数均值'})
scores3_median_bymonth=m11_data.groupby('applymonth')['scores3'].median().reset_index().rename(columns={'scores3':'三期模型分数中位数'})
scores3_counts=pd.merge(scores3_mean_bymonth,scores3_median_bymonth,on='applymonth',how='inner')
scores3_counts.to_excel('./三期监控输出文件/三期分数均值中位数按月统计'+str(datetime.datetime.now().date())+'.xlsx',index=False)

##增加若干审批变量，比如建议成数、最终审批通过率、放款成数等
shenpi_data = pd.read_table('shenpi_20190320.txt', encoding='gbk', dtype={'applycode': str}) #读取决策引擎入参数据，从tbd中提取
shenpi_data=shenpi_data.replace('\\N',np.nan)
m11_data=pd.merge(m11_data,shenpi_data,left_on='app_applycode',right_on='applycode',how='left')
m11_data=m11_data.drop('applycode',axis=1)
m11_data['fin_to_value']=m11_data['lastauditamt']/m11_data['vehicle_evtrpt_evaluateprice']
m11_data.loc[m11_data['fin_to_value']==np.inf,'fin_to_value']=np.nan
m11_data.loc[m11_data['fin_to_value']>10,'fin_to_value']=np.nan


'''
## 统计反欺诈划分的客户数占比，可调整日期范围、押证类型、网点地区等变量
'''
def  system_steady_monitor(mdata0,start_date,end_date):

    mdata1 = mdata0.ix[(mdata0.app_applydate >= start_date) & (mdata0.app_applydate <= end_date) , :]
    mdata1 = mdata1.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
    mdata1['applymonth']=mdata1.app_applydate.str.slice(0,7)
    mdata1['applyday']=mdata1.app_applydate.str.slice(0,10)

    mdata1=mdata1.replace('\\N',np.nan)
    mdata1=mdata1.ix[mdata1.scores3.notnull(),:]
    mdata1.scores3=mdata1.scores3.astype('float64')
    cuts = [0.0,6.595,6.994,7.354,7.692,8.036,8.384,8.736,9.117,9.563,10.051,10.629,11.313,12.107,13.058,14.240,15.728,17.644,20.455,25.720,100]
    mdata1['newgrp']=pd.cut(mdata1.scores3,cuts,right=False)

    score_dist = pd.crosstab(mdata1['newgrp'], mdata1['applymonth'], margins=True)
    score_dist=score_dist.div(score_dist.ix['All', :], axis=1).drop('All', axis=1).reset_index()

    applynumber_byday_count=pd.crosstab(mdata1['newgrp'], mdata1['applyday'], margins=True)
    applynumber_byday_count.reset_index(drop=False).to_csv('./三期监控输出文件/时间外押证各分数段客户数按天统计' + str(datetime.datetime.now().date()) + '.csv', index=False)

    return  score_dist

start_date1 = '2017-10-01 00:00:00'
end_date1 = '2019-03-13 23:59:59'
system_steady_count=system_steady_monitor(m11_data,start_date1,end_date1)
system_steady_count.to_csv('./三期监控输出文件/模型20分段各段分数占比按月统计'+str(datetime.datetime.now().date())+'.csv',index=False)
system_steady_count_worstsegment=system_steady_count.iloc[19,:]
system_steady_count_worstsegment.to_csv('./三期监控输出文件/最差分数段占比按月统计'+str(datetime.datetime.now().date())+'.csv')


'''
## 反欺诈分与放款情况统计，可调整日期范围、押证类型、网点地区等变量
'''
## 用TBD上的底层还款表计算曾经最大逾期天数（表现期截止到2019-01-06）
paymentdata = pd.read_table('payment_20190315_3q.txt', delimiter='\u0001')
paymentdata= paymentdata.replace('\\N', np.nan)

performer_day='2019-03-15 00:00:00'
paymentdata=paymentdata.loc[paymentdata.shouldpaydate < performer_day,['contractno','paydate','shouldpaydate','shouldcapital', 'shouldgpsf', 'shouldint', 'shouldmanage','update_time','totalphases','payphases','phases',
                                                                       'loandate','applydate','update_time']]
paymentdata.loc[(paymentdata.paydate >= performer_day), 'paydate'] =performer_day  ## 将表现窗口设为截止到2019-01-06

paymentdata=paymentdata[paymentdata.paydate>='1970-12-03 00:00:00'].copy() #异常值处理
paymentdata['overdue_days'] = (pd.to_datetime(paymentdata.paydate) - pd.to_datetime(paymentdata.shouldpaydate)).dt.days
paymentdata.loc[(paymentdata[['shouldcapital', 'shouldgpsf', 'shouldint', 'shouldmanage']] == 0).all(axis=1), 'overdue_days'] = 0  ##计算历史最大逾期天数
paymentdata.loc[paymentdata['overdue_days'] < 0, 'overdue_days'] = 0
paymentdata[['totalphases', 'payphases', 'phases']] = paymentdata[['totalphases', 'payphases', 'phases']].astype('int64')  # 将一些字段转成整型数据
paymentdata_maxdue = paymentdata.groupby(['contractno']).overdue_days.max().reset_index().rename(columns={'overdue_days': 'maxoverduedays'})
paymentdata_firstdue=paymentdata[paymentdata.payphases<=1].groupby(['contractno']).overdue_days.max().reset_index().rename(columns={'overdue_days': 'firstduedays'})
paymentdata_maxdue=pd.merge(paymentdata_maxdue,paymentdata_firstdue,on='contractno',how='left')
paymentdata_maxdue.loc[paymentdata_maxdue['firstduedays'].isnull(),'firstduedays']=0

paymentdata_totalphases = paymentdata.groupby(['contractno']).totalphases.max().reset_index()  # 计算贷款总期限,不包括展期
paymentdata_realtotalphases = paymentdata[paymentdata.update_time< performer_day].groupby(['contractno']).payphases.max().reset_index().rename(columns={'payphases': 'max_payphases'})  # 包括是否展期
paymentdata_totalphases = pd.merge(paymentdata_totalphases, paymentdata_realtotalphases, on='contractno', how='inner')
paymentdata_totalphases['realtotalphases'] = paymentdata_totalphases[['totalphases', 'max_payphases']].max(axis=1)  # 在实际贷款期限与是否展期合并获得总贷款期限

paymentdata_returnphases = paymentdata.groupby(['contractno']).payphases.max().reset_index().rename(columns={'payphases': 'returnphases'})  # 计算已还期数
paymentdata_phases_counts = pd.merge(paymentdata_returnphases, paymentdata_totalphases, on='contractno',how='inner')  # 合并贷款期限与已还期数
paymentdata_phases_counts = pd.merge(paymentdata_phases_counts, paymentdata_maxdue, on='contractno',how='inner')  # 合并最大逾期与贷款期限
del paymentdata,paymentdata_totalphases,paymentdata_returnphases,paymentdata_maxdue

findata0 = pd.read_table('fnsche_over20190315.txt',encoding='gbk',dtype={'contractno': str})#导入进度逾期表
findata0=findata0.loc[findata0['loandate'].notnull(),['loandate','contractno','totalduedays','totalphases','returnstatus','currentduedays']].copy()
findata0['loandate'] = findata0['loandate'].map(lambda x: datetime.datetime.strptime(str(x),"%d%b%Y")).copy()
findata0=findata0[findata0['loandate']>='2017-05-16 00:00:00'].copy()

# apply_contract_report=qurry_hdp('SELECT tlf.applycode,tlf.applydate,tlc.contractno FROM ods_lms.tb_lb_applyinfo  tlf left join  ods_lms.tb_lm_contract   tlc on tlf.id=tlc.applyid',flag=0)
# apply_contract_report=pd.read_table('applyid_applycode_htbh_20190321.txt',encoding='gbk',dtype={'applycode': str})#导入进度逾期表
# apply_contract_report=apply_contract_report.replace('\\N',np.nan)
# apply_contract_report = apply_contract_report.loc[(apply_contract_report.applycode.isnull() == False) & (apply_contract_report.contractno.isnull() == False),:].drop_duplicates()

apply_contract_report=pd.read_table('applycode_20190322v2.txt',sep='\u0001',dtype={'applycode': str})#导入进度逾期表
apply_contract_report=apply_contract_report.replace('\\N',np.nan)


def final_scores_monitor(mdata0,apply_contract_report,paymentdata_phases_counts,findata0,start_date,end_date):

    mdata1 = mdata0.loc[(mdata0.app_applydate >= start_date) & (mdata0.app_applydate <= end_date) ,:]
    mdata1 = mdata1.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
    mdata1['applymonth']=mdata1.app_applydate.str.slice(0,7)
    mdata1 = pd.merge(mdata1, apply_contract_report, left_on='app_applycode', right_on='applycode', how='left')

    mdata2=mdata1.copy()
    del mdata1
    mdata2=mdata2.replace('\\N',np.nan)
    mdata2 = mdata2.ix[mdata2.scores3.notnull(), :]

    findata=findata0.loc[findata0['loandate'].notnull(),['loandate','contractno','currentduedays','returnstatus']].copy()
    findata = findata[(findata['loandate']>= start_date)].copy() #.rename(columns={'ContractNo': 'contractno'})
    findata = findata.sort_values(['contractno', 'loandate']).drop_duplicates('contractno', keep='last')
    findata = pd.merge(findata, paymentdata_phases_counts, on='contractno', how='left')
    fin_data=pd.merge(mdata2, findata, on='contractno', how='inner')

    findata['whether_loan']=1
    final_data=pd.merge(mdata2, findata, on='contractno', how='left')
    del findata,mdata2
    final_data.loc[final_data['whether_loan'].isnull(),'whether_loan']=0
    cuts = [0.0,6.595,6.994,7.354,7.692,8.036,8.384,8.736,9.117,9.563,10.051,10.629,11.313,12.107,13.058,14.240,15.728,17.644,20.455,25.720,100]
    final_data['newgrp']=pd.cut(final_data.scores3,cuts,right=False)
    final_data['shenpi_pass']=final_data['lastauditresult'].isin([1]).astype('float64')
    fin_percents = final_data.groupby('newgrp')[['shenpi_pass','whether_loan']].mean().reset_index()
    fin_percents.to_csv('./三期监控输出文件/某个时间段审批通过率与放款率统计'+str(datetime.datetime.now().date())+'.csv',index=False)
    print(final_data[['shenpi_pass','whether_loan']].mean())

    fin_percents_bymonth =pd.crosstab(index=final_data['newgrp'],columns=final_data['applymonth'],values=final_data['whether_loan'],aggfunc='mean',margins=True)
    fin_percents_bymonth.to_csv('./三期监控输出文件/放款率按月统计' + str(datetime.datetime.now().date()) + '.csv')

    fin_data['newgrp']=pd.cut(fin_data.scores3,cuts,right=False)
    fin_chenshu=fin_data.groupby('newgrp')['fin_to_value'].mean().reset_index()
    fin_chenshu.to_csv('./三期监控输出文件/某个时间段放款成数统计'+str(datetime.datetime.now().date())+'.csv',index=False)
    print(fin_data['fin_to_value'].mean())
    fin_data=fin_data.drop('newgrp',axis=1)

    del final_data

    return  fin_data

start_date1 = '2018-08-28 00:00:00'
end_date1 = '2019-01-28 23:59:59'
fin_data=final_scores_monitor(m11_data,apply_contract_report,paymentdata_phases_counts,findata0,start_date1,end_date1)

'''
## 计算好坏客户区分能力,可调整日期范围、押证类型、网点地区等变量
'''

def comput_judge_bad_ability(mdata0,apply_contract_report,paymentdata_phases_counts,findata0,start_date,end_date,payment_way=None):
    fin_data = final_scores_monitor(mdata0, apply_contract_report,paymentdata_phases_counts,findata0, start_date,end_date)

    if payment_way==1:
        fin_data=fin_data[fin_data.hkfs.isin(['先息后本'])].copy()
    if payment_way==2:
        fin_data = fin_data[fin_data.hkfs.isin(['等本等息','等额本息'])].copy()

    my_data = fin_data.copy()
    del fin_data
    my_data['overdue_flag'] = (my_data.maxoverduedays >= 16)
    my_data['bad'] = (my_data.overdue_flag == True)  # 占比约为4.6%
    my_data['chargeoff_flag'] = (my_data.maxoverduedays == 0) & (my_data.returnstatus.isin(['已结清']))  # 结清里面大概有75%没有逾期
    my_data['r6_good_flag'] = (my_data.returnphases >= 6) & (my_data.maxoverduedays == 0)
    my_data['good'] = my_data.chargeoff_flag | my_data.r6_good_flag
    my_data['y'] = 2
    my_data.loc[(my_data.bad == True) & (my_data.good == False), 'y'] = 1
    my_data.loc[(my_data.bad == False) & (my_data.good == True), 'y'] = 0

    my_data['currentdue4+']=(my_data.currentduedays >= 4).astype('float64')
    my_data['currentdue16+']=(my_data.currentduedays >= 16).astype('float64')
    my_data['currentdue30+'] = (my_data.currentduedays >= 30).astype('float64')

    my_data['firstdue4+']=(my_data.firstduedays >= 4).astype('float64')
    my_data['firstdue16+']=(my_data.firstduedays >= 16).astype('float64')
    my_data['firstdue30+'] = (my_data.firstduedays >= 30).astype('float64')

    my_data.scores2=my_data.scores3.astype('float64')
    cuts = [0.0,6.595,6.994,7.354,7.692,8.036,8.384,8.736,9.117,9.563,10.051,10.629,11.313,12.107,13.058,14.240,15.728,17.644,20.455,25.720,100]
    cuts_check=[0,11.313,17.644,25.720,100]
    my_data['newgrp']=pd.cut(my_data.scores3,cuts,right=False)
    my_data['newgrp_check'] = pd.cut(my_data.scores3, cuts_check, right=False)

    score_dist = pd.crosstab(my_data['newgrp'], my_data['y'], margins=True).reset_index()
    current_score_dist = my_data.groupby(['newgrp'])[['currentdue4+', 'currentdue16+', 'currentdue30+','firstdue4+', 'firstdue16+', 'firstdue30+']].mean().reset_index()
    current_sum = pd.DataFrame(my_data[['currentdue4+', 'currentdue16+', 'currentdue30+','firstdue4+', 'firstdue16+', 'firstdue30+']].mean()).T
    current_sum['newgrp'] = 'All'
    current_sum = current_sum[['newgrp', 'currentdue4+', 'currentdue16+', 'currentdue30+','firstdue4+', 'firstdue16+', 'firstdue30+']]
    current_score_dist = pd.concat([current_score_dist, current_sum], axis=0).reset_index()
    score_dist_count1 = pd.merge(score_dist, current_score_dist, on='newgrp', how='inner')

    score_dist_check = pd.crosstab(my_data['newgrp_check'], my_data['y'], margins=True).reset_index()
    current_score_dist = my_data.groupby(['newgrp_check'])[['currentdue4+', 'currentdue16+', 'currentdue30+','firstdue4+', 'firstdue16+', 'firstdue30+']].mean().reset_index()
    current_sum = pd.DataFrame(my_data[['currentdue4+', 'currentdue16+', 'currentdue30+','firstdue4+', 'firstdue16+', 'firstdue30+']].mean()).T
    current_sum['newgrp_check'] = 'All'
    current_sum = current_sum[['newgrp_check', 'currentdue4+', 'currentdue16+', 'currentdue30+','firstdue4+', 'firstdue16+', 'firstdue30+']]
    current_score_dist = pd.concat([current_score_dist, current_sum], axis=0).reset_index()
    score_dist_count2 = pd.merge(score_dist_check, current_score_dist, on='newgrp_check', how='inner')
    score_dist_count2 = score_dist_count2.rename(columns={'newgrp_check': 'newgrp'})

    score_dist_count = pd.concat([score_dist_count1, score_dist_count2], axis=0)


    del my_data

    return  score_dist_count

start_date1 = ['2017-05-16 00:00:00','2017-09-16 00:00:00','2017-05-16 00:00:00','2018-08-29 00:00:00']
end_date1=['2017-09-15 23:59:59','2018-08-28 23:59:59','2018-08-28 23:59:59','2019-01-28 23:59:59']
for i in range(len(start_date1)):
      score_dist_count=comput_judge_bad_ability(m11_data,apply_contract_report,paymentdata_phases_counts,findata0,start_date1[i],end_date1[i])
      if i==0:
          score_dist_count.to_csv('./三期监控输出文件/三期模型建模样本区分能力_'+str(datetime.datetime.now().date())+'.csv',index=False)
      if i==1:
          score_dist_count.to_csv('./三期监控输出文件/三期模型时间外样本区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)
      if i==2:
          score_dist_count.to_csv('./三期监控输出文件/三期模型全部样本区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)
      if i==3:
          score_dist_count.to_csv('./三期监控输出文件/三期模型其他未还期数小于6期的样本区分能力_' + str(datetime.datetime.now().date()) + '.csv',index=False)


'''
# 分结清再贷客户、新增贷款客户来统计ks值
'''

findata0 = pd.read_table('fnsche_over20190315.txt', encoding='gbk', dtype={'contractno': str})  # 导入进度逾期表
findata0 = findata0.loc[findata0['loandate'].notnull(), ['loandate','contractno','totalduedays','totalphases','returnstatus','currentduedays','returnphases', 'docno_md5','carno_md5']].copy()
findata0['loandate'] = findata0['loandate'].map(lambda x: datetime.datetime.strptime(str(x), "%d%b%Y")).copy()  # pd.to_datetime(findata0['loandate']).astype('str')
new_loan_data= findata0.sort_values(['docno_md5','loandate']).drop_duplicates('carno_md5', keep='first')[['contractno','loandate','carno_md5']]
new_loan_data['new_loan']=1
findata0=pd.merge(findata0,new_loan_data,on=['contractno','loandate','carno_md5'],how='left')
findata0.ix[findata0['new_loan'].isnull(),'new_loan']=0
del new_loan_data


## 获取新增客户标识以及数据
def segment_scores_monitor(mdata0,apply_contract_report,paymentdata_phases_counts,findata0,start_date,end_date):

    mdata1 = mdata0.loc[(mdata0.app_applydate >= start_date) & (mdata0.app_applydate <= end_date) ,:]
    mdata1 = mdata1.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
    mdata1['applymonth']=mdata1.app_applydate.str.slice(0,7)
    mdata1 = pd.merge(mdata1, apply_contract_report, left_on='app_applycode', right_on='applycode', how='left')

    mdata2=mdata1.copy()
    del mdata1
    mdata2=mdata2.replace('\\N',np.nan)
    mdata2 = mdata2.ix[mdata2.scores3.notnull(), :]

    findata=findata0.loc[findata0['loandate'].notnull(),['loandate','contractno','currentduedays','returnstatus','new_loan']].copy()
    findata = findata[(findata['loandate']>= start_date)].copy() #.rename(columns={'ContractNo': 'contractno'})
    findata = findata.sort_values(['contractno', 'loandate']).drop_duplicates('contractno', keep='last')
    findata = pd.merge(findata, paymentdata_phases_counts, on='contractno', how='left')
    fin_data=pd.merge(mdata2, findata, on='contractno', how='inner')

    del findata,mdata2

    return  fin_data

## 计算模型的区分能力
def comput_judge_segments_bad_ability(mdata0,apply_contract_report,paymentdata_phases_counts,findata0,start_date,end_date,new_consumer):
    fin_data = segment_scores_monitor(mdata0, apply_contract_report,paymentdata_phases_counts,findata0, start_date,end_date)
    fin_data=fin_data[fin_data.new_loan==new_consumer].copy()

    my_data = fin_data.copy()
    del fin_data
    my_data['overdue_flag'] = (my_data.maxoverduedays >= 16)
    my_data['bad'] = (my_data.overdue_flag == True)  # 占比约为4.6%
    my_data['chargeoff_flag'] = (my_data.maxoverduedays == 0) & (my_data.returnstatus.isin(['已结清']))  # 结清里面大概有75%没有逾期
    my_data['r6_good_flag'] = (my_data.returnphases >= 6) & (my_data.maxoverduedays == 0)
    my_data['good'] = my_data.chargeoff_flag | my_data.r6_good_flag
    my_data['y'] = 2
    my_data.loc[(my_data.bad == True) & (my_data.good == False), 'y'] = 1
    my_data.loc[(my_data.bad == False) & (my_data.good == True), 'y'] = 0

    my_data['currentdue4+']=(my_data.currentduedays >= 4).astype('float64')
    my_data['currentdue16+']=(my_data.currentduedays >= 16).astype('float64')
    my_data['currentdue30+'] = (my_data.currentduedays >= 30).astype('float64')

    my_data.scores2=my_data.scores3.astype('float64')
    cuts = [0.0,6.595,6.994,7.354,7.692,8.036,8.384,8.736,9.117,9.563,10.051,10.629,11.313,12.107,13.058,14.240,15.728,17.644,20.455,25.720,100]
    cuts_check=[0,11.313,17.644,25.720,100]
    my_data['newgrp']=pd.cut(my_data.scores3,cuts,right=False)
    my_data['newgrp_check'] = pd.cut(my_data.scores3, cuts_check, right=False)

    score_dist = pd.crosstab(my_data['newgrp'], my_data['y'], margins=True).reset_index()
    current_score_dist = my_data.groupby(['newgrp'])[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean().reset_index()
    current_sum = pd.DataFrame(my_data[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean()).T
    current_sum['newgrp'] = 'All'
    current_sum = current_sum[['newgrp', 'currentdue4+', 'currentdue16+', 'currentdue30+']]
    current_score_dist = pd.concat([current_score_dist, current_sum], axis=0).reset_index()
    score_dist_count1 = pd.merge(score_dist, current_score_dist, on='newgrp', how='inner')

    score_dist_check = pd.crosstab(my_data['newgrp_check'], my_data['y'], margins=True).reset_index()
    current_score_dist = my_data.groupby(['newgrp_check'])[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean().reset_index()
    current_sum = pd.DataFrame(my_data[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean()).T
    current_sum['newgrp_check'] = 'All'
    current_sum = current_sum[['newgrp_check', 'currentdue4+', 'currentdue16+', 'currentdue30+']]
    current_score_dist = pd.concat([current_score_dist, current_sum], axis=0).reset_index()
    score_dist_count2 = pd.merge(score_dist_check, current_score_dist, on='newgrp_check', how='inner')
    score_dist_count2 = score_dist_count2.rename(columns={'newgrp_check': 'newgrp'})

    score_dist_count = pd.concat([score_dist_count1, score_dist_count2], axis=0)

    del my_data

    return  score_dist_count

start_date1 = ['2017-05-16 00:00:00','2017-09-16 00:00:00','2017-05-16 00:00:00']
end_date1=['2017-09-15 23:59:59','2018-08-28 23:59:59','2018-08-28 23:59:59']
for i in range(3):
      score_dist_count=comput_judge_segments_bad_ability(m11_data,apply_contract_report,paymentdata_phases_counts,findata0,start_date1[i],end_date1[i],1)
      if i==0:
          score_dist_count.to_csv('./三期监控输出文件/三期模型新增建模样本区分能力_'+str(datetime.datetime.now().date())+'.csv',index=False)
      if i==1:
          score_dist_count.to_csv('./三期监控输出文件/三期模型新增时间外样本区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)
      if i==2:
          score_dist_count.to_csv('./三期监控输出文件/三期模型新增全部样本区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)

      score_dist_count=comput_judge_segments_bad_ability(m11_data,apply_contract_report,paymentdata_phases_counts,findata0,start_date1[i],end_date1[i],0)
      if i==0  :
          score_dist_count.to_csv('./三期监控输出文件/三期模型结清建模样本区分能力_'+str(datetime.datetime.now().date())+'.csv',index=False)
      if i==1:
          score_dist_count.to_csv('./三期监控输出文件/三期模型结清时间外样本区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)
      if i==2:
          score_dist_count.to_csv('./三期监控输出文件/三期模型结清全部样本区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)

'''
固定6个月表现期的模型分析
'''
paymentdata0 = pd.read_table('payment_20190315_3q.txt', sep='\u0001') #导入还款表)
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
paymentdata_totalphases=paymentdata.groupby(['contractno']).totalphases.max().reset_index()
paymentdata_maxdue=pd.merge(paymentdata_maxdue,paymentdata_totalphases,on='contractno',how='inner')
paymentdata_maxdue=paymentdata_maxdue.drop('totalphases',axis=1)
del paymentdata

findata0 = pd.read_table('fnsche_over20190315.txt',encoding='gbk',dtype={'contractno': str})#导入进度逾期表
findata0=findata0.loc[findata0['loandate'].notnull(),['loandate','contractno','totalduedays','totalphases','returnstatus','currentduedays','returnphases']].copy()
findata0['loandate'] = findata0['loandate'].map(lambda x: datetime.datetime.strptime(str(x),"%d%b%Y")).copy()
findata0=findata0[findata0['loandate']>='2017-05-16 00:00:00'].copy()


def final_scores_monitor(mdata0,apply_contract_report,paymentdata_maxdue,findata0,start_date,end_date):

    mdata1 = mdata0.loc[(mdata0.app_applydate >= start_date) & (mdata0.app_applydate <= end_date) ,:]
    mdata1 = mdata1.sort_values(['app_applycode', 'last_update_date']).drop_duplicates('app_applycode', keep='last')
    mdata1['applymonth']=mdata1.app_applydate.str.slice(0,7)
    mdata1 = pd.merge(mdata1, apply_contract_report, left_on='app_applycode', right_on='applycode', how='left')

    mdata2=mdata1.copy()
    del mdata1
    mdata2=mdata2.replace('\\N',np.nan)
    mdata2 = mdata2.loc[mdata2.scores3.notnull(), :]

    findata=findata0.loc[findata0['loandate'].notnull(),['loandate','contractno','currentduedays','returnstatus']].copy()
    findata = findata[(findata['loandate']>= start_date)].copy() #.rename(columns={'ContractNo': 'contractno'})
    findata = findata.sort_values(['contractno', 'loandate']).drop_duplicates('contractno', keep='last')
    findata = pd.merge(findata, paymentdata_maxdue, on='contractno', how='left')
    fin_data=pd.merge(mdata2, findata, on='contractno', how='inner')

    del findata,mdata2

    return  fin_data

def comput_judge_bad_ability(mdata0,apply_contract_report,paymentdata_maxdue,findata0,start_date,end_date):
    fin_data = final_scores_monitor(mdata0, apply_contract_report,paymentdata_maxdue,findata0, start_date,end_date)

    my_data = fin_data.copy()
    del fin_data
    my_data['overdue_flag'] = (my_data.maxoverduedays >= 16)
    my_data['bad'] = (my_data.overdue_flag == True)  # 占比约为4.6%
    my_data['good'] =(my_data.maxoverduedays == 0)
    my_data['y'] = 2
    my_data.loc[(my_data.bad == True) & (my_data.good == False), 'y'] = 1
    my_data.loc[(my_data.bad == False) & (my_data.good == True), 'y'] = 0

    my_data['currentdue4+']=(my_data.currentduedays >= 4).astype('float64')
    my_data['currentdue16+']=(my_data.currentduedays >= 16).astype('float64')
    my_data['currentdue30+'] = (my_data.currentduedays >= 30).astype('float64')

    my_data.scores2=my_data.scores3.astype('float64')
    cuts = [0.0,6.595,6.994,7.354,7.692,8.036,8.384,8.736,9.117,9.563,10.051,10.629,11.313,12.107,13.058,14.240,15.728,17.644,20.455,25.720,100]
    cuts_check=[0,11.313,17.644,25.720,100]
    my_data['newgrp']=pd.cut(my_data.scores3,cuts,right=False)
    my_data['newgrp_check'] = pd.cut(my_data.scores3, cuts_check, right=False)

    score_dist = pd.crosstab(my_data['newgrp'], my_data['y'], margins=True).reset_index()
    current_score_dist = my_data.groupby(['newgrp'])[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean().reset_index()
    current_sum = pd.DataFrame(my_data[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean()).T
    current_sum['newgrp'] = 'All'
    current_sum = current_sum[['newgrp', 'currentdue4+', 'currentdue16+', 'currentdue30+']]
    current_score_dist = pd.concat([current_score_dist, current_sum], axis=0).reset_index()
    score_dist_count1 = pd.merge(score_dist, current_score_dist, on='newgrp', how='inner')

    score_dist_check = pd.crosstab(my_data['newgrp_check'], my_data['y'], margins=True).reset_index()
    current_score_dist = my_data.groupby(['newgrp_check'])[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean().reset_index()
    current_sum = pd.DataFrame(my_data[['currentdue4+', 'currentdue16+', 'currentdue30+']].mean()).T
    current_sum['newgrp_check'] = 'All'
    current_sum = current_sum[['newgrp_check', 'currentdue4+', 'currentdue16+', 'currentdue30+']]
    current_score_dist = pd.concat([current_score_dist, current_sum], axis=0).reset_index()
    score_dist_count2 = pd.merge(score_dist_check, current_score_dist, on='newgrp_check', how='inner')
    score_dist_count2 = score_dist_count2.rename(columns={'newgrp_check': 'newgrp'})

    score_dist_count = pd.concat([score_dist_count1, score_dist_count2], axis=0)

    test_data=my_data[['scores3','y','app_applycode','app_applydate']].copy()

    del my_data

    return  score_dist_count,test_data

start_date1 = ['2017-05-16 00:00:00','2017-09-16 00:00:00','2017-05-16 00:00:00']
end_date1=['2017-09-15 23:59:59','2018-08-28 23:59:59','2018-08-28 23:59:59']
for i in range(3):
      score_dist_count,test_data=comput_judge_bad_ability(m11_data,apply_contract_report,paymentdata_maxdue,findata0,start_date1[i],end_date1[i])
      if i==0:
          score_dist_count.to_csv('./三期监控输出文件/三期模型建模样本固定6个月表现期的区分能力_'+str(datetime.datetime.now().date())+'.csv',index=False)
      if i==1:
          score_dist_count.to_csv('./三期监控输出文件/三期模型时间外样本固定6个月表现期区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)
      if i==2:
          score_dist_count.to_csv('./三期监控输出文件/三期模型全部样本固定6个月表现期区分能力_' + str(datetime.datetime.now().date()) + '.csv', index=False)


start_date1 = '2017-05-16 00:00:00'
end_date1 = '2018-08-28 23:59:59'
score_dist_count,testdata=comput_judge_bad_ability(m11_data,apply_contract_report,paymentdata_maxdue,findata0,start_date1,end_date1)

## 按月统计模型ks值
testdata['applymonth']=testdata['app_applydate'].str.slice(0,7)
testdata=testdata[testdata['y'].isin([0,1])]
num_counts=pd.crosstab(testdata['applymonth'],testdata['y'],margins=True).reset_index()
ks_list={}
for cn in testdata['applymonth'].unique():
    temp=testdata.ix[testdata['applymonth']==cn,['y','scores3']].copy()
    fpr, tpr, th = roc_curve(temp['y'], temp['scores3'].values/100)
    ks2 = tpr - fpr
    auc_value=roc_auc_score(temp['y'], temp['scores3'].values/100)
    ks_list[cn]=[max(ks2),auc_value]

ks_pd=pd.DataFrame(ks_list,index=['ks','auc']).T
ks_pd=ks_pd.reset_index().rename(columns={'index':'applymonth'})

#r_p_chart2(y, pred_p3, min_scores, part=20)

ks_count=pd.merge(num_counts,ks_pd,on='applymonth',how='left')
ks_count.to_csv('./三期监控输出文件/固定6个月表现期模型的区分能力情况_'+ str(datetime.datetime.now().date()) + '.csv', index=False)

## 每月放款客户占比或者好坏客户占比
fin_data=testdata.copy()
fin_data['applymonth']=fin_data['app_applydate'].str.slice(0,7)
cuts = [0, 10.565, 10.862, 11.128, 11.471, 11.755, 12.117, 12.497, 12.935, 13.419, 14.096, 14.576, 15.199, 15.927,16.865, 18.172, 19.846, 21.782, 24.207, 28.348, 100]
fin_data['newgrp'] = pd.cut(fin_data.scores2, cuts, right=False)
total_num_counts=pd.crosstab(fin_data['newgrp'],fin_data['applymonth'],margins=True)
total_num_counts=total_num_counts.div(total_num_counts.ix['All',:],axis=1)

goodbad_data=fin_data[fin_data['y'].isin([0,1])].copy()
cuts = [0, 10.565, 10.862, 11.128, 11.471, 11.755, 12.117, 12.497, 12.935, 13.419, 14.096, 14.576, 15.199, 15.927,16.865, 18.172, 19.846, 21.782, 24.207, 28.348, 100]
goodbad_data['newgrp'] = pd.cut(goodbad_data.scores2, cuts, right=False)
goodbad_num_counts=pd.crosstab(goodbad_data['newgrp'],goodbad_data['applymonth'],margins=True)
goodbad_num_counts=goodbad_num_counts.div(goodbad_num_counts.ix['All',:],axis=1)