# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:38:54 2019

@author: wj56740
"""

from sklearn.externals import joblib
import datetime
import os
from pandas import DataFrame

# model_path = os.path.dirname(os.path.abspath(__file__))
#
# #添加模型文件路径
# clf = joblib.load(os.path.join(model_path, 'newcar3_xgb_model.pkl'))
lr = joblib.load( 'D:/llm/联合建模算法开发/自有数据_逻辑回归_剔除聚信立_增加若干车辆信息/CarLoan_OwnModel.pkl')


class model():
    def gbdt(self,data_json):

       # 函数输出变量
       status='0'
       msg=''
       res='-1'

       # 设置报错机制，若传入的为非json格式则报错
       if type(data_json)!=dict:
          status='1'
          msg='error：输入的参数不是json串格式！'
          return res,status,msg

       try:                   #统一将变量名修改为小写
            data_json1={}
            for cn in data_json.keys():
               data_json1[cn.lower()]=data_json[cn]
       except Exception as e:
               status = '1'
               msg = str(repr(e))
               return res, status, msg

       # 数值型变量
       x1 = ['vehicle_evtrpt_b2bprice','yix_other_orgcount','apt_ec_overduephasetotallastyear',
             'yx_org_inqry_tms','zix_aptpboc_worstodcycle','m3_pho_mulplatloan_allnum','apt_age']
       ##字符型变量
       x2=['vehicle_buymode','vehicle_minput_ownerway','vehicle_minput_attribution']

       #加工变量province_consume
       x3=['app_siteprovince']


       # 日期型变量
       dates = [	'zix_personal_email_time',
	             'apt_ec_lastloansettleddate',
	             'vehicle_minput_lastreleasedate']

       # 数值型变量默认值处理函数
       def fun(x):
           if x == '' or x == ' ' or float(x) == -1 or float(x) == -99 :
               return -99
           else:
               return float(x)

       # 字符型变量处理函数，统一处理成整型字符串形如'1'这种字符格式
       def str_fun(x):
            if x == '' or x == ' ' or float(x) == -1 or float(x) == -99 :
               return -99
            try:
                x=float(x)
            except:
                pass
            return x

       def fun_date(x, app_applydate):
            if str(x)=='9999-12-31 00:00:00' or str(app_applydate)=='9999-12-31 00:00:00':
                return -99
            x = x.replace('/', '-')
            try:
                x = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
            except:
                return -99
            app_applydate = app_applydate.replace('/', '-')
            try:
                 now_time = datetime.datetime.strptime(app_applydate, '%Y-%m-%d %H:%M:%S')
            except:
                return -99
            x = (now_time - x).days
            return x

       # 调用各种处理函数来处理变量
       try:
            x1 = [fun(data_json1[i]) for i in x1]
            x2=  [str_fun(data_json1[i]) for i in x2]
            dates = [fun_date(data_json1[i], data_json1['app_applydate']) for i in dates]
       except  Exception as e:
            status = '1'
            msg = str(repr(e)) + '，json串中缺少该变量'
            return res, status, msg
        
        
       #加工变量：age_and_gender、province_consume

       if (x1[6]>=39 and data_json1['apt_gender']=='男') or (data_json1['apt_gender']=='女'):
            x1[6]=0
       else:
            x1[6]=1

       province_consume_dist={}
       keys=['北京','天津','河北','山西','内蒙古','辽宁','吉林','黑龙江','上海','江苏','浙江','安徽','福建','江西','山东',
      '河南','湖北','湖南','广东','广西','海南','重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆']
       values=[134994,96965,65266,61547,67688,62545,62908,59995,130765,79741,82642,67927,69029,63069,69305,
              55997,67736,65994,80020,66456,69062,73272,71631,75109,73515,115549,67433,65726,76535,72779,68641]
       for key,value in zip(keys,values):
          province_consume_dist[key]=value  #加工变量province_consume
       if data_json1['app_siteprovince'] not in keys:
          x3[0]=-99
       else:
          x3[0]=province_consume_dist[data_json1['app_siteprovince']]


       #woe值转换
       def apt_ec_overduephasetotallastyear(x):
         if x == -99:
              return   -0.0311
         elif x <   1.0:
              return   -0.0311
         elif x <  2.0:
              return  0.7648
         elif x< 3.0:
              return   0.7648
         elif x< 4.0:
              return  0.7648
         elif x>= 4.0:
              return  0.7648

       def yx_org_inqry_tms(x):
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

       def zix_personal_email_time(x):
          if x == -99:
              return   -0.2087
          elif x < -29.0:
              return    0.3602
          elif x<  96.0:
              return    0.3602
          elif x < 424.0:
              return    0.3602
          elif x>= 424.0:
              return   -0.2087

       def zix_aptpboc_worstodcycle(x):
          if x == -99:
              return     -0.057
          elif x < 1.0:
             return     -0.057
          elif x<  2.0:
             return     1.2906
          elif x>= 2.0:
             return     1.2906

       def apt_ec_lastloansettleddate(x):
          if x == -99:
             return     0.0509
          elif x < -1.0:
             return     0.0509
          elif x<  0.0:
             return     0.0509
          elif x<  72.0:
             return     -0.1841
          elif x>= 72.0:
             return     -0.1841

       def m3_pho_mulplatloan_allnum(x):
          if x == -99:
             return  -0.4487
          elif x < 8.0:
             return   0.1125
          elif x< 26.0:
             return   0.1125
          elif x>= 26.0:
             return   0.1125

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

       def vehicle_buymode(x):
          if x == -99:
             return  -0.0068
          elif x ==1.0:
             return   -0.0068
          elif x==2.0:
             return  0.8142

       def vehicle_minput_ownerway(x):
          if x == -99:
             return    0.1628
          elif x== 1.0:
             return   -0.1669
          else:
             return    0.1628

       def vehicle_evtrpt_b2bprice(x):
          if x == -99:
             return    0.327
          elif x< 3.33:
             return    0.327
          elif x <4.3:
             return    0.327
          elif x < 10.6:
             return    -0.0902
          elif x>= 10.6:
             return    -0.0902

       def yix_other_orgcount(x):
          if x == -99:
             return    -0.1248
          elif x< 1.0:
             return   -0.1248
          elif x <3.0:
             return    0.4587
          elif x>= 3.0:
             return    1.0691

       def province_consume(x):
          if x == -99:
             return    -0.1262
          elif x< 65726.0:
             return    0.1898
          elif x <67927.0:
             return    0.1898
          elif x < 69062.0:
             return    0.1898
          elif x>= 69062.0:
             return    -0.1262

       def age_and_gender(x):
          if x == -99:
             return    -0.2182
          elif x==0:
             return    -0.2182
          elif x==1:
             return    0.0986

       def vehicle_minput_attribution(x):
          if x == -99:
            return    -0.0655
          elif x==1:
            return    -0.0655
          else:
            return    0.2294
    
       def apt_facetrial_housetype(x):
          if x == -99:
            return      0.1415
          elif x==1:
            return     -0.1982
          elif x==2:
            return     -0.1982
          elif x==3:
            return      0.1415
          elif x== 4:
            return      0.1415
          elif x==5:
            return      0.1415
          elif x==6:
            return      0.1415
          else:
            return      0.1415
        
       x2[0]=vehicle_buymode(x2[0])
       x2[1]=vehicle_minput_ownerway(x2[1])
       x1[0]=vehicle_evtrpt_b2bprice(x1[0])
       x1[1]=yix_other_orgcount(x1[1])
       x1[2]=apt_ec_overduephasetotallastyear(x1[2])
       x1[3]=yx_org_inqry_tms(x1[3])
       dates[0]=zix_personal_email_time(dates[0])
       x1[4]=zix_aptpboc_worstodcycle(x1[4])
       x3[0]=province_consume(x3[0])
       x1[6]=age_and_gender(x1[6])
       x2[2]=vehicle_minput_attribution(x2[2])
       dates[1]=apt_ec_lastloansettleddate(dates[1])
       x1[5]=m3_pho_mulplatloan_allnum(x1[5])
       dates[2]=vehicle_minput_lastreleasedate(dates[2])

      # 全部入模变量
       columns=['vehicle_buymode','vehicle_minput_ownerway','vehicle_evtrpt_b2bprice','yix_other_orgcount',
                'apt_ec_overduephasetotallastyear','yx_org_inqry_tms','zix_personal_email_time','zix_aptpboc_worstodcycle',
                'province_consume','age_and_gender','vehicle_minput_attribution','apt_ec_lastloansettleddate',
                'm3_pho_mulplatloan_allnum','vehicle_minput_lastreleasedate']
       columns1=['vehicle_buymode','vehicle_minput_ownerway','vehicle_evtrpt_b2bprice','qtorg_query_orgcnt',
                'apt_ec_overduephasetotallastyear','times_by_current_org','email_info_date','max_overdue_terms',
                'province_consume','age_and_gender','vehicle_minput_attribution','apt_ec_lastloansettleddate',
                'm3_apply_mobile_platform_cnt','vehicle_minput_lastreleasedate']
       rename_list={}
       for key,value in zip(columns,columns1):
           rename_list[key]=value


      # 调用模型，输出分数
       x = x2[0:2]+x1[0:4]+[dates[0],x1[4],x3[0],x1[6],x2[2],dates[1],x1[5],dates[2]]
       try:
          res = float(lr.predict_proba(DataFrame(x,index=columns).T.rename(columns=rename_list))[:, 1]) * 100.0
       except Exception as e:
          status='1'
          msg=str(repr(e))
          return res,status,msg

       return '%.3f' % res,status,msg


# '''
# 实例
# '''

data_json={'vehicle_buymode': '1',
'vehicle_minput_ownerway': '3',
'vehicle_evtrpt_b2bprice': 28.41,
'yix_other_orgcount': '1.0',
'apt_ec_overduephasetotallastyear': '0',
'yx_org_inqry_tms': 1.0,
'zix_personal_email_time': '2017-12-28 00:00:00',
'zix_aptpboc_worstodcycle': 0.0,
'app_siteprovince': '甘肃',
'apt_age': 24.0,
'apt_gender': '男',
'vehicle_minput_attribution': '1.0',
'apt_ec_lastloansettleddate':  '-99',
'm3_pho_mulplatloan_allnum': -99,
'vehicle_minput_lastreleasedate':'-99',
'app_applydate':'2018-03-01 12:30:57'}


print(model().gbdt(data_json))











