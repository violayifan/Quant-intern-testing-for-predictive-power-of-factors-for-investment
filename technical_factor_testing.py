# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:49:11 2020

@author: Viola
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import scipy.stats as ss

import seaborn as sns

import random

#spearman correlation 
def spearman_correlation(factor,l_rt,df):
    # overall
    
    df_rst_overall=pd.DataFrame()
    for rt in l_rt:
        df_1=df.dropna(subset=[rt])
        if len(df_1) !=0:
            dic={}
            dic["Return"]=rt
            dic["Spearman Correlation"]=ss.spearmanr(df_1[factor],df_1[rt])[0]
            dic["P value"]=ss.spearmanr(df_1[factor],df_1[rt])[1]
            df_rst_overall=df_rst_overall.append(pd.DataFrame([dic]))
    
    # monthly
    df_rst_monthly=pd.DataFrame()
    month=np.sort(df["month"].unique())
    for m in month:
        df_sub=df[df["month"]==m]
        for rt in l_rt:
            df_sub=df_sub.dropna(subset=[rt])
            if len(df_sub) !=0:
                dic={}
                dic["Month"]=m
                dic["Return"]=rt
                dic["Spearman Correlation"]=ss.spearmanr(df_sub[factor],df_sub[rt])[0]
                dic["P value"]=ss.spearmanr(df_sub[factor],df_sub[rt])[1]
                df_rst_monthly= df_rst_monthly.append(pd.DataFrame([dic]))
                print("Completed {} correlation calculation".format(m))
    return df_rst_overall, df_rst_monthly

def linear_correlation(factor,l_rt,df):
    # overall
    df_rst_overall=pd.DataFrame()
    for rt in l_rt:
        df_1=df.dropna(subset=[rt])
        if len(df_1) !=0 :
            slope,_,r2,pvalue,_=ss.linregress(df_1[factor],df_1[rt])
            dic={}
            dic["Return"]=rt
            dic["Factor Return"]=slope
            dic["Linear Correlation"]=r2
            dic["P value"]=pvalue
            df_rst_overall=df_rst_overall.append(pd.DataFrame([dic]))
    
    # monthly
    df_rst_monthly=pd.DataFrame()
    month=np.sort(df["month"].unique())
    for m in month:
        df_sub=df[df["month"]==m]
        if len(df_sub) !=0:
            for rt in l_rt:
                df_sub=df_sub.dropna(subset=[rt])
                if len(df_sub) !=0:
                    slope,_,r2,pvalue,_=ss.linregress(df_sub[factor],df_sub[rt])
                    dic={}
                    dic["Month"]=m
                    dic["Return"]=rt
                    dic["Return"]=rt
                    dic["Factor Return"]=slope
                    dic["Linear Correlation"]=r2
                    dic["P value"]=pvalue
                    df_rst_monthly= df_rst_monthly.append(pd.DataFrame([dic]))
                    print("Completed {} correlation calculation".format(m))
    return df_rst_overall, df_rst_monthly

#long stock with positive signal
def port_long(df,signal,rebalancing_day,yearly_details=False):
    l_date=np.sort(df["date"].unique())
    l_date=pd.to_datetime(l_date, format="%Y-%m-%d")
    df=df.set_index(["date"])
    if yearly_details:
        dic_rt_year={}
        for year in np.sort(data["year"].unique()):
            df_rt_year=df[df["year"]==year]
            l_date_year=np.sort(df_rt_year.index.unique())
            l_date_year=pd.to_datetime(l_date_year, format="%Y-%m-%d")
            dic_rt={}
            for i in range(0,len(l_date_year)-rebalancing_day,rebalancing_day):
                df1=df.loc[l_date_year[i]]
                df2=df.loc[l_date_year[i+rebalancing_day]]

                ticker_long_1=df1[df1[signal]==1]["ticker"].unique()
                ticker_long_2=df2[df2["ticker"].isin(ticker_long_1)]["ticker"].unique()
        
                ticker=set(ticker_long_1) & set (ticker_long_2)
            
                port_long_1=df1[df1["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
                port_long_2=df2[df2["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
            
                if len(ticker) != 0:
                    l_rt=(port_long_2-port_long_1)/port_long_1
                    #equally weighted
                    w=1/len(ticker)
                    l_rt=w*l_rt
                    dic_rt[l_date_year[i+rebalancing_day]]=l_rt.sum()
                    print("completed date {}".format(l_date_year[i+rebalancing_day]))
            df_rt=pd.DataFrame([dic_rt]).T
            df_rt.columns=["return"]
            df_rt['cumrt']=np.cumprod(df_rt["return"]+1)
            if len(df_rt)!=0:
                dic_rt_year[year]=df_rt
                print("completed year {}".format(year))
        return dic_rt_year      
    else:
        dic_rt={}
        dic_count={}
        for i in range(0,len(l_date)-rebalancing_day,rebalancing_day):
            
            df1=df.loc[l_date[i]]
            df2=df.loc[l_date[i+rebalancing_day]]

            ticker_long_1=df1[df1[signal]==1]["ticker"].unique()
        
            ticker_long_2=df2[df2["ticker"].isin(ticker_long_1)]["ticker"].unique()
        
            ticker=set(ticker_long_1) & set (ticker_long_2)
            
            port_long_1=df1[df1["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
            port_long_2=df2[df2["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
            
            if len(ticker) != 0:
                l_rt=(port_long_2-port_long_1)/port_long_1
                #equally weighted
                w=1/len(ticker)
                l_rt=w*l_rt
                dic_rt[l_date[i+rebalancing_day]]=l_rt.sum()
                dic_count[l_date[i+rebalancing_day]]=len(ticker)
                print("completed date {}".format(l_date[i+rebalancing_day]))
            df_rt=pd.DataFrame([dic_rt]).T
            df_rt.columns=["return"]
            df_rt['cumrt']=np.cumprod(df_rt["return"]+1)
        df_count=pd.DataFrame([dic_count]).T
        return df_rt,df_count

#long all stock
#return cumulative return,number of trades during the whole backtesting period/each year
def port_long_group(df,group_name,rebalancing_day,yearly_details=False):
    l_date=np.sort(df["date"].unique())
    l_date=pd.to_datetime(l_date, format="%Y-%m-%d")
    df=df.set_index(["date"])
    if yearly_details:
        dic_rt_year={}
        for year in np.sort(data["year"].unique()):
            df_rt_year=df[df["year"]==year]
            l_date_year=np.sort(df_rt_year.index.unique())
            l_date_year=pd.to_datetime(l_date_year, format="%Y-%m-%d")
            dic_rt={}
            for group in df[group_name].unique():
                dic_group={}
                for i in range(0,len(l_date_year)-rebalancing_day,rebalancing_day):
                    df1=df.loc[l_date_year[i]]
                    df2=df.loc[l_date_year[i+rebalancing_day]]

                    ticker_long_1=df1[df1[group_name]==group]["ticker"].unique()
            
                    ticker_long_2=df2[df2["ticker"].isin(ticker_long_1)]["ticker"].unique()
            
                    ticker=set(ticker_long_1) & set (ticker_long_2)
                
                    port_long_1=df1[df1["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
                    port_long_2=df2[df2["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
                
                    if len(ticker) != 0:
                        l_rt=(port_long_2-port_long_1)/port_long_1
                        #equally weighted
                        w=1/len(ticker)
                        l_rt=w*l_rt
                        dic_group[l_date_year[i+rebalancing_day]]=l_rt.sum()
                        print("completed date {}".format(l_date_year[i+rebalancing_day]))
                df_rt=pd.DataFrame([dic_group]).T
                df_rt.columns=["return"]
                df_rt['cumrt']=np.cumprod(df_rt["return"]+1)
                dic_rt[group]=df_rt
                print("completed group {}".format(group))
            dic_rt_year[year]=dic_rt 
            print("completed year {}".format(year))
        return dic_rt_year      
    else:
        dic_rt={}
        dic_count_all={}
        for group in df[group_name].unique():
            dic_group={}
            dic_count={}
            for i in range(0,len(l_date)-rebalancing_day,rebalancing_day):
                
                df1=df.loc[l_date[i]]
                df2=df.loc[l_date[i+rebalancing_day]]

                ticker_long_1=df1[df1[group_name]==group]["ticker"].unique()
            
                ticker_long_2=df2[df2["ticker"].isin(ticker_long_1)]["ticker"].unique()
            
                ticker=set(ticker_long_1) & set (ticker_long_2)
                
                port_long_1=df1[df1["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
                port_long_2=df2[df2["ticker"].isin(ticker)][["ticker","close"]].set_index(["ticker"])["close"]
                
                if len(ticker) != 0:
                    l_rt=(port_long_2-port_long_1)/port_long_1
                    #equally weighted
                    w=1/len(ticker)
                    l_rt=w*l_rt
                    dic_group[l_date[i+rebalancing_day]]=l_rt.sum()
                    dic_count[l_date[i+rebalancing_day]]=len(ticker)
                    print("completed date {}".format(l_date[i+rebalancing_day]))
                df_rt=pd.DataFrame([dic_group]).T
                df_rt.columns=["return"]
                df_rt['cumrt']=np.cumprod(df_rt["return"]+1)
            dic_rt[group]=df_rt
            dic_count_all[group]=pd.DataFrame([dic_count]).T
            print("completed group {}".format(group))
        return dic_rt,dic_count_all



#randomly simulated portfolio 
def port_simulated_group(df,group_name,rebalancing_day,simulated_stock_number,simulated_times):
    
    rst=[]
    l_date=np.sort(df["date"].unique())
    l_date=pd.to_datetime(l_date, format="%Y-%m-%d")
    df=df.set_index(["date"])
    for n in range(simulated_times) : 
        dic_rst={}
        for group in df[group_name].unique():
            dic_group={}
            for i in range(0,len(l_date)-rebalancing_day,rebalancing_day):
                
                df1=df.loc[l_date[i]]
                df2=df.loc[l_date[i+rebalancing_day]]
                ticker_long_1=df1[df1[group_name]==group]["ticker"].unique()
                ticker_long_2=df2[df2["ticker"].isin(ticker_long_1)]["ticker"].unique()
            
                ticker=list(set(ticker_long_1) & set (ticker_long_2))
                
                #random portfolio
                if len(ticker) < simulated_stock_number:
                    simulated_stock_number= len(ticker)
                l_stock=np.array(ticker)[random.sample(range(0,len(ticker)),simulated_stock_number)]
                port_long_1=df1[df1["ticker"].isin(l_stock)][["ticker","close"]].set_index(["ticker"])["close"]
                port_long_2=df2[df2["ticker"].isin(l_stock)][["ticker","close"]].set_index(["ticker"])["close"]
                
                l_rt=(port_long_2-port_long_1)/port_long_1
                #equally weighted
                w=1/len(ticker)
                l_rt=w*l_rt
                dic_group[l_date[i+rebalancing_day]]=l_rt.sum()         
                print("completed date {}".format(l_date[i+rebalancing_day]))
                df_rt=pd.DataFrame([dic_group]).T
                df_rt.columns=["return"]
                df_rt['cumrt']=np.cumprod(df_rt["return"]+1)
            dic_rst[group]=df_rt 
            print("completed group {}".format(group))
        rst.append(dic_rst)
        print("completed simulation {}".format(n))
    return rst



# benchmark yearly/overall performance
def benchmark_perform(df_benchmark,rebalancing_day,start,end,yearly_details=False):
    
    df_benchmark_test=df_benchmark[(df_benchmark["date"]>=start) & (df_benchmark["date"]<=end)]
    if yearly_details:
        dic_rst_year={}
        for year in np.sort(df_benchmark_test["year"].unique()):
            df_benchmark_test_year= df_benchmark_test[df_benchmark_test["year"]==year]
            df_benchmark_test_year["return"]=df_benchmark_test_year["price_index"].pct_change(rebalancing_day)
            df_benchmark_test_year.dropna(inplace=True)
            df_benchmark_test_year=df_benchmark_test_year.iloc[::rebalancing_day]
            df_benchmark_test_year["cumrt"]=np.cumprod(df_benchmark_test_year["return"]+1)
            df_benchmark_test_year=df_benchmark_test_year[["date","return","cumrt"]].set_index(["date"])
            dic_rst_year[year]=df_benchmark_test_year
        return dic_rst_year
    else:
        df_benchmark_test["return"]=df_benchmark_test["price_index"].pct_change(rebalancing_day)
        df_benchmark_test.dropna(inplace=True)
        df_benchmark_test=df_benchmark_test.iloc[::rebalancing_day]
        df_benchmark_test["cumrt"]=np.cumprod(df_benchmark_test["return"]+1)
        df_benchmark_test=df_benchmark_test[["date","return","cumrt"]].set_index(["date"])
        return df_benchmark_test
    


#plot the backtesting result and compare it with the benchmark
def plot_grouped(dic_rt,df_benchmark,group_name,rebalancing_day):
    
    l_rt_all=[]
    for group,df in dic_rt.items():
        df_sub=df[["cumrt"]]
        if type(group) == np.float64:
            df_sub.rename(columns={"cumrt": group_name + str(int(group))},inplace=True)
        else:
            df_sub.rename(columns={"cumrt": group_name + str(group)},inplace=True)
        l_rt_all.append(df_sub)
    
    #benchmark cumulative return
    def get_index_rt(df_benchmark):
        df_benchmark=df_benchmark.copy()
        df_benchmark.set_index(["date"],inplace=True)
        df_benchmark=df_benchmark.loc[l_rt_all[0].index]
        df_benchmark["return"]=df_benchmark["price_index"].pct_change()+1
        df_benchmark["MSCI All Country World Index"]=np.cumprod(df_benchmark["return"])
        return df_benchmark
    
    df_benchmark=get_index_rt(df_benchmark)
    
    l_rt_all.append(df_benchmark[["MSCI All Country World Index"]])
    df_rt_all=pd.concat(l_rt_all,axis=1,join="inner")
    
    #plot
    plt.figure(figsize=(20,15))    
    plt.style.use('seaborn')
    sns.set(font_scale=3)
    for col in  df_rt_all.columns:
        #ax = sns.lineplot(data=df_plot)
        plt.plot(df_rt_all[col],label=col)    
        plt.xlabel("year",fontsize=30)
        plt.ylabel("Cumulative Return",fontsize=30)
        plt.title("Cumulative Return of {} across the years with {}-day rebalancing frequency".format(group_name,rebalancing_day),fontsize=30)
        plt.legend(loc="upper left")


class performance_matrix(object):
    
    #annualized
    def Sharpe_ratio(self,r_mean,std,freq_day):
        multiple=360/freq_day
        sharpe=(r_mean*multiple)/(std*np.sqrt(multiple))
        return sharpe
    
    def max_drawdown(self,net_value):
        Roll_Max = net_value.cummax()
        Daily_Drawdown = net_value/Roll_Max - 1
        Max_Daily_Drawdown = Daily_Drawdown.cummin()
        return Max_Daily_Drawdown[-1]
        
           
    def trade_number_line(self,df_count,lab):
        
        plt.figure(figsize=(20,15))    
        plt.style.use('seaborn')
        sns.set(font_scale=3)
        plt.plot(df_count,label=lab)
        plt.xlabel("year",fontsize=30)
        plt.ylabel("Number of trades",fontsize=30)
        plt.title("Number of trades for {}".format(lab),fontsize=30)
        plt.legend(loc="upper right")

    def success_ratio(self,df_rt):
        a=np.array(df_rt)
        number_success=len(a[a>0])
        total_number=len(df_rt)
        success_ratio=number_success/total_number
        return success_ratio
    
    def CAGR(self,df_rt_cum,year_period):
        return (df_rt_cum.values[-1]/1)**(1/year_period)-1
    
    
        





if __name__=="__main__":
    
    '''
    Data Processing
    '''
    
    data=pd.read_csv('cleaned_data.csv',usecols=['date','ticker','company_name','technical_attribute',\
                                                 'daily_momentum','weekly_momentum','monthly_momentum',\
                                                   "50_day_ma",'150_day_ma','200_day_ma',"weekly_obos"])
    
    data=data[['date','ticker','company_name','technical_attribute',\
                                                 'daily_momentum','weekly_momentum','monthly_momentum',\
                                                   "50_day_ma",'150_day_ma','200_day_ma',"weekly_obos"]]
    
    
    data["date"]=pd.to_datetime(data["date"],format="%Y-%m-%d")

    data=data.sort_values(by=["ticker","date"]).reset_index(drop=True)
    
    data["year"]=data["date"].apply(lambda x :x.year).values
    data["month"]=data["date"].apply(lambda x :x.to_period('M')).values
    
        
    # get historical closing price
    df_hist_price=pd.read_csv("historicalprices.csv")
    df_hist_price=df_hist_price.T
    df_hist_price.columns=df_hist_price.iloc[0]
    df_hist_price.drop(index="RIC",inplace=True)
    
    df_hist_price=df_hist_price.stack()
    df_hist_price=df_hist_price.reset_index()
    df_hist_price.columns=["date","RIC","close"]
    df_hist_price["date"]=pd.to_datetime(df_hist_price["date"],format="%m/%d/%Y")
    
    #drop missing value
    df_hist_price["close"]=[i if isinstance(i,float) else np.nan for i in df_hist_price["close"]]
    df_hist_price.dropna(inplace=True)
    
    df_hist_price.drop_duplicates(inplace=True)
    # calculate forward precent returns
    df_hist_price_unstack= df_hist_price.set_index(["date","RIC"]).unstack(level="RIC")
    df_hist_price_unstack.reset_index(inplace=True)
    df_hist_price_unstack.columns= [i[1] for i in df_hist_price_unstack.columns]
    df_hist_price_unstack.rename(columns={"":"date"},inplace=True)
    df_hist_price_unstack.set_index("date",inplace=True)
    
    df_5d=df_hist_price_unstack.pct_change(5).shift(-5)
    df_21d=df_hist_price_unstack.pct_change(21).shift(-21)
    df_63d=df_hist_price_unstack.pct_change(63).shift(-63)
    df_126d=df_hist_price_unstack.pct_change(126).shift(-126)

    df_forward_rt=pd.concat([df_5d.stack(),df_21d.stack(),df_63d.stack(),df_126d.stack()],axis=1)
    df_forward_rt.reset_index(inplace=True)
    df_forward_rt.columns=["date","RIC","5_day_forward_return","21_day_forward_return","63_day_forward_return","126_day_forward_return"]
    
    df_price_rt= df_hist_price.merge(df_forward_rt,how="left",on=["date",'RIC'])

    
    # mapping rule
    df_ticker=pd.read_csv("isins.csv")
    df_price_rt= df_price_rt.merge(df_ticker,how="left",on='RIC').drop_duplicates(subset=["ticker","date"])
    df_price_rt.drop(columns="RIC",inplace=True)
    

    # combine data & history price
    data=data.merge(df_price_rt,how="left",on=["date","ticker"]).drop_duplicates()
    
    # input benchmark data
    benchmark=pd.read_excel('msciworLD.xlsx')
    benchmark["date"]=pd.to_datetime(benchmark["date"],format="%Y-%m-%d")

    benchmark["year"]=benchmark["date"].apply(lambda x :x.year).values
    benchmark["month"]=benchmark["date"].apply(lambda x :x.to_period('M')).values

    #define performance metrix
    perform_measure=performance_matrix()
    
    # benchmark performance
    #5_day rebalance
    #(1) yearly
    dic_benchmark_5d_year=benchmark_perform(benchmark,5,data["date"].min(),data["date"].max(),yearly_details=True)
    #(2) overall
    df_benchmark_5d=benchmark_perform(benchmark,5,data["date"].min(),data["date"].max(),yearly_details=False)

    #21_day rebalance
    #(1) yearly
    dic_benchmark_21d_year=benchmark_perform(benchmark,21,data["date"].min(),data["date"].max(),yearly_details=True)
    #(2) overall
    df_benchmark_21d=benchmark_perform(benchmark,21,data["date"].min(),data["date"].max(),yearly_details=False)
    #63_day rebalance
    #(1) yearly
    dic_benchmark_63d_year=benchmark_perform(benchmark,63,data["date"].min(),data["date"].max(),yearly_details=True)
    #(2) overall
    df_benchmark_63d=benchmark_perform(benchmark,63,data["date"].min(),data["date"].max(),yearly_details=False)
    #126_day rebalance
    #(1) yearly
    dic_benchmark_126d_year=benchmark_perform(benchmark,126,data["date"].min(),data["date"].max(),yearly_details=True)
    #(2) overall
    df_benchmark_126d=benchmark_perform(benchmark,126,data["date"].min(),data["date"].max(),yearly_details=False)
    
    #rebalancing day list
    l_rebalancing_day=[5,21,63,126]
    # CAGR yearly
    l_dic_benchmark_year=[dic_benchmark_5d_year,dic_benchmark_21d_year,dic_benchmark_63d_year,dic_benchmark_126d_year]
    
    def CAGR_yearly(l_dic_year,l_rebalancing_day):
        dic_rst={}
        for dic_year,day in zip(l_dic_year,l_rebalancing_day):
            dic_sub={}
            for year,df in dic_year.items():
                if len (df) !=0:
                    dic_sub[year]=perform_measure.CAGR(df["cumrt"],1)
            dic_rst["CAGR_yearly_{}_day".format(str(day))]=pd.DataFrame([dic_sub]).T
        return dic_rst
               
    dic_benchmark_CAGR_yearly=CAGR_yearly(l_dic_benchmark_year,l_rebalancing_day)  
        
    # performance summary
    l_df_benchmark=[df_benchmark_5d,df_benchmark_21d,df_benchmark_63d,df_benchmark_126d]
    def perform_summary(l_df,l_rebalancing_day,year_period=(data["date"].max()-data["date"].min()).days/365):
        dic_perform_summary={}
        for df,day in zip(l_df,l_rebalancing_day):
            # CAGR overall
             dic_perform_summary["CAGR_{}_day".format(str(day))]=perform_measure.CAGR(df["cumrt"],year_period)
             # sharpe ratio
             dic_perform_summary["Sharpe_ratio_{}_day".format(str(day))]=perform_measure.Sharpe_ratio(df["return"].mean(),df["return"].std(),day)
             # maximum drawdown
             dic_perform_summary["Max_drawdown_{}_day".format(str(day))]=perform_measure.max_drawdown(df["cumrt"])
             #success ratio
             dic_perform_summary["Success_ratio_{}_day".format(str(day))]=perform_measure.success_ratio(df["return"])    
        df_perform_summary=pd.DataFrame([dic_perform_summary])  
        return df_perform_summary
    
    df_benchmark_perform_summary=perform_summary(l_df_benchmark,l_rebalancing_day).T
    
    #remove stocks whose historical price <1
    l_remove_stock=data[data["close"]<1]["ticker"].unique()
    data=data[~data["ticker"].isin(l_remove_stock)]
   
    '''
    Factor testing--technical attribute
    '''  

    #forward return analysis
    #plot monthly summary of forward return
    df_ta_m=data[["technical_attribute","month",\
          "5_day_forward_return","21_day_forward_return","63_day_forward_return","126_day_forward_return"]].groupby(["technical_attribute","month"]).mean()

    for rt in df_ta_m.columns:
    
        plt.figure(figsize=(20,15))    
        plt.style.use('seaborn')
        sns.set(font_scale=3)
        df_sub=df_ta_m[rt]
        df_sub=df_sub.reset_index()
        df_sub["month"]=pd.to_datetime(df_sub["month"].astype("str"),format="%Y-%m")
        df_sub["technical_attribute"]="attribute "+ df_sub["technical_attribute"].astype("str")
        ax = sns.lineplot(x="month",y=rt,data=df_sub,hue="technical_attribute",palette = 'RdBu_r')
        
        plt.xlabel("year",fontsize=50)
        plt.ylabel(rt,fontsize=50)
        plt.title("Forward return evolution of technical attribute across the years",fontsize=50)
        plt.legend(loc="upper right")
        
    #technical attribute distribution
    stats_sum={}
    ticker_count=data.groupby(["technical_attribute","date"])["ticker"].count().reset_index()
    for ta in ticker_count["technical_attribute"].unique():
        plt.figure(figsize=(20,15)) 
        plt.style.use('seaborn')
        sns.set(font_scale=3)
        plt.hist(ticker_count[ticker_count["technical_attribute"]==ta]["ticker"],color="c")
        stats_sum[ta]=ticker_count[ticker_count["technical_attribute"]==ta]["ticker"].describe()
        plt.xlabel("Count of technical attribute every day",fontsize=50)
        plt.ylabel("Number",fontsize=50)
        plt.title("Distribution of technical attribute {}".format(ta),fontsize=50)

    
    #overall average return grouped by value of  technical attribute 
    df_ta_all=data[["technical_attribute","5_day_forward_return","21_day_forward_return","63_day_forward_return","126_day_forward_return"]].groupby(["technical_attribute"]).mean()
    #plot bar chart
    for rt in df_ta_all.columns:
        plt.figure(figsize=(20,15))    
        plt.style.use('seaborn')
        sns.set(font_scale=3)
        df_sub=df_ta_all[rt]
        df_sub=df_sub.reset_index()
        df_sub["technical_attribute"]="attribute "+ df_sub["technical_attribute"].astype("str")
        #ax = sns.barplot(x="technical_attribute",y=rt,data=df_sub)
        plt.bar(df_sub["technical_attribute"],df_sub[rt])
        plt.ylabel(rt,fontsize=50)
        plt.title("Forward return of technical attribute",fontsize=50)
    
    # cumulative return for each year
    #(1)5_days_rets
    dic_ta_5days_year=port_long_group(data,"technical_attribute",5,yearly_details=True)
    #(2)21_days_rets
    dic_ta_21days_year=port_long_group(data,"technical_attribute",21,yearly_details=True)
    #(3)63_days_rets
    dic_ta_63days_year=port_long_group(data,"technical_attribute",63,yearly_details=True)
    #(4)126_days_rets
    dic_ta_126days_year=port_long_group(data,"technical_attribute",126,yearly_details=True)

 
    # cumulative return throughout the whole backtesting period
    rst_ta_5days=port_long_group(data,"technical_attribute",5,yearly_details=False)
    rst_ta_21days=port_long_group(data,"technical_attribute",21,yearly_details=False)
    rst_ta_63days=port_long_group(data,"technical_attribute",63,yearly_details=False)
    rst_ta_126days=port_long_group(data,"technical_attribute",126,yearly_details=False)

    #(1)5_days_rets
    dic_ta_5days_ret=rst_ta_5days[0]
    #(2)21_days_rets
    dic_ta_21days_ret=rst_ta_21days[0]
    #(3)63_days_rets
    dic_ta_63days_ret=rst_ta_63days[0]
    #(4)126_days_rets
    dic_ta_126days_ret=rst_ta_126days[0]
    
    plot_grouped(dic_ta_5days_ret,benchmark,"technical_attribute",5)
    plot_grouped(dic_ta_21days_ret,benchmark,"technical_attribute",21)
    plot_grouped(dic_ta_63days_ret,benchmark,"technical_attribute",63)
    plot_grouped(dic_ta_126days_ret,benchmark,"technical_attribute",126)


    #simulated portfoilo
    simulated_ta_5days=port_simulated_group(data,"technical_attribute",5,60,50)
    
    simulated_ta_21days=port_simulated_group(data,"technical_attribute",21,60,50)

    simulated_ta_63days=port_simulated_group(data,"technical_attribute",63,60,50)

    simulated_ta_126days=port_simulated_group(data,"technical_attribute",126,60,50)


    # cumulative average growth return yearly
    dic_ta_CAGR={}
    for group in range(0,6):
        l_dic_year_ta=[]
        for rst_ta_dic in [dic_ta_5days_year,dic_ta_21days_year,dic_ta_63days_year,dic_ta_126days_year]:
            dic_sub={}
            for year, dic in rst_ta_dic.items():
                dic_sub[year]=dic[group]
            l_dic_year_ta.append(dic_sub)
            dic_ta_CAGR[group]=CAGR_yearly(l_dic_year_ta,l_rebalancing_day)
    
    # CAGR, max drawdown,annualized sharpe ratio,success ratio
    dic_ta_perform={}
    for group in range(0,6):
        l_df_ta=[]
        for df in [dic_ta_5days_ret,dic_ta_21days_ret,dic_ta_63days_ret,dic_ta_126days_ret]:
            l_df_ta.append(df[group])
        
        dic_ta_perform[group]=perform_summary(l_df_ta,l_rebalancing_day).T
    
    
    
    # CAGR & its distribution of randam portfolio for each technical attributes and different rebalancing period
    l_simulated_rst=[simulated_ta_5days,simulated_ta_21days,simulated_ta_63days,simulated_ta_126days]
    dic_simulated_CAGR={}
    for simulated_rst,day in zip(l_simulated_rst,l_rebalancing_day):
        dic_simulated_group={}
        for group in range(0,6):
            l_ta_CAGR=[]
            for s in simulated_rst:
                l_ta_CAGR.append(perform_measure.CAGR(s[group]["cumrt"],(data["date"].max()-data["date"].min()).days/365))
            dic_simulated_group[group]=l_ta_CAGR
        dic_simulated_CAGR["CAGR_{}_day".format(str(day))]=dic_simulated_group
    
    #calcalate the average of simulated CAGR
    dic_mean_CAGR={}
    for re_day,dict_group in dic_simulated_CAGR.items():
        dic_mean={}
        for group,CAGR in dict_group.items():
            dic_mean[group]=np.mean(CAGR)
        dic_mean_CAGR[re_day]=dic_mean
        
    #calcalate the range of simulated CAGR
    dic_range_CAGR={}
    for re_day,dict_group in dic_simulated_CAGR.items():
        dic_range={}
        for group,CAGR in dict_group.items():
            dic_range[group]=[np.min(CAGR),np.max(CAGR)]
        dic_range_CAGR[re_day]=dic_range

          
    #plot distribution
    for re_day,dict_group in dic_simulated_CAGR.items():
        for group,CAGR in dict_group.items():
            plt.figure(figsize=(20,15))    
            plt.style.use('seaborn')
            sns.set(font_scale=3)   
            plt.hist(CAGR,color="c")
            plt.title("Distribution for CAGR of technical attribute {} with {}-day rebalancing frequency".format(group,re_day.split("_")[1]),fontsize=30)

    # dic_ta_perform_simulated={}
    # for group in range(0,6):
    #     l_df_ta_simulated=[]
    #     for df in [simulated_ta_5days[0],simulated_ta_21days[0],simulated_ta_63days[0],simulated_ta_126days[0]]:
    #         l_df_ta_simulated.append(df[group])
        
    #     dic_ta_perform_simulated[group]=perform_summary(l_df_ta_simulated,l_rebalancing_day).T
          
    # trades analysis
    #(1)5_days_rets
    dic_ta_5days_trades=rst_ta_5days[1]
    #(2)21_days_rets
    dic_ta_21days_trades=rst_ta_21days[1]
    #(3)63_days_rets
    dic_ta_63days_trades=rst_ta_63days[1]
    #(4)126_days_rets
    dic_ta_126days_trades=rst_ta_126days[1]
    
    #average trade & plots
    dic_trade_mean_ta={}
    for dic_trade,day in zip([dic_ta_5days_trades,dic_ta_21days_trades,dic_ta_63days_trades,dic_ta_126days_trades],l_rebalancing_day):
        dic_sub_count={}
        for group,df_count in dic_trade.items():
           lab="technical attribute {} with {}-day rebalancing frequency".format(group,str(day))
           perform_measure.trade_number_line(df_count,lab)
           dic_sub_count[group]=df_count.mean().values[0]
        dic_trade_mean_ta[day]=pd.DataFrame([dic_sub_count]).T
        
    #1 year outperformance
    dic_1_year_outperform={}
    for group,dic_CAGR_yearly in dic_ta_CAGR.items():
        dic_outperform={}
        for re_day,df_CAGR in dic_CAGR_yearly.items():
            outperform=(df_CAGR-dic_benchmark_CAGR_yearly[re_day]).values
            dic_outperform[re_day]=len(outperform[outperform>0])/len(outperform)
        dic_1_year_outperform[group]=dic_outperform
            
        
    '''
    Factor testing--daily momentum
    '''  
    #(1) 5 day
    dic_dm_5d_yearly=port_long(data,"daily_momentum",5,yearly_details=True)
    dic_dm_5d=port_long(data,"daily_momentum",5,yearly_details=False)
    #(2) 21 day
    dic_dm_21d_yearly=port_long(data,"daily_momentum",21,yearly_details=True)
    dic_dm_21d=port_long(data,"daily_momentum",21,yearly_details=False)

    #(3) 63 day
    dic_dm_63d_yearly=port_long(data,"daily_momentum",63,yearly_details=True)
    dic_dm_63d=port_long(data,"daily_momentum",63,yearly_details=False)
    #(4) 126 day
    dic_dm_126d_yearly=port_long(data,"daily_momentum",126,yearly_details=True)
    dic_dm_126d=port_long(data,"daily_momentum",126,yearly_details=False)

        
    '''
    Factor testing--weekly momentum
    ''' 
    #(1) 5 day
    dic_wm_5d_yearly=port_long(data,"weekly_momentum",5,yearly_details=True)
    dic_wm_5d=port_long(data,"weekly_momentum",5,yearly_details=False)
    #(2) 21 day
    dic_wm_21d_yearly=port_long(data,"weekly_momentum",21,yearly_details=True)
    dic_wm_21d=port_long(data,"weekly_momentum",21,yearly_details=False)

    #(3) 63 day
    dic_wm_63d_yearly=port_long(data,"weekly_momentum",63,yearly_details=True)
    dic_wm_63d=port_long(data,"weekly_momentum",63,yearly_details=False)
    #(4) 126 day
    dic_wm_126d_yearly=port_long(data,"weekly_momentum",126,yearly_details=True)
    dic_wm_126d=port_long(data,"weekly_momentum",126,yearly_details=False)


    '''
    Factor testing--monthly momentum
    '''  
    #(1) 5 day
    dic_mm_5d_yearly=port_long(data,"monthly_momentum",5,yearly_details=True)
    dic_mm_5d=port_long(data,"monthly_momentum",5,yearly_details=False)
    #(2) 21 day
    dic_mm_21d_yearly=port_long(data,"monthly_momentum",21,yearly_details=True)
    dic_mm_21d=port_long(data,"monthly_momentum",21,yearly_details=False)

    #(3) 63 day
    dic_mm_63d_yearly=port_long(data,"monthly_momentum",63,yearly_details=True)
    dic_mm_63d=port_long(data,"monthly_momentum",63,yearly_details=False)
    #(4) 126 day
    dic_mm_126d_yearly=port_long(data,"monthly_momentum",126,yearly_details=True)
    dic_mm_126d=port_long(data,"monthly_momentum",126,yearly_details=False)

    #net value growth
    df_dm_5d=dic_dm_5d[0]
    df_wm_5d=dic_wm_5d[0]
    df_mm_5d=dic_mm_5d[0]
    dic_momentum_5d ={"_daily":df_dm_5d,"_weekly":df_wm_5d,"_monthly":df_mm_5d}
    plot_grouped(dic_momentum_5d,benchmark,"Momentum",5)
    
    df_dm_21d=dic_dm_21d[0]
    df_wm_21d=dic_wm_21d[0]
    df_mm_21d=dic_mm_21d[0]
    dic_momentum_21d ={"_daily": df_dm_21d,"_weekly":df_wm_21d,"_monthly":df_mm_21d}
    plot_grouped(dic_momentum_21d,benchmark,"Momentum",21)

    
    df_dm_63d=dic_dm_63d[0]
    df_wm_63d=dic_wm_63d[0]
    df_mm_63d=dic_mm_63d[0]
    dic_momentum_63d ={"_daily": df_dm_63d,"_weekly":df_wm_63d,"_monthly":df_mm_63d}
    plot_grouped(dic_momentum_63d,benchmark,"Momentum",63)


    df_dm_126d=dic_dm_126d[0]
    df_wm_126d=dic_wm_126d[0]
    df_mm_126d=dic_mm_126d[0]
    dic_momentum_126d ={"_daily": df_dm_126d,"_weekly":df_wm_126d,"_monthly":df_mm_126d}
    plot_grouped(dic_momentum_126d,benchmark,"Momentum",126)

    
    #trades
    dic_mom_5d_trades={"daily":dic_dm_5d[1],"weekly":dic_wm_5d[1],"monthly":dic_mm_5d[1]}
    dic_mom_21d_trades={"daily":dic_dm_21d[1],"weekly":dic_wm_21d[1],"monthly":dic_mm_21d[1]}
    dic_mom_63d_trades={"daily":dic_dm_63d[1],"weekly":dic_wm_63d[1],"monthly":dic_mm_63d[1]}
    dic_mom_126d_trades={"daily":dic_dm_126d[1],"weekly":dic_wm_126d[1],"monthly":dic_mm_126d[1]}

    #average trade & plots
    dic_trade_mean_momentum={}
    for dic_trade,day in zip([dic_mom_5d_trades,dic_mom_21d_trades,dic_mom_63d_trades,dic_mom_126d_trades],l_rebalancing_day):
        dic_sub_count={}
        for group,df_count in dic_trade.items():
           lab="{} momentum with {}-day rebalancing frequency".format(group,str(day))
           perform_measure.trade_number_line(df_count,lab)
           dic_sub_count[group]=df_count.mean().values[0]
        dic_trade_mean_momentum[day]=pd.DataFrame([dic_sub_count]).T

    
    # performance analysis
        
    # cumulative average growth return yearly
    dic_momentum_CAGR={}
    dic_momentum_yearly={"daily":[ dic_dm_5d_yearly, dic_dm_21d_yearly, dic_dm_63d_yearly, dic_dm_126d_yearly],\
                        "weekly":[ dic_wm_5d_yearly, dic_wm_21d_yearly, dic_wm_63d_yearly, dic_wm_126d_yearly],\
                        "monthly":[ dic_mm_5d_yearly, dic_mm_21d_yearly, dic_mm_63d_yearly, dic_mm_126d_yearly]}
    for group,l_dic_year_mom in dic_momentum_yearly.items():
        dic_momentum_CAGR[group]=CAGR_yearly(l_dic_year_mom,l_rebalancing_day)
    
    # CAGR, max drawdown,annualized sharpe ratio,success ratio
    dic_mom_perform={}
    dic_momentum={"daily":[ df_dm_5d, df_dm_21d, df_dm_63d, df_dm_126d],\
                  "weekly":[ df_wm_5d, df_wm_21d, df_wm_63d, df_wm_126d],\
                  "monthly":[ df_mm_5d, df_mm_21d, df_mm_63d, df_mm_126d]}

    for group,l_df_mom in dic_momentum.items():  
        dic_mom_perform[group]=perform_summary(l_df_mom,l_rebalancing_day).T
    
    #1 year outperformance
    dic_1_year_outperform_mom={}
    for group,dic_CAGR_yearly in dic_momentum_CAGR.items():
        dic_outperform={}
        for re_day,df_CAGR in dic_CAGR_yearly.items():
            outperform=(df_CAGR-dic_benchmark_CAGR_yearly[re_day]).values
            dic_outperform[re_day]=len(outperform[outperform>0])/len(outperform)
        dic_1_year_outperform_mom[group]=dic_outperform

    
    '''
    Factor testing--50MA vs 150MA
    '''        
    # handle anomalies & create signals
    data_1=data[data["50_day_ma"]!=0]
    data_1["50vs150"]=[1 if i else 0 for i in data_1["50_day_ma"]>data_1["150_day_ma"]]
    data_1["50vs200"]=[1 if i else 0 for i in data_1["50_day_ma"]>data_1["200_day_ma"]]
        
    data_1=data_1.sort_values(by=["ticker","date"]).reset_index(drop=True)
    
    #check the data completeness
    check=data_1.groupby(by=["date"])[["ticker"]].count()
    #only select the dates when majority of data is available
    l_date_ma=np.sort(check[check["ticker"]>=1000].index)
    
    data_1=data_1.set_index(["date"])
    data_1=data_1.loc[l_date_ma]
    data_1=data_1.reset_index()
    
    
    #(1) 5 day
    dic_50vs150_5d_yearly=port_long(data_1,"50vs150",5,yearly_details=True)
    dic_50vs150_5d=port_long(data_1,"50vs150",5,yearly_details=False)
    #(2) 21 day
    dic_50vs150_21d_yearly=port_long(data_1,"50vs150",21,yearly_details=True)
    dic_50vs150_21d=port_long(data_1,"50vs150",21,yearly_details=False)

    #(3) 63 day
    dic_50vs150_63d_yearly=port_long(data_1,"50vs150",63,yearly_details=True)
    dic_50vs150_63d=port_long(data_1,"50vs150",63,yearly_details=False)
    #(4) 126 day
    dic_50vs150_126d_yearly=port_long(data_1,"50vs150",126,yearly_details=True)
    dic_50vs150_126d=port_long(data_1,"50vs150",126,yearly_details=False)
     
    '''
    Factor testing--50MA vs 200MA
    '''  
    #(1) 5 day
    dic_50vs200_5d_yearly=port_long(data_1,"50vs200",5,yearly_details=True)
    dic_50vs200_5d=port_long(data_1,"50vs200",5,yearly_details=False)
    #(2) 21 day
    dic_50vs200_21d_yearly=port_long(data_1,"50vs200",21,yearly_details=True)
    dic_50vs200_21d=port_long(data_1,"50vs200",21,yearly_details=False)

    #(3) 63 day
    dic_50vs200_63d_yearly=port_long(data_1,"50vs200",63,yearly_details=True)
    dic_50vs200_63d=port_long(data_1,"50vs200",63,yearly_details=False)
    #(4) 126 day
    dic_50vs200_126d_yearly=port_long(data_1,"50vs200",126,yearly_details=True)
    dic_50vs200_126d=port_long(data_1,"50vs200",126,yearly_details=False)    
    
    #net value growth
    df_ma150_5d=dic_50vs150_5d[0]
    df_ma200_5d=dic_50vs200_5d[0]
    dic_ma_5d ={"_50vs150":df_ma150_5d,"_50vs200":df_ma200_5d}
    plot_grouped(dic_ma_5d,benchmark,"Moving_average",5)
    
    df_ma150_21d=dic_50vs150_21d[0]
    df_ma200_21d=dic_50vs200_21d[0]
    dic_ma_21d ={"_50vs150": df_ma150_21d,"_50vs200":df_ma200_21d}
    plot_grouped(dic_ma_21d,benchmark,"Moving_average",21)

    
    df_ma150_63d=dic_50vs150_63d[0]
    df_ma200_63d=dic_50vs200_63d[0]
    dic_ma_63d ={"_50vs150": df_ma150_63d,"_50vs200":df_ma200_63d}
    plot_grouped(dic_ma_63d,benchmark,"Moving_average",63)


    df_ma150_126d=dic_50vs150_126d[0]
    df_ma200_126d=dic_50vs200_126d[0]
    dic_ma_126d ={"_50vs150": df_ma150_126d,"_50vs200":df_ma200_126d}
    plot_grouped(dic_ma_126d,benchmark,"Moving_average",126)

    
    #trades
    dic_ma_5d_trades={"50vs150":dic_50vs150_5d[1],"50vs200":dic_50vs200_5d[1]}
    dic_ma_21d_trades={"50vs150":dic_50vs150_21d[1],"50vs200":dic_50vs200_21d[1]}
    dic_ma_63d_trades={"50vs150":dic_50vs150_63d[1],"50vs200":dic_50vs200_63d[1]}
    dic_ma_126d_trades={"50vs150":dic_50vs150_126d[1],"50vs200":dic_50vs200_126d[1]}

    #average trade & plots
    dic_trade_mean_ma={}
    for dic_trade,day in zip([dic_ma_5d_trades,dic_ma_21d_trades,dic_ma_63d_trades,dic_ma_126d_trades],l_rebalancing_day):
        dic_sub_count={}
        for group,df_count in dic_trade.items():
           lab="{} moving average with {}-day rebalancing frequency".format(group,str(day))
           perform_measure.trade_number_line(df_count,lab)
           dic_sub_count[group]=df_count.mean().values[0]
        dic_trade_mean_ma[day]=pd.DataFrame([dic_sub_count]).T

    
    # performance analysis
     
    # cumulative average growth return yearly
    dic_ma_CAGR={}
    dic_ma_yearly={"50vs150":[ dic_50vs150_5d_yearly, dic_50vs150_21d_yearly, dic_50vs150_63d_yearly, dic_50vs150_126d_yearly],\
                   "50vs200":[ dic_50vs200_5d_yearly, dic_50vs200_21d_yearly, dic_50vs200_63d_yearly, dic_50vs200_126d_yearly]}
    for group,l_dic_year_ma in dic_ma_yearly.items():
        dic_ma_CAGR[group]=CAGR_yearly(l_dic_year_ma,l_rebalancing_day)
    
    # CAGR, max drawdown,annualized sharpe ratio,success ratio
    dic_ma_perform={}
    dic_ma={"50vs150":[df_ma150_5d, df_ma150_21d, df_ma150_63d, df_ma150_126d],\
                  "50vs200":[df_ma200_5d, df_ma200_21d, df_ma200_63d, df_ma200_126d]}

    for group,l_df_ma in dic_ma.items():  
        dic_ma_perform[group]=perform_summary(l_df_ma,l_rebalancing_day,year_period=(data_1["date"].max()-data_1["date"].min()).days/365).T
    
    #1 year outperformance
    dic_1_year_outperform_ma={}
    for group,dic_CAGR_yearly in dic_ma_CAGR.items():
        dic_outperform={}
        for re_day,df_CAGR in dic_CAGR_yearly.items():
            outperform=(df_CAGR-dic_benchmark_CAGR_yearly[re_day]).dropna().values
            dic_outperform[re_day]=len(outperform[outperform>0])/len(outperform)
        dic_1_year_outperform_ma[group]=dic_outperform


    '''
    Factor testing--weekly obos
    '''  
    
    # Group (value small to large) for return analysis
    
    data_2= data.sort_values(by=["weekly_obos"]).reset_index(drop=True)
    
    # remove the extreme value and normalized
    def remove_extreme(x,median1,median2):
        upper= median1+ 5*median2
        down=  median1- 5*median2
        
        if x > upper:
            x=upper
        if x< down:
            x=down
        return x
    
    median1=data_2["weekly_obos"].median()
    median2=abs((data_2["weekly_obos"]-median1)).median()

    # remove extre
    data_2["norm_obos"]=[remove_extreme(x, median1,median2) for x in data_2["weekly_obos"].values]
    
    # normalized
    data_2["norm_obos"]=(data_2["norm_obos"]-data_2["norm_obos"].mean())/data_2["norm_obos"].std()
    
    #Spearman correlation (Rank IC) overall & monthly change &  average ( mean & std)    
    df_all_spear,df_month_spear=spearman_correlation("norm_obos",["5_day_forward_return","21_day_forward_return",\
                                                                  "63_day_forward_return","126_day_forward_return"],data_2)
            
    #Monthly 
    #summary
    df_month_spear_mean=df_month_spear.groupby(["Return"]).mean()
    df_ir=df_month_spear.groupby(["Return"])["Spearman Correlation"].mean()/df_month_spear.groupby(["Return"])["Spearman Correlation"].std()
    
    #plot monthly change
    plt.figure(figsize=(20,15))    
    plt.style.use('seaborn')
    sns.set(font_scale=2)
    df_spear_plot=df_month_spear
    df_spear_plot["Month"]=pd.to_datetime(df_spear_plot["Month"].astype("str"),format="%Y-%m")
    ax = sns.lineplot(x="Month",y="Spearman Correlation",data=df_spear_plot,hue="Return",palette = 'Accent')
    
    plt.xlabel("year",fontsize=20)
    plt.ylabel("IC",fontsize=20)
    plt.title("IC evolution of weekly obos across the years",fontsize=20)
    plt.legend(loc="upper right")


    # Linear regression (IC) overall & monthly change & average (R square, return rate, t value, change of sign)
    df_all_linear,df_month_linear=linear_correlation("norm_obos",["5_day_forward_return","21_day_forward_return",\
                                                                  "63_day_forward_return","126_day_forward_return"],data_2)
    

        
    #Monthly 
    #summary
    df_month_linear_mean=df_month_linear.groupby(["Return"]).mean()
    df_ir_linear=df_month_linear.groupby(["Return"])["Linear Correlation"].mean()/df_month_linear.groupby(["Return"])["Linear Correlation"].std()

    
    #plot monthly change
  
    plt.figure(figsize=(20,15))    
    plt.style.use('seaborn')
    sns.set(font_scale=2)
    df_month_linear["Month"]=pd.to_datetime(df_month_linear["Month"].astype("str"),format="%Y-%m")
    ax = sns.lineplot(x="Month",y="Linear Correlation",data=df_month_linear,hue="Return",palette = 'Accent')
    
    plt.xlabel("year",fontsize=20)
    plt.ylabel("IC",fontsize=20)
    plt.title("IC evolution of weekly obos across the years",fontsize=20)
    plt.legend(loc="upper right")
    
    #Relationship with forward return (average by group)
    data_2_adj_1=pd.DataFrame()
    l_date=np.sort(data_2["date"].unique())
    l_date=pd.to_datetime(l_date, format="%Y-%m-%d")
    data_2.set_index(["date"],inplace=True)
    for date in l_date:
        df_sub=data_2.loc[date]
        df_sub=df_sub.sort_values(by=["norm_obos"])
        group_cut=[]
        for i in range(1,10):
            group_cut.append( int(i*((len(df_sub)+1)/10)))
        #create group 
        df_sub["group_obos"]=np.nan
        for i in range(8):
            if i ==0:
                df_sub.iloc[:group_cut[i],-1]=1
                df_sub.iloc[group_cut[i]:group_cut[i+1],-1]=2
    
            else:
                df_sub.iloc[group_cut[i]:group_cut[i+1],-1]=i+2
        df_sub.iloc[group_cut[-1]:,-1]=10
        data_2_adj_1=data_2_adj_1.append(df_sub)
        print(date)
    # plot bar chart
    df_obos_all=data_2_adj_1[["group_obos","5_day_forward_return","21_day_forward_return",\
                                                                  "63_day_forward_return","126_day_forward_return"]].groupby(["group_obos"]).mean()
    #plot bar chart
    for rt in df_obos_all.columns:
        plt.figure(figsize=(20,15))    
        plt.style.use('seaborn')
        sns.set(font_scale=3)
        df_sub=df_obos_all[rt]
        df_sub=df_sub.reset_index()
        plt.bar(df_sub["group_obos"],df_sub[rt])
        plt.ylabel(rt,fontsize=50)
        plt.title("Forward return of weekly obos",fontsize=50)

    #liner chart change across the years
    df_obos_m=data_2_adj_1[["group_obos","month",\
         "5_day_forward_return","21_day_forward_return",\
          "63_day_forward_return","126_day_forward_return"]].groupby(["group_obos","month"]).mean()

    
    for rt in df_obos_m.columns:
    
        plt.figure(figsize=(20,15))    
        plt.style.use('seaborn')
        sns.set(font_scale=3)
        df_sub=df_obos_m[rt]
        df_sub=df_sub.reset_index()
        df_sub["month"]=pd.to_datetime(df_sub["month"].astype("str"),format="%Y-%m")
        df_sub["group_obos"]="group "+ df_sub["group_obos"].astype("int").astype("str")
        ax = sns.lineplot(x="month",y=rt,data=df_sub,hue="group_obos",palette = 'RdBu_r')
        
        plt.xlabel("year",fontsize=50)
        plt.ylabel(rt,fontsize=50)
        plt.title("Forward return evolution of weekly obos across the years",fontsize=50)
        plt.legend(loc="upper right")

    data_2_adj_1.reset_index(inplace=True)
    #(1) 5 day
    dic_obos_5d_yearly=port_long_group(data_2_adj_1,"group_obos",5,yearly_details=True)
    rst_obos_5d=port_long_group(data_2_adj_1,"group_obos",5,yearly_details=False)
    #(2) 21 day
    dic_obos_21d_yearly=port_long_group(data_2_adj_1,"group_obos",21,yearly_details=True)
    rst_obos_21d=port_long_group(data_2_adj_1,"group_obos",21,yearly_details=False)

    #(3) 63 day
    dic_obos_63d_yearly=port_long_group(data_2_adj_1,"group_obos",63,yearly_details=True)
    rst_obos_63d=port_long_group(data_2_adj_1,"group_obos",63,yearly_details=False)
    #(4) 126 day
    dic_obos_126d_yearly=port_long_group(data_2_adj_1,"group_obos",126,yearly_details=True)
    rst_obos_126d=port_long_group(data_2_adj_1,"group_obos",126,yearly_details=False)    
    
    #net value growth
    dic_obos_5d=rst_obos_5d[0]
    plot_grouped(dic_obos_5d,benchmark,"Group",5)
    
    dic_obos_21d=rst_obos_21d[0]
    plot_grouped(dic_obos_21d,benchmark,"Group",21)
    
    dic_obos_63d=rst_obos_63d[0]
    plot_grouped(dic_obos_63d,benchmark,"Group",63)

    dic_obos_126d=rst_obos_126d[0]
    plot_grouped(dic_obos_126d,benchmark,"Group",126)

    # cumulative average growth return yearly
    dic_obos_CAGR={}
    for group in range(1,11):
        l_dic_year_obos=[]
        for rst_obos_dic in [dic_obos_5d_yearly,dic_obos_21d_yearly,dic_obos_63d_yearly,dic_obos_126d_yearly]:
            dic_sub={}
            for year, dic in rst_obos_dic.items():
                dic_sub[year]=dic[group]
            l_dic_year_obos.append(dic_sub)
            dic_obos_CAGR[group]=CAGR_yearly(l_dic_year_obos,l_rebalancing_day)
    
    # CAGR, max drawdown,annualized sharpe ratio,success ratio
    dic_obos_perform={}
    for group in range(1,11):
        l_df_obos=[]
        for df in [dic_obos_5d,dic_obos_21d,dic_obos_63d, dic_obos_126d]:
            l_df_obos.append(df[group])
        
        dic_obos_perform[group]=perform_summary(l_df_obos,l_rebalancing_day).T
    
    # trades analysis
    #(1)5_days_rets
    dic_obos_5days_trades=rst_obos_5d[1]
    #(2)21_days_rets
    dic_obos_21days_trades=rst_obos_21d[1]
    #(3)63_days_rets
    dic_obos_63days_trades=rst_obos_63d[1]
    #(4)126_days_rets
    dic_obos_126days_trades=rst_obos_126d[1]
    
    #average trade & plots
    dic_trade_mean_obos={}
    for dic_trade,day in zip([dic_obos_5days_trades,dic_obos_21days_trades,dic_obos_63days_trades,dic_obos_126days_trades],l_rebalancing_day):
        dic_sub_count={}
        for group,df_count in dic_trade.items():
           lab="Group {} with {}-day rebalancing frequency".format(int(group),str(day))
           perform_measure.trade_number_line(df_count,lab)
           dic_sub_count[group]=df_count.mean().values[0]
        dic_trade_mean_obos[day]=pd.DataFrame([dic_sub_count]).T
        
    #1 year outperformance
    dic_1_year_outperform_obos={}
    for group,dic_CAGR_yearly in dic_obos_CAGR.items():
        dic_outperform={}
        for re_day,df_CAGR in dic_CAGR_yearly.items():
            outperform=(df_CAGR-dic_benchmark_CAGR_yearly[re_day]).values
            dic_outperform[re_day]=len(outperform[outperform>0])/len(outperform)
        dic_1_year_outperform_obos[group]=dic_outperform

