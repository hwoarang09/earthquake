#!/usr/bin/env python
# coding: utf-8

# In[3]:


import flask
import dash
import vizro
import vizro.plotly.express as px
from vizro import Vizro
import vizro.models as vm
import pandas as pd
import numpy as np


path = r"C:\Users\hwoar\OneDrive\바탕 화면\sw_train\튜터알바\메가스터디 과외자료\현우\\"
koreafile = 'data_preprocessing.csv'
worldfile = 'df_worlds.csv'

df_world = pd.read_csv(path+worldfile,
                       index_col = 'Date',  parse_dates=['Time', 'Date'])

df_korea = pd.read_csv(path+koreafile, 
                 index_col = '발생시각',  parse_dates=['Time', '발생시각'])

df_korea = df_korea.rename(columns={'규모' : 'Magnitude', 'latitude' : 'Latitude', 'longitude' : 'Longitude'})



df_korea['Year5'] = pd.cut(df_korea.index.year,
                          bins=range(df_korea.index.year.min(), df_korea.index.year.max() + 6, 5),
                          right=False,
                          labels=[f"{i}-{i+4}" for i in range(df_korea.index.year.min(), df_korea.index.year.max(), 5)])
df_world['Year5'] = pd.cut(df_world.index.year,
                          bins=range(df_world.index.year.min(), df_world.index.year.max() + 6, 5),
                          right=False,
                          labels=[f"{i}-{i+4}" for i in range(df_world.index.year.min(), df_world.index.year.max(), 5)])


def get_distplot(df, w=800,h=600):
    fig = px.histogram(df, x='Magnitude',  marginal='violin',
                   opacity=0.75)

    return fig


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

df_korea['Month'] = pd.Categorical(df_korea['Month'], month_order)
df_world['Month'] = pd.Categorical(df_world['Month'], month_order)

df_korea_sort_month = df_korea.sort_values('Month')
df_world_sort_month = df_world.sort_values('Month')


def get_month_chart_simple(df,w=800,h=600):
    fig = px.histogram(df, x="Month")
    fig.update_layout(width=w, height=h)
    return fig


def get_month_chart_stack(df,w=800,h=600):
    fig = px.histogram(df, x="Month", color="Year",
                       category_orders={"Month": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']},
                       barmode='stack',
                      )

    fig.update_layout(width=w, height=h)
    return fig


def get_month_chart_stack5(df, w=800,h=600):

    fig = px.histogram(df, x="Month", color="Year5",
                       category_orders={"Month": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']},
                       barmode='stack',
                      )

    fig.update_layout(width=w, height=h)


    return fig 


def get_year_chart(df,w=800,h=600):
    yearly_groups = df.groupby(df.index.year)
    data = yearly_groups[['Magnitude']].count()
    fig = px.scatter(data_frame=data)
    fig.update_layout(width=w, height=h)
    return fig


dk_mag_range = [df_korea['Magnitude'].describe()['25%'],df_korea['Magnitude'].describe()['50%']
                ,df_korea['Magnitude'].describe()['75%'],df_korea['Magnitude'].describe()['max']]
dw_mag_range = [df_world['Magnitude'].describe()['25%'],df_world['Magnitude'].describe()['50%']
                ,df_world['Magnitude'].describe()['75%'],df_world['Magnitude'].describe()['max']]

k_magnitude_order=[str(r)+'미만' for r in dk_mag_range]
w_magnitude_order=[str(r)+'미만' for r in dw_mag_range]

df_world['Magnitude Range 5 Year'] = pd.cut(df_world['Magnitude'], bins=[-float('inf')] + dw_mag_range, right=False, 
                               labels=w_magnitude_order)
df_korea['Magnitude Range 5 Year'] = pd.cut(df_korea['Magnitude'], bins=[-float('inf')] + dk_mag_range, right=False, 
                               labels=k_magnitude_order)


def get_mag_range(df, magnitude_order):


    fig = px.histogram(df, x='Year', color="Magnitude Range 5 Year",
                       category_orders={'Magnitude Range 5 Year': magnitude_order},
                      )
    return fig


# In[4]:


Vizro._reset()

dist_k = get_distplot(df_korea)
month_k = get_month_chart_simple(df_korea_sort_month)
year_k = get_year_chart(df_korea)
year_5_k = get_mag_range(df_korea, k_magnitude_order)
month_stack_k = get_month_chart_stack(df_korea)
month_stack5_k = get_month_chart_stack5(df_korea)

dist_w = get_distplot(df_world)
month_w = get_month_chart_simple(df_world_sort_month)
year_w = get_year_chart(df_world)
year_5_w = get_mag_range(df_world, w_magnitude_order)
month_stack_w = get_month_chart_stack(df_world)
month_stack5_w = get_month_chart_stack5(df_world)





page_Korea_Magnitude = vm.Page(
    id = "Korea Magnitude",
    title="Magnitude",
    components=[
        vm.Graph(id="dist_korea", figure=dist_k),
    ]
)

page_Korea_Month = vm.Page(
    id="Korea Month",
    title="Month",
    components=[
        vm.Graph(id="month_korea", figure=month_k),
        vm.Graph(id="month_stack_korea", figure=month_stack_k),
        vm.Graph(id="month_stack5_korea", figure=month_stack5_k),        
    ]
)

page_Korea_Year = vm.Page(
    id ="Korea Year",
    title="Year",
    layout=vm.Layout(grid=[[0, 0, 2],
                           [1, 1, 2],
                            [1, 1, 2]]),     
    components=[
        vm.Graph(id="year_korea", figure=year_k),
        vm.Graph(id="year_5_korea", figure=year_5_k),
        vm.Card(
            text="""
                ### Card Title
                Commodi repudiandae consequuntur voluptatum.
            """,
        ),
    ],
    controls=[
        vm.Parameter(
                      targets=["year_korea.trendline"],
                     selector=vm.Dropdown(
                      options=['lowess', 'expanding', 'ols'],
                multi=False,
                value="ols",
                     )),
    
    ]    
)

page_World_Magnitude = vm.Page(
    id="World Magnitude",
    title="Magnitude",
    components=[
        vm.Graph(id="dist_world", figure=dist_w),
    ]

)

page_World_Month = vm.Page(
    id="World Month",
    title="Month",
    layout=vm.Layout(grid=[[0, 0, 2],
                           [1, 1, 2],
                            [1, 1, 2]]),    
    components=[
        vm.Graph(id="month_world", figure=month_w),
        vm.Graph(id="month_stack_world", figure=month_stack_w),
        vm.Graph(id="month_stack5_world", figure=month_stack5_w),
    ],
 
)

page_World_Year = vm.Page(
    id="World Year",
    title="Year",
    layout=vm.Layout(grid=[[0, 0, 2],
                           [1, 1, 2],
                            [1, 1, 2]]),       
    components=[
        vm.Graph(id="year_world", figure=year_w),
        vm.Graph(id="year_5_world", figure=year_5_w),
        vm.Card(
            text="""
                # 데이터 분석
                
                ## 연간 지진 발생 횟수\n\n\n
                
                ### 지진 발생 횟수 자체가 늘어나고 있는 것인가?
                아님. 지진 관측 능력이 향상되어 많이 관측 되는 것.
                ### 지진 발생 횟수가 계단식이 아닌 점진적 증가를 나타내는 이유?
                
                지진관측 능력은 관측 기기의 성능보다(계단식 증가)
                지진 관측소의 개수에 영향을 많이 받기 때문(점진적 증가)
                
                # 
            """,
        ),        
    ],
    controls=[
        vm.Parameter(
                      targets=["year_world.trendline"],
                     selector=vm.Dropdown(
                      options=['lowess', 'expanding', 'ols'],
                multi=False,
                value='ols',
                     ))
    ]
)

dashboard = vm.Dashboard(pages=[page_Korea_Magnitude, page_Korea_Month, page_Korea_Year,
                                page_World_Magnitude, page_World_Month, page_World_Year],
                         navigation=vm.Navigation(pages={"Korea" : ["Korea Magnitude","Korea Month","Korea Year"], 
                                                         "World" : ["World Magnitude", "World Month", "World Year"]}))

Vizro().build(dashboard).run()


# In[ ]:




