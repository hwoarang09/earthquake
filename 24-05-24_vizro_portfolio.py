#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import flask
import dash
import vizro
import vizro.plotly.express as px

from vizro import Vizro
from vizro.models.types import capture
import vizro.models as vm
import pandas as pd
import numpy as np

from datetime import datetime, date


# # Data Load

# In[2]:


path = r"C:\Users\hwoar\OneDrive\바탕 화면\sw_train\튜터알바\메가스터디 과외자료\현우\git\\"
koreafile = 'data_preprocessing.csv'
worldfile = 'df_worlds.csv'


# In[3]:


pd.__version__


# In[4]:


#외국지진데이터
df_world = pd.read_csv(path+worldfile,
                       index_col = 'Date',  parse_dates=['Time', 'Date'])

df_world.index = pd.to_datetime(df_world.index, format='ISO8601')
df_world


# In[5]:


#한국지진데이터
df_korea = pd.read_csv(path+koreafile, 
                 index_col = '발생시각',  parse_dates=['Time', '발생시각'])

df_korea = df_korea.rename(columns={'규모' : 'Magnitude', 'latitude' : 'Latitude', 'longitude' : 'Longitude'})
df_korea


# In[6]:


df_korea['Year5'] = pd.cut(df_korea.index.year,
                          bins=range(df_korea.index.year.min(), df_korea.index.year.max() + 6, 5),
                          right=False,
                          labels=[f"{i}-{i+4}" for i in range(df_korea.index.year.min(), df_korea.index.year.max(), 5)])
df_world['Year5'] = pd.cut(df_world.index.year,
                          bins=range(df_world.index.year.min(), df_world.index.year.max() + 6, 5),
                          right=False,
                          labels=[f"{i}-{i+4}" for i in range(df_world.index.year.min(), df_world.index.year.max(), 5)])


df_world['Magnitude'] = df_world['Magnitude'].apply(lambda x: round(x,1))
df_korea = df_korea.rename(columns ={'도' : 'region'})

df_world.region = df_world.region.fillna('error')


# In[7]:


df_world.index


# In[8]:


df_korea.index.year


# In[9]:


print(df_korea.info())
print(df_world.info())


# In[10]:


print(df_korea.index)
print(df_world.index)


# In[11]:


df_korea.describe()


# In[12]:


df_world.describe()


# # 시각화 차트 정리

# ## distplot
# 

# In[13]:


import plotly.io as pio
pio.templates


# In[14]:


pio.templates.default = "plotly_white"


# In[15]:


mag_min = df_korea['Magnitude'].min()
mag_min


# In[16]:


fig = px.histogram(df_korea, x='Magnitude',  marginal='violin',
                   opacity=0.75, title='Distplot with Histogram and KDE',
                  )

count_max = df_korea.groupby('Magnitude')['Time'].count().max()*1.1
mag_min = df_korea['Magnitude'].min()
mag_max = mag_min+0.5

lower_bound = mag_min
upper_bound = mag_max

# 구간 내 값들의 비율 계산
within_range = df_korea[(df_korea['Magnitude'] >= lower_bound) & (df_korea['Magnitude'] <= upper_bound)]
percentage = round((within_range.shape[0] / df_korea.shape[0]) * 100,1)


anno_text = f'Values ranging from {lower_bound} to {upper_bound}<br> account for {percentage}% of the total data'

fig.add_vrect(x0=mag_min, x1=mag_max,
            
              fillcolor="yellow", opacity=0.25, line_width=0)


fig.add_annotation(
        x=mag_max,
        y=count_max,
        xref="x",
        yref="y",
        text=anno_text,
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=120,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
        )

fig.update_xaxes(showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across")
fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=2)
fig.update_layout(spikedistance=1000, hoverdistance=100)


fig.show()


# In[17]:


@capture("graph")
def get_distplot(data_frame, w=800,h=600):
    count_max = data_frame.groupby('Magnitude')['Time'].count().max()*1.1
    mag_min = data_frame['Magnitude'].min()
    mag_max = mag_min+0.1

    lower_bound = mag_min
    upper_bound = mag_max

    # 구간 내 값들의 비율 계산
    within_range = data_frame[(data_frame['Magnitude'] >= lower_bound) & (data_frame['Magnitude'] <= upper_bound)]
    percentage = round((within_range.shape[0] / data_frame.shape[0]) * 100,1)


    anno_text = f'Values ranging from {lower_bound} to {upper_bound}<br> account for {percentage}% of the total data'

    fig = px.histogram(data_frame, x='Magnitude',  marginal='violin',
                       opacity=0.75, title='Distplot with Histogram and KDE',
                      )

    fig.add_vrect(x0=mag_min, x1=mag_min+0.5,
                  #annotation_text=anno_text, annotation_position="top right",
                  fillcolor="yellow", opacity=0.25, line_width=0)

    fig.add_annotation(
            x=mag_min+0.5,
            y=count_max,
            xref="x",
            yref="y",
            text=anno_text,
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
                ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=120,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
            )    
    
    fig.update_xaxes(showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=2)
    fig.update_layout(spikedistance=1000, hoverdistance=100)    
    return fig


# ## 월별 통계 - simple

# In[18]:


#기존에는 이렇게 만들엇는데,
# 이렇게 하면 vizro에서 순서가 원상복구됨.
fig = px.histogram(df_korea, x="Month")
fig.update_xaxes(categoryorder='array', categoryarray= ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
fig.update_layout(width=800, height=600)
fig.show()


# In[19]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

df_korea['Month'] = pd.Categorical(df_korea['Month'], month_order)
df_world['Month'] = pd.Categorical(df_world['Month'], month_order)

df_korea_sort_month = df_korea.sort_values('Month')
df_world_sort_month = df_world.sort_values('Month')


# In[20]:


fig = px.histogram(df_korea_sort_month, x="Month")
fig.update_layout(width=800, height=600)
fig.show()


# In[21]:


def get_month_chart_simple(df,w=800,h=600):
    fig = px.histogram(df, x="Month")
    fig.update_layout(width=w, height=h)
    return fig


# ## 월별 통계-stack

# In[22]:


fig = px.histogram(df_world, x="Month", color="Year",
                   category_orders={"Month": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']},
                   barmode='stack',
                  )

fig.update_layout(width=800, height=600)
fig.show()


# In[23]:


def get_month_chart_stack(df,w=800,h=600):
    fig = px.histogram(df, x="Month", color="Year",
                       category_orders={"Month": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']},
                       barmode='stack',
                      )

    fig.update_layout(width=w, height=h)
    return fig


# ## 월별 통계 stack 5년

# In[24]:


fig = px.histogram(df_korea, x="Month", color="Year5",
                   category_orders={"Month": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']},
                   barmode='stack',
                  )

fig.update_layout(width=800, height=600)


fig.show()


# In[25]:


def get_month_chart_stack5(df, w=800,h=600):

    fig = px.histogram(df, x="Month", color="Year5",
                       category_orders={"Month": ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']},
                       barmode='stack',
                      )

    fig.update_layout(width=w, height=h)


    return fig  


# ## 연도별 차트 - Trendline

# In[26]:


yearly_groups = df_world.groupby(df_world.index.year)
data = yearly_groups[['Magnitude']].count()
fig = px.scatter(data_frame=data, trendline='ols')
fig.update_layout(width=800, height=600)
fig.show()


# In[27]:


yearly_groups = df_korea.groupby(df_korea.index.year)
data = yearly_groups[['Magnitude']].count()
fig = px.scatter(data_frame=data,trendline=None)
fig.update_layout(width=800, height=600)
fig.show()


# In[28]:


def get_year_chart(df,w=800,h=600):
    yearly_groups = df.groupby(df.index.year)
    data = yearly_groups[['Magnitude']].count()
    fig = px.scatter(data_frame=data)
    fig.update_layout(width=w, height=h)
    return fig


# ## 연도별차트 - Range

# In[29]:


dk_mag_range = [df_korea['Magnitude'].describe()['25%'],df_korea['Magnitude'].describe()['50%']
                ,df_korea['Magnitude'].describe()['75%'],df_korea['Magnitude'].describe()['max']]
dw_mag_range = [df_world['Magnitude'].describe()['25%'],df_world['Magnitude'].describe()['50%']
                ,df_world['Magnitude'].describe()['75%'],df_world['Magnitude'].describe()['max']]


# In[30]:


k_magnitude_order=[str(r)+'미만' for r in dk_mag_range]

df_korea['Magnitude Range 5 Year'] = pd.cut(df_korea['Magnitude'], bins=[-float('inf')] + dk_mag_range, right=False, 
                               labels=k_magnitude_order)
fig = px.histogram(df_korea, x='Year', color="Magnitude Range 5 Year",
                                      category_orders={'Magnitude Range 5 Year': k_magnitude_order},
                   #nbins = len(df['Year'].value_counts().index)   #이거 빼면 2년씩 묶임
                  )
fig.show()


# In[31]:


df_world[df_world['Year'].isin([2015,2016])]


# In[32]:


w_magnitude_order=[str(r)+'미만' for r in dw_mag_range]

df_world['Magnitude Range 5 Year'] = pd.cut(df_world['Magnitude'], bins=[-float('inf')] + dw_mag_range, right=False, 
                               labels=w_magnitude_order)

fig = px.histogram(df_world, x='Year', color="Magnitude Range 5 Year",
                   category_orders={'Magnitude Range 5 Year': w_magnitude_order},
                  )
fig.show()


# In[33]:


def get_mag_range(df, magnitude_order):


    fig = px.histogram(df, x='Year', color="Magnitude Range 5 Year",
                       category_orders={'Magnitude Range 5 Year': magnitude_order},
                      )
    return fig


# # vizro 시작

# In[34]:


Vizro._reset()


dist_k = get_distplot(df_korea)
dist_k_filter = get_distplot(df_korea)
month_k = get_month_chart_simple(df_korea_sort_month)
year_k = get_year_chart(df_korea)
year_5_k = get_mag_range(df_korea, k_magnitude_order)
month_stack_k = get_month_chart_stack(df_korea)
month_stack5_k = get_month_chart_stack5(df_korea)



dist_w = get_distplot(df_world)
dist_w_filter = get_distplot(df_world)
japan_unique = list(np.unique(np.array([a for a in df_world.region.values if 'Japan' in a ])))

month_w = get_month_chart_simple(df_world_sort_month)
year_w = get_year_chart(df_world)
year_5_w = get_mag_range(df_world, w_magnitude_order)
month_stack_w = get_month_chart_stack(df_world)
month_stack5_w = get_month_chart_stack5(df_world)





page_Korea_Magnitude = vm.Page(
    id = "Korea Magnitude",
    title="Magnitude2",
    layout=vm.Layout(grid=[[0, 1],
                           [0, 1],
                            [2, 3]]),      
    components=[
        vm.Graph(id="dist_korea", figure=dist_k),
        vm.Graph(id="dist_korea_filter", figure=dist_k_filter),
        vm.Card(
            text="""
                # 한국 지진 규모 분석&nbsp;
                * 1978년부터 현재까지 남북한 전체의 지진발생을 기록한 데이터입니다.
                * 대부분의 지진은 규모 3.5미만입니다.&nbsp;
                * 규모 2.0~2.1 구간에서 전체 지진의 18.7%가 발생합니다.
                * 가장 규모가 큰 지진도 규모6이 되지 않습니다.
            """,
        ),            
        vm.Card(
            text="""
                ### 지역,날짜로 필터링한 한국 지진 규모 분석&nbsp;
                * 기본값은 2016년1월1일 - 2018년12월31일 까지 경남,경북에서 발생한 지진을 필터링 한 것입니다.
                * 이 좁은 지역, 짧은 기간 동안 한반도 전체 지진의 1/4가 발생했습니다. 
                * 다른 지역도 살펴보고 싶다면 왼쪽 filter를 조정해주세요
            """,
        ),                    
    ],
    controls=[
        vm.Filter(column="Time", targets = ['dist_korea_filter'],selector=vm.DatePicker(value=[date(2016,1,1), date(2018,12,31)])),
        vm.Filter(column="region", targets = ['dist_korea_filter'], selector=vm.Dropdown(value=['경북','경남'])),
    
    ], 
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
    layout=vm.Layout(grid=[[0, 1],
                           [0, 1],
                            [2, 3]]),      
    components=[
        vm.Graph(id="dist_world", figure=dist_w),
        vm.Graph(id="dist_world_filter", figure=dist_w_filter),
        vm.Card(
            id = 'typewriter',
            text="""  
                # 외국 지진 규모 분석&nbsp;
                * 1965년부터 현재까지 세계각지의 지진발생을 기록한 데이터입니다.
                * 5.5미만의 여진은 기록되지 않은 데이터입니다.
                * 대부분의 지진은 규모 6미만입니다.&nbsp;
                * 규모 5.5~5.6 구간에서 전체 지진의 37.1%가 발생합니다.
            """,
        ),            
        vm.Card(
            text="""
                ### 지역,날짜로 필터링한 세계 지진 규모 분석&nbsp;
                * 기본값은 2016년1월1일 - 2018년12월31일 까지 일본에서 발생한 지진을 필터링 한 것입니다.
                
                * 다른 지역도 살펴보고 싶다면 왼쪽 filter를 조정해주세요
            """,
        ),                    
    ],
    controls=[
        vm.Filter(column="Time", targets = ['dist_world_filter'],selector=vm.DatePicker(value=[date(2016,1,1), date(2018,12,31)])),
        vm.Filter(column="region", targets = ['dist_world_filter'], selector=vm.Dropdown(value=japan_unique)),
    
    ],     
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
                                                         "World" : ["World Magnitude", "World Month", "World Year"]}),
                        theme='vizro_light')

Vizro().build(dashboard).run()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




