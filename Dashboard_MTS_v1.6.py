#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# In[2]:


st.set_page_config(layout="wide")


# # Reading Processed Data

# In[208]:


@st.cache
def read_df():
    # df = pd.read_pickle(r"C:\Users\waqar\OneDrive\Python\Medgulf\Quotations\MTS\MTS_ProdDataUY21&22_atJune22_Processed.pkl")
    df = pd.read_pickle(
        r"\\mgfs1\R&D Health and Life\R&D - TECHNICAL SECTION\GENERAL DIVISION\MOTOR TARIFF REVIEW\Python Projects\Motor Dashboard\MTS\Code\MTS_ProdDataUY21&22_upto10102022_Processed_v2.pkl")
    # df['day'] = df.groupby(['Year','Month','Week'])['Day'].transform(lambda x: x.min()) # day = minimum day in that week representing the start of the week
    df['wc_day'] = df['Issue Date'].dt.to_period('W').dt.start_time.dt.day  # finding the start day of each week
    df['wc_month'] = df['Issue Date'].dt.to_period('W').dt.start_time.dt.month
    df['wc_year'] = df['Issue Date'].dt.to_period('W').dt.start_time.dt.year

    # df['WC_Date'] = pd.to_datetime(df[['wc_day','wc_month','wc_year']]) # creating a new date to be used in charts as "Week Commencing Date"
    df['WC_Date'] = df.apply(lambda x: str(x['wc_day']) + '-' + str(x['wc_month']) + '-' + str(x['wc_year']), axis=1)
    df['WC_Date'] = pd.to_datetime(df['WC_Date'], infer_datetime_format=True)
    df['WC_Date'] = df['WC_Date'].astype('category')

    # df['WC_Date'] = pd.to_datetime(df[['day','Month','Year']]) # creating a new date to be used in charts as "Week Commencing Date"
    df['Price'] = df['Price'].dt.strftime('%d-%m-%Y')
    df = df[df['Issue Date'] > '2021-07-01']
    # df['WC_Date'] = df['WC_Date'].astype('category')

    return df


df = read_df()
prices = pd.Series(pd.to_datetime(df['Price'], format='%d-%m-%Y').sort_values(ascending=False).unique())
prices = [price.strftime('%d-%m-%Y') for price in prices]
sort_order = {'Up to 21': 0, '22 – 25': 1, '26 – 29': 2, '30+': 3}

# In[4]:


# df.head(2)


# In[86]:


df.info()

# # Analysis and Plotting

# In[4]:


colors = ['indigo', 'lightseagreen', 'lightcoral', 'lawngreen', 'lightyellow', 'lightblue', 'indianred', 'honeydew',
          'hotpink', 'ivory', 'gainsboro', 'greenyellow', 'firebrick', 'gold', 'forestgreen', 'dodgerblue', 'dimgrey',
          'fuchsia', 'lightblue', 'purple', 'sky blue', 'magenta']


# #### Age Plotting

# In[212]:


# pd.Series(pd.to_datetime(df['Price'], infer_datetime_format=True).sort_values(ascending=False).unique())#.strftime('%d-%m-%Y')


# In[222]:


# Sales Proportion Chart by Age
@st.cache
def agePlotSalesProp_select(df, brands_select, regions_select, age_select, prices_strategies):
    df_ = df[(df['Brand English'].isin(brands_select)) & (df['Insured Region'].isin(regions_select)) &
             (df['Age'].isin(age_select)) & (df['Price'].isin(prices_strategies))]

    plt = df_.groupby(['Age', 'Price']).count().reset_index()
    plt['Prop'] = plt['Policy No'] / plt.groupby('Price')['Policy No'].transform(
        'sum') * 100  # Calculating proportion of policies by Age

    # Plotting
    color_map = {}
    for idx, i in enumerate(plt['Price'].value_counts().index.values):
        color_map[i] = colors[idx]

    color_array = plt['Price'].map(color_map)

    plots_age = make_subplots(specs=[[{"secondary_y": True}]])

    plt['Price'] = pd.to_datetime(plt['Price'], format='%d-%m-%Y')
    plt = plt.sort_values(by=['Price', 'Age'], ascending=False)

    # Plotting Exposures at ages
    # plot_names = ['Old Model', 'Soft Launch','New Model']
    for idx, date in enumerate(plt['Price'].value_counts().index.sort_values(ascending=False)):
        # plots_age.add_trace(go.Bar(name=str(date.strftime('%d-%m-%Y')), x=plt[plt['Price']==date]['Age'], y=plt[plt['Price']==date]['Policy No'],
        plots_age.add_trace(go.Bar(name=date.strftime('%d-%m-%Y'), x=plt[plt['Price'] == date]['Age'],
                                   y=plt[plt['Price'] == date]['Policy No'],
                                   marker_color=colors[idx], opacity=0.3), secondary_y=True)
    # Plotting Sales Proportion
    for idx, date in enumerate(plt['Price'].value_counts().index.sort_values(ascending=False)):
        # plots_age.add_trace(go.Scatter(name=str(date.strftime('%d-%m-%Y')), x=plt[plt['Price']==date]['Age'], y=plt[plt['Price']==date]['Prop'], marker_color = colors[idx]))
        plots_age.add_trace(go.Scatter(name=date.strftime('%d-%m-%Y'), x=plt[plt['Price'] == date]['Age'],
                                       y=plt[plt['Price'] == date]['Prop'], marker_color=colors[idx]))

    plots_age.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    # plots_age.update_xaxes(
    #     dtick="M1",
    #     tickformat="%d-%m-%Y")

    plots_age.update_yaxes(title_text="Policies Sold %", secondary_y=False)
    plots_age.update_yaxes(title_text="Number of Policies Sold", secondary_y=True)
    plots_age.update_layout(height=600, width=1200, title_text="Sales by Age")
    plots_age.update_layout(barmode='stack')

    # plots_age.show()
    return plots_age


# In[229]:


# Average Premium Chart by Age
@st.cache
def agePlotAvgPrem_select(df, brands_select, regions_select, age_select, prices_strategies):
    df_ = df[(df['Brand English'].isin(brands_select)) & (df['Insured Region'].isin(regions_select)) &
             (df['Age'].isin(age_select)) & (df['Price'].isin(prices_strategies))]

    # Plotting
    plots_age = make_subplots(specs=[[{"secondary_y": True}]])
    avg_prem = df_.groupby(['Age', 'Price']).mean().reset_index()
    avg_prem = avg_prem.dropna(subset=['Gross Premium'])
    avg_prem['Price'] = pd.to_datetime(avg_prem['Price'], format='%d-%m-%Y')
    avg_prem = avg_prem.sort_values(by=['Price', 'Age'], ascending=False)

    color_map = {}
    for idx, i in enumerate(avg_prem['Price'].value_counts().index.values):
        color_map[i] = colors[idx]
    color_array = avg_prem['Price'].map(color_map)

    # Stacked bars for exposures
    plt = df_.groupby(['Age', 'Price']).count().reset_index()
    # plt['Prop'] = plt['Policy No'] / plt.groupby('Price')['Policy No'].transform('sum')*100 # Calculating proportion of policies by Age
    plt['Price'] = pd.to_datetime(plt['Price'], format='%d-%m-%Y')
    plt = plt.sort_values(by=['Price', 'Age'], ascending=False)

    # plot_names = ['Old Model', 'Soft Launch','New Model']
    for idx, date in enumerate(plt['Price'].value_counts().index.sort_values(ascending=False)):
        # plots_age.add_trace(go.Bar(name=str(date.strftime('%d-%m-%Y')), x=plt[plt['Price']==date]['Age'], y=plt[plt['Price']==date]['Policy No'],
        plots_age.add_trace(go.Bar(name=date.strftime('%d-%m-%Y'), x=plt[plt['Price'] == date]['Age'],
                                   y=plt[plt['Price'] == date]['Policy No'],
                                   marker_color=colors[idx], opacity=0.3), secondary_y=True)

    # Plotting line charts for average premium
    for idx, date in enumerate(avg_prem['Price'].value_counts().index.sort_values(ascending=False)):
        # plots_age.add_trace(go.Scatter(name=str(date.strftime('%d-%m-%Y')), x=avg_prem[avg_prem['Price']==date]['Age'], y=avg_prem[avg_prem['Price']==date]['Gross Premium'],
        plots_age.add_trace(go.Scatter(name=date.strftime('%d-%m-%Y'), x=avg_prem[avg_prem['Price'] == date]['Age'],
                                       y=avg_prem[avg_prem['Price'] == date]['Gross Premium'],
                                       marker_color=colors[idx]))
    plots_age.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    plots_age.update_yaxes(title_text="Number of Policies Sold", secondary_y=True)
    plots_age.update_yaxes(title_text="Average Premium Sold SAR", secondary_y=False)
    plots_age.update_layout(height=600, width=1200, title_text="Average Premium by Age")
    plots_age.update_layout(barmode='stack')
    df_['Price'] = pd.to_datetime(df_['Price'], format='%d-%m-%Y')
    avgprem = df_.groupby(['Price']).mean().reset_index().loc[:, ['Price', 'Gross Premium']].sort_values(by=['Price'],
                                                                                                         ascending=False)
    # plots_age.show()
    return plots_age, avgprem


# In[7]:


# prices_strategies


# In[ ]:


# brands = df['Brand English'].unique()
# regions = df['Insured Region'].unique()
# ages = df['Age'].unique()
# # prices_strategies = df['Price'].unique()
# prices_strategies = [prices[0], prices[1], prices[2]]#, prices[3]]
# # agePlotSalesProp_select(df, brands, regions, ages, prices_strategies)
# agePlotSalesProp_select(df, brands_select = brands, regions_select = regions, age_select=ages, prices_strategies = prices_strategies)


# In[230]:


# brands = df['Brand English'].unique()
# regions = df['Insured Region'].unique()
# ages = df['Age'].unique()
# prices_strategies = [prices[0], prices[1], prices[2]]
# plot, avgprem = agePlotAvgPrem_select(df, brands, regions, ages, prices_strategies)
# plot


# In[553]:


# agePlot(df)


# ### By Region

# In[95]:


# brands = df['Brand English'].unique()
# regions = df['Insured Region'].unique()
# ages = df['Age'].unique()
# # prices_strategies = df['Price'].unique()
# prices_strategies = [prices[0], prices[1], prices[2]]
# regionPlot_select(df, brands, regions, ages, prices_strategies)


# In[143]:


@st.cache
def regionPlot_select(df, brands_select, regions_select, age_select, prices_strategies):
    df_ = df[(df['Brand English'].isin(brands_select)) & (df['Insured Region'].isin(regions_select)) &
             (df['Age'].isin(age_select)) & (df['Price'].isin(prices_strategies))]

    # Plotting
    plots = make_subplots(specs=[[{"secondary_y": True}]])
    avg_prem = df_.groupby(['Insured Region', 'Price']).mean().reset_index()
    avg_prem = avg_prem.dropna(subset=['Gross Premium'])
    avg_prem['Price'] = pd.to_datetime(avg_prem['Price'], format='%d-%m-%Y')
    avg_prem = avg_prem.sort_values(by=['Price', 'Insured Region'], ascending=False)

    color_map = {}
    for idx, i in enumerate(avg_prem['Price'].value_counts().index.values):
        color_map[i] = colors[idx]
    color_array = avg_prem['Price'].map(color_map)

    # Stacked bars for exposures
    plt = df_.groupby(['Insured Region', 'Price']).count().reset_index()
    # plt['Prop'] = plt['Policy No'] / plt.groupby('Price')['Policy No'].transform('sum')*100 # Calculating proportion of policies by Age
    plt['Price'] = pd.to_datetime(plt['Price'], format='%d-%m-%Y')
    plt = plt.sort_values(by=['Price', 'Insured Region'], ascending=False)

    # plot_names = ['Old Model', 'Soft Launch','New Model']
    for idx, date in enumerate(plt['Price'].value_counts().index):
        # plots.add_trace(go.Bar(name=str(date.strftime('%d-%m-%Y')), x=plt[plt['Price']==date]['Age'], y=plt[plt['Price']==date]['Policy No'],
        plots.add_trace(go.Bar(name=date.strftime('%d-%m-%Y'), x=plt[plt['Price'] == date]['Insured Region'],
                               y=plt[plt['Price'] == date]['Policy No'],
                               marker_color=colors[idx], opacity=0.3), secondary_y=False)

    # Plotting line charts for average premium
    for idx, date in enumerate(avg_prem['Price'].value_counts().index):
        # plots.add_trace(go.Scatter(name=str(date.strftime('%d-%m-%Y')), x=avg_prem[avg_prem['Price']==date]['Age'], y=avg_prem[avg_prem['Price']==date]['Gross Premium'],
        plots.add_trace(
            go.Scatter(name=date.strftime('%d-%m-%Y'), x=avg_prem[avg_prem['Price'] == date]['Insured Region'],
                       y=avg_prem[avg_prem['Price'] == date]['Gross Premium'],
                       marker_color=colors[idx]), secondary_y=True)
    plots.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    # plots_region.add_trace(traces[0])
    plots.update_xaxes(type='category')  # , categoryarray=regions_select)
    plots.update_yaxes(title_text="No of Policies Sold", secondary_y=False)
    plots.update_yaxes(title_text="Average Premium", secondary_y=True)
    plots.update_layout(height=600, width=800, title_text="Policy Distribution By Region")
    # plots_region.show()
    return plots


# #### Production by Top 10 Brands

# In[193]:


# brandsPlot_select(df, brands_select, regions_select, age_select, prices_strategies, priceRegime='oldbrands')


# In[192]:


@st.cache
def brandsPlot_select(df, brands_select, regions_select, age_select, prices_strategies, priceRegime='oldbrands'):
    df_ = df[(df['Brand English'].isin(brands_select)) & (df['Insured Region'].isin(regions_select)) &
             (df['Age'].isin(age_select)) & (df['Price'].isin(prices_strategies))]

    # Plotting
    plots = make_subplots(specs=[[{"secondary_y": True}]])

    # Stacked bars for exposures
    plt = df_.groupby(['Brand English', 'Price']).count().reset_index()
    plt['Price'] = pd.to_datetime(plt['Price'], format='%d-%m-%Y')
    plt = plt.sort_values(by=['Price', 'Policy No'], ascending=False)

    # Average Premium
    avg_prem = df_[df_['Brand English'].isin(plt['Brand English'])].groupby(
        ['Brand English', 'Price']).mean().reset_index()
    avg_prem = avg_prem.dropna(subset=['Gross Premium'])
    avg_prem['Price'] = pd.to_datetime(avg_prem['Price'], format='%d-%m-%Y')
    avg_prem = avg_prem.sort_values(by=['Price', 'Brand English'], ascending=False).iloc[0:10]

    color_map = {}
    for idx, i in enumerate(avg_prem['Price'].value_counts().index.values):
        color_map[i] = colors[idx]
    color_array = avg_prem['Price'].map(color_map)

    # Stacked bars for exposures
    # plot_names = ['Old Model', 'Soft Launch','New Model']
    top_brands = pd.Series(dtype='category')
    for idx, date in enumerate(plt['Price'].value_counts().index.sort_values(ascending=False)):
        # plots.add_trace(go.Bar(name=str(date.strftime('%d-%m-%Y')), x=plt[plt['Price']==date]['Age'], y=plt[plt['Price']==date]['Policy No'],
        plt_ = plt[plt['Price'] == date].sort_values(by=['Policy No'], ascending=False).iloc[0:10]
        top_brands = pd.concat([top_brands, plt_['Brand English']])

        # Average Premium Calculation
        avg_prem = df_[(df_['Brand English'].isin(top_brands)) & (df_['Price'] == date.strftime('%d-%m-%Y'))].groupby(
            ['Brand English']).mean().reset_index()
        avg_prem = avg_prem.dropna(subset=['Gross Premium'])

        plots.add_trace(go.Bar(name=date.strftime('%d-%m-%Y'), x=plt_['Brand English'], y=plt_['Policy No'],
                               marker_color=colors[idx], opacity=0.3), secondary_y=False)

        plots.add_trace(
            go.Scatter(name=date.strftime('%d-%m-%Y'), x=avg_prem['Brand English'], y=avg_prem['Gross Premium'],
                       marker_color=colors[idx], mode='markers'), secondary_y=True)

    # # Plotting line charts for average premium
    # for idx, date in enumerate (avg_prem['Price'].value_counts().index):
    #     # plots.add_trace(go.Scatter(name=str(date.strftime('%d-%m-%Y')), x=avg_prem[avg_prem['Price']==date]['Age'], y=avg_prem[avg_prem['Price']==date]['Gross Premium'],
    #     plots.add_trace(go.Scatter(name=date.strftime('%d-%m-%Y'), x=avg_prem[avg_prem['Price']==date]['Brand English'].iloc[0:10], y=avg_prem[avg_prem['Price']==date]['Gross Premium'],
    #                                    marker_color = colors[idx]), secondary_y=True)
    plots.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    top_brands = top_brands.drop_duplicates()
    plots.update_xaxes(type='category', categoryarray=top_brands)
    plots.update_yaxes(title_text="No of Policies Sold", secondary_y=False)
    plots.update_yaxes(title_text="Average Premium", secondary_y=True)
    plots.update_layout(height=600, width=800)
    # plots_oldbrands.show()
    return plots


# In[10]:


@st.cache
def brands_df_select(df, prices_strategies):
    # merged new and old brands to create a dataframe

    # prices_strategies represents the list of stratgies in the dataframe
    newprice = prices_strategies[0]  # latest strategy
    oldprice = prices_strategies[1]  # latest - 1 strategy

    oldPricedf = df[df['Price'] == oldprice]
    newPricedf = df[df['Price'] == newprice]

    oldbrands = oldPricedf.groupby('Brand English').count().sort_values(by='D1 ID', ascending=False).reset_index()[
                    'Brand English'][0:10]
    newbrands = newPricedf.groupby('Brand English').count().sort_values(by='D1 ID', ascending=False).reset_index()[
                    'Brand English'][0:9]

    brands = pd.concat([oldbrands, newbrands])
    brands = brands.drop_duplicates()

    oldPrice = []
    newPrice = []
    for brand in brands:
        oldPrice.append(oldPricedf[oldPricedf['Brand English'] == brand].loc[:, 'Gross Premium'].sum() / oldPricedf[
                                                                                                             oldPricedf[
                                                                                                                 'Brand English'] == brand].loc[
                                                                                                         :,
                                                                                                         'D1 ID'].count())
        newPrice.append(newPricedf[newPricedf['Brand English'] == brand].loc[:, 'Gross Premium'].sum() / newPricedf[
                                                                                                             newPricedf[
                                                                                                                 'Brand English'] == brand].loc[
                                                                                                         :,
                                                                                                         'D1 ID'].count())

    priceComp = pd.DataFrame(data={'BrandName': brands, 'OldPrice': oldPrice, 'NewPrice': newPrice})
    priceComp['PriceVar%'] = (priceComp['NewPrice'] / priceComp['OldPrice'] - 1) * 100
    return priceComp


# # Weekly Sales Analysis

# In[200]:


@st.cache
def w_sales(df, brands_select, regions_select, age_select):
    df_ = df[(df['Brand English'].isin(brands_select)) & (df['Insured Region'].isin(regions_select)) &
             (df['Age'].isin(age_select))]

    plt = df_.groupby(['WC_Date', 'Price']).count().reset_index()

    # Plotting
    color_map = {}
    for idx, i in enumerate(plt['Price'].value_counts().index.values):
        color_map[i] = colors[idx]

    color_array = plt['Price'].map(color_map)

    fig_wSales = make_subplots(specs=[[{"secondary_y": True}]])
    plt['Price'] = pd.to_datetime(plt['Price'], format='%d-%m-%Y')
    plt = plt.sort_values(by=['Price', 'WC_Date'], ascending=False)

    avg_prem = df_.groupby(['WC_Date', 'Price']).mean().reset_index()
    avg_prem = avg_prem.dropna(subset=['Gross Premium'])
    fig_wSales.add_trace(
        go.Scatter(name='Avg Premium', x=avg_prem['WC_Date'], y=avg_prem['Gross Premium'], showlegend=True,
                   marker_color='green'), secondary_y=True)

    for idx, date in enumerate(plt['Price'].value_counts().index):
        fig_wSales.add_trace(go.Bar(name=str(date.strftime('%d-%m-%Y')), x=plt[plt['Price'] == date]['WC_Date'],
                                    y=plt[plt['Price'] == date]['Policy No'], marker_color=colors[idx]))

    fig_wSales.update_layout(barmode='stack')
    fig_wSales.update_yaxes(title_text="Avg Premium", secondary_y=True)

    fig_wSales.update_xaxes(
        dtick="M1",
        tickformat="%d-%m-%Y")
    fig_wSales.update_yaxes(title_text='No of Policies Sold')
    fig_wSales.update_layout(height=400, width=1200, title_text="Weekly Policies Sold - New and Old Prices")
    return fig_wSales


# # Weekly Sales Analysis - New vs Renewal

# In[10]:


@st.cache
def w_sales_newRenewRatio(df, brands_select, regions_select, age_select):
    plt = df.groupby(['WC_Date', 'Policy Type']).count().reset_index()
    plt['Weekly Sales'] = plt.groupby(['WC_Date']).transform('sum')['Policy No']
    plt['SalesRatio'] = plt['Policy No'] / plt['Weekly Sales']
    plt['SalesRatio'] = plt['SalesRatio'].apply(lambda x: format(x, '.1%'))

    fig = px.bar(plt, x='WC_Date', y='Policy No', color='Policy Type', text='SalesRatio',
                 color_discrete_map={'New': 'blue', 'Renew': 'green'})
    fig.update_xaxes(
        dtick="M1",
        tickformat="%d-%m-%Y")
    fig.update_yaxes(title_text='No of Policies Sold')
    fig.update_layout(height=400, width=1200, title_text="Weekly Policies Sold - New and Old Prices")

    #     fig.show()
    return fig


# In[636]:


# w_sales(df, brands, regions)


# # Test Plotting Area

# In[100]:


# Weekly Sales Trends across all Brands and Regions
# brands = df['Brand English'].unique()
# regions = df['Insured Region'].unique()
# ages = df['Age'].unique()
# w_sales(df, brands, regions, ages)


# In[13]:


# w_sales_newRenewRatio(df, brands, regions, ages)


# In[ ]:


# In[ ]:


# # Streamlit Dashboard

# In[ ]:


st.title('Motor Third Party')

# ### Overall Section

# In[617]:


mts_price_change_week_oldstg = pd.to_datetime('4-08-2021',
                                              format='%d-%m-%Y').week  # Not used other than quotation conversion chart
# mts_price_change_week_oldstg = pd.to_datetime(prices[1],format='%d-%m-%Y').week


# In[137]:


mts_price_change_week = pd.to_datetime('19-04-2022',
                                       format='%d-%m-%Y').week  # Not used other than quotation conversion chart
# mts_price_change_week = pd.to_datetime(prices[0],format='%d-%m-%Y').week


# In[ ]:


with st.container():
    st.subheader('Weekly Sales Trends across all Brands and Regions')
    brands = df['Brand English'].unique()
    regions = df['Insured Region'].unique()
    ages = df['Age'].unique()

    fig_wSales = w_sales(df, brands, regions, ages)
    st.plotly_chart(fig_wSales)

# In[ ]:


with st.container():
    st.subheader('New vs Renew across all Brands and Regions')
    brands = df['Brand English'].unique()
    regions = df['Insured Region'].unique()
    ages = df['Age'].unique()

    fig_wSales = w_sales_newRenewRatio(df, brands, regions, ages)
    st.plotly_chart(fig_wSales)


# In[ ]:


# Weekly Conversion Rates
@st.cache
def convRates_agg():
    convData = pd.read_excel(
        r"Z:\R&D - TECHNICAL SECTION\GENERAL DIVISION\MOTOR TARIFF REVIEW\Python Projects\Motor Dashboard\MTS\Code\ConversionRates_MTS_Jan-May22.xlsx",
        sheet_name='Data')
    convData_agg = convData.groupby(['Week']).sum().reset_index()
    convData_agg['Price'] = convData_agg['Week'].apply(
        lambda x: 'NewPrice' if x >= mts_price_change_week else 'OldPrice')
    convData_agg['ConvRates'] = np.round(convData_agg['Policy No'] / convData_agg['Quot.No'] * 100, 1)
    return convData_agg


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Quotation to Sales Conversion Rates - Tameeni')
        convData_agg = convRates_agg()
        convRates = px.bar(convData_agg, x='Week', y='ConvRates', color='Price')
        convRates.update_layout(height=400, width=1000)

        st.plotly_chart(convRates)

# #### Selecting Strategies for Comparison

# In[ ]:


# Selecting Strategies for Comparison
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        current_price = st.multiselect('Current Prices', prices, default=prices[0])

    with col2:
        old_price = st.multiselect('Old Prices', prices, default=prices[1])
        prices_strategies = []

        # Flatening the multiselect list
        for price in current_price:
            prices_strategies.append(price)

        for price in old_price:
            prices_strategies.append(price)

        # prices_strategies = [current_price, old_price]
        # st.write(prices_strategies)

# In[ ]:


# Ploting Distribution by Age - Sales Proportions
with st.container():
    # col1 = st.columns(1)
    # with col1:
    st.subheader('Sales Proportion by Age')
    agePlot = agePlotSalesProp_select(df, brands_select=brands, regions_select=regions, age_select=ages,
                                      prices_strategies=prices_strategies)
    st.plotly_chart(agePlot)

# In[ ]:


# Ploting Distribution by Age - Average Premium & Exposures
with st.container():
    # col1 = st.columns(1)
    # with col1:
    st.subheader('Average Premium by Age')
    agePlot, avgprem = agePlotAvgPrem_select(df, brands_select=brands, regions_select=regions, age_select=ages,
                                             prices_strategies=prices_strategies)
    st.plotly_chart(agePlot)

# In[ ]:


# Ploting Distribution by Regions
with st.container():
    # col1 = st.columns(1)
    # with col1:
    st.subheader('Distribution of Sales by Regions')
    regionPlot = regionPlot_select(df, brands_select=brands, regions_select=regions, age_select=ages,
                                   prices_strategies=prices_strategies)
    st.plotly_chart(regionPlot)

# In[ ]:


# Ploting Distribution by Brands
with st.container():
    # col1 = st.columns(1)
    # with col1:
    st.subheader('Top 10 Brands')
    plot = brandsPlot_select(df, brands_select=brands, regions_select=regions, age_select=ages,
                             prices_strategies=prices_strategies, priceRegime='oldbrands')
    st.plotly_chart(plot)

    # with col2:
    #     st.subheader('Top 10 Brands under New Prices')
    #     plot = brandsPlot_select(df, brands_select = brands, regions_select=regions, age_select=ages, prices_strategies = prices_strategies, priceRegime='newbrands')
    #     st.plotly_chart(plot)

# ## Filtering Section - Brand

# #### Multiselect Box

# In[550]:


# Multiselect Boxes
with st.container():
    st.header('Analysis by Brand')
    col1, col2 = st.columns(2)

    with col1:
        brands_select = st.multiselect('Brands', brands, default='HYUNDAI ACCENT')

#     with col2:
#         regions_select = st.multiselect('Regions', regions, default = 'Central Region Main')

# df = df[(df['Brand English'].isin(brands_select)) & (df['Insured Region'].isin(regions_select))]


# ### Charts for a Single Brand

# In[ ]:


# Ploting Weekly Sales - New vs Old Prices
st.subheader('Weekly Sales Trend')
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        fig_wSales = w_sales(df, brands_select, regions_select=regions, age_select=ages)
        st.plotly_chart(fig_wSales)

# In[ ]:


# Ploting Distribution by Age & Regions
with st.container():
    # col1, col2 = st.columns(2)
    # with col1:
    st.subheader('Sales by Age')
    plot, avgprem = agePlotAvgPrem_select(df, brands_select, regions_select=regions, age_select=ages,
                                          prices_strategies=prices_strategies)
    # agePlot, avgprem = agePlotAvgPrem_select(df, brands_select = brands, regions_select = regions, age_select=ages, prices_strategies = prices_strategies)
    st.plotly_chart(plot)

# In[ ]:


# Ploting Distribution by Age & Regions
with st.container():
    # col1, col2 = st.columns(2)
    # with col2:
    st.subheader('Sales by Regions')
    plot = regionPlot_select(df, brands_select, regions_select=regions, age_select=ages,
                             prices_strategies=prices_strategies)
    st.plotly_chart(plot)

# ## Filtering Section - Age

# #### Multiselect Box

# In[550]:


# Multiselect Boxes
with st.container():
    st.header('Analysis by Age')
    col1, col2 = st.columns(2)

    with col1:
        age_select = st.multiselect('Age', ages, default=30)

#     with col2:
#         regions_select = st.multiselect('Regions', regions, default = 'Central Region Main')

# df = df[(df['Brand English'].isin(brands_select)) & (df['Insured Region'].isin(regions_select))]


# ### Charts for a Single Age

# In[ ]:


# Ploting Weekly Sales - New vs Old Prices
st.subheader('Weekly Sales Trend')
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        fig_wSales = w_sales(df, brands_select=brands, regions_select=regions, age_select=age_select)
        st.plotly_chart(fig_wSales)

# In[ ]:


# Ploting Distribution by Age & Regions
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Sales by Regions')
        plot = regionPlot_select(df, brands_select=brands, regions_select=regions, age_select=age_select,
                                 prices_strategies=prices_strategies)
        st.plotly_chart(plot)

# In[ ]:


# Ploting Distribution by Brands
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Top 10 Brands under Old Prices')
        plot = brandsPlot_select(df, brands_select=brands, regions_select=regions, age_select=age_select,
                                 prices_strategies=prices_strategies, priceRegime='oldbrands')
        st.plotly_chart(plot)

    with col2:
        st.subheader('Top 10 Brands under New Prices')
        plot = brandsPlot_select(df, brands_select=brands, regions_select=regions, age_select=age_select,
                                 prices_strategies=prices_strategies, priceRegime='newbrands')
        st.plotly_chart(plot)

# In[ ]:





