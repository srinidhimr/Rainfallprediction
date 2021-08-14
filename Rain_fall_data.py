#!/usr/bin/env python
# coding: utf-8

# In[20]:


import requests
import numpy as np


# In[21]:


main_url='https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-monthly/access/'


# In[27]:


years=np.arange(1979, 2020)
months=np.arange(1,13)
print(years)
print(months)


# In[28]:


for year in years:
    for month in months:
        if len(str(month))==1:
            month='0'+str(month)
        else:
            month=str(month)
        url=main_url+ str(year) + '/' + 'gpcp_v02r03_monthly_d' + str(year) + month + '_c20170616.nc'
        #print(url)
        r=requests.get(url)
        open('E:/Rain_Metrics/data_gpcp/'+str(year)+month+'.nc','wb').write(r.content)
    print('Data read for year='+str(year))


# In[ ]:




