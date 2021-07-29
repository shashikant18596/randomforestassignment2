#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets


# In[100]:


dataset = datasets.load_boston()
dir(dataset)


# In[101]:


feature = pd.DataFrame(dataset.data,columns=dataset.feature_names)
feature.head(3)


# In[102]:


label = pd.DataFrame(dataset.target)
label.head(3)


# # Univariate Analysis

# In[103]:


plt.figure(figsize=(16,9))
plt.subplot(3,5,1)
plt.plot(feature['CRIM'],np.zeros_like(feature['CRIM']),'o')
plt.subplot(3,5,2)
plt.plot(feature['ZN'],np.zeros_like(feature['ZN']),'o')
plt.subplot(3,5,3)
plt.plot(feature['INDUS'],np.zeros_like(feature['INDUS']),'o')
plt.subplot(3,5,4)
plt.plot(feature['CHAS'],np.zeros_like(feature['CHAS']),'o')
plt.subplot(3,5,5)
plt.plot(feature['NOX'],np.zeros_like(feature['NOX']),'o')
plt.subplot(3,5,6)
plt.plot(feature['RM'],np.zeros_like(feature['RM']),'o')
plt.subplot(3,5,7)
plt.plot(feature['AGE'],np.zeros_like(feature['AGE']),'o')
plt.subplot(3,5,8)
plt.plot(feature['DIS'],np.zeros_like(feature['DIS']),'o')
plt.subplot(3,5,9)
plt.plot(feature['LSTAT'],np.zeros_like(feature['LSTAT']),'o')
plt.subplot(3,5,10)
plt.plot(feature['RAD'],np.zeros_like(feature['RAD']),'o')
plt.subplot(3,5,11)
plt.plot(feature['TAX'],np.zeros_like(feature['TAX']),'o')
plt.subplot(3,5,12)
plt.plot(feature['PTRATIO'],np.zeros_like(feature['PTRATIO']),'o')
plt.subplot(3,5,13)
plt.plot(feature['B'],np.zeros_like(feature['B']),'o')
plt.show()


# # Multivariate Analysis

# In[104]:


data= pd.concat([feature,label],axis=1)
data.head(2)


# In[105]:


data = data.rename(columns={0:"Target"})
data.head(3)


# In[107]:


plt.figure(figsize=(16,9))
sns.pairplot(data,hue='Target')
plt.show()


# In[108]:


corr = feature.corr()
corr.head(2)


# In[109]:


plt.figure(figsize=(16,10))
sns.heatmap(corr,annot=True)
plt.show()


# In[110]:


x = data.drop('Target',axis=1)
y = (data['Target']).astype('int')
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# # Applying random forest classifier

# In[111]:


model = RandomForestRegressor()
model.fit(x_train,y_train)


# In[112]:


print(f"Train_model_score: {model.score(x_train,y_train)}")
print(f"Test_model_score: {model.score(x_test,y_test)}")


# # Hyper Tunning By Using GridSearchCV

# In[113]:


grid_params = {"n_estimators" : [10,40,65,100],
              "max_depth" : range(2,20,1),
              "min_samples_leaf" : range(1,10,1),
              "min_samples_split" : range(2,10,1),
              "max_features" : ['auto','log2']
              }


# In[114]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=model,param_grid=grid_params,cv=5,n_jobs=-1,verbose=3)


# In[115]:


grid_search.fit(x_train,y_train)


# In[116]:


grid_search.best_params_


# In[119]:


model2 = RandomForestRegressor(n_estimators = 10,max_depth = 10,min_samples_split = 2,min_samples_leaf = 1,max_features = 'auto')
model2


# In[120]:


model2.fit(x_train,y_train)


# In[122]:


model2.score(x_test,y_test)


# In[125]:


y_predicted = model2.predict(x_test)


# In[131]:


from sklearn.metrics import r2_score,mean_squared_error
print(f'R^2 : {r2_score(y_test,y_predicted)}')
print(f'MSE : {mean_squared_error(y_test,y_predicted)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_predicted))}')


# In[ ]:




