#!/usr/bin/env python
# coding: utf-8

#  IMPORT ALL NECESSARY LIBRARIES FOR DATA CLEANING,PREPROCESSING AND CLUSTERING

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, inconsistent
from scipy.cluster.hierarchy import fcluster
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = FutureWarning)


# UPLOAD THE DATASET TO THE NOTEBOOK USING PANDAS 

# In[2]:


#Read the CSV file using pandas
germancredit = pd.read_csv('german_credit_data.csv')


# USE A FEW FUNCTIONS TO DO A QUICK ANALYSIS OF THE DATA

# In[3]:


#Check the first five rows in the dataset
germancredit.head()


# In[4]:


#Check the important information in the dataset
germancredit.info()


# In[5]:


#Check a quick statistical review of the numerical columns in the dataset
germancredit.describe()


# In[6]:


#Check for the missing values in the dataset
germancredit.isnull().sum()


# In[7]:


#Replace missing values with the most frequent
catImputer = SimpleImputer(missing_values= np.nan, strategy = 'most_frequent')
catImputer = catImputer.fit(germancredit[['Saving accounts','Checking account']])
germancredit[['Saving accounts',
              'Checking account']] = catImputer.transform(germancredit[['Saving accounts','Checking account']])


# In[8]:


#Check for the missing values
germancredit.isnull().sum()


# EMPLOY THE LABELENCODER TO NORMALIZE THE ORDINAL CATEGORICAL VALUES INTO NUMERICAL VALUES

# In[9]:


#encode categorical values into numerical values
le = LabelEncoder()
for col in germancredit.columns:
    if germancredit[col].dtype == 'object':
        germancredit[col] = le.fit_transform(germancredit[col])


# In[10]:


#check the first 100 rows in the dataset
germancredit.head(100)


# FURTHER CLEAN THE DATA BY DROPPING THE IRRELEVANT COLUMNS

# In[11]:


#drop the irrelvant column
germancredit = germancredit.drop('Unnamed: 0', axis = 1)


# In[12]:


germancredit.head(50)


# THE RELATIONSHIPS BETWEEN SOME OF THE VARIABLES CAN BE VIEWED BY PLOTTING HISTOGRAMS AND SCATTERPLOTS

# In[13]:


#plot histograms for the variables in the dataset
germancredit.hist(figsize = (10, 10), bins = 50)
plt.show()


# CREATE A CORRELATION MATRIX OF THE VARIABLES IN THE DATASET

# In[14]:


# Create a correlation matrix of the variables in the dataset
cormat = germancredit.corr()
sns.heatmap(cormat, annot = True, cmap = 'coolwarm')
plt.show()


# PLOT A SCATTERPLOT OF JOB AGAINST CREDIT AMOUNT WHICH SHOWS THE RELATIONSHIP BETWEEN NUMBER OF JOBS AND CREDIT AMOUNT REQUESTED

# In[15]:


# Plot scatterplot of "Job" against "Credit amount"
sns.pairplot(germancredit.iloc[:,[2,6]])


# THE RELATIONSHIP BETWEEN HOUSING TYPE AND PURPOSE OF THE CREDIT TAKEN IS ALSO EXPLORED

# In[16]:


# Plot scatterplot of "Housing" against "Purpose"
sns.pairplot(germancredit.iloc[:, [3,8]])


# CREATE A BOXPLOT TO VIEW THE OUTLIERS IN THE DATASET

# In[17]:


#Create a boxplot to view outliers in the dataset
plt.figure(figsize=(15,5))
sns.boxplot(data = germancredit)
plt.xticks(rotation = 45)
plt.show()


# Distribution of 'Credit amount'

# In[18]:


#Plot the distribution of Credit amount
sns.histplot(data = germancredit, x = 'Credit amount', kde = True)
plt.show()


# In[19]:


#Plot a probability plot og Credit amount
stats.probplot(germancredit['Credit amount'], plot = plt)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('Probability Plot of Credit amount')
plt.show()


# THE DATA IS SKEWED AND NON-GAUSSIAN SO WE USE THE IQR METHOD TO GET RID OF THE OUTLIERS

# In[20]:


#Use the IQR method to remove outliers
Q1 = germancredit.quantile(0.25)
Q3 = germancredit.quantile(0.75)
IQR = Q3 - Q1
germancredit_no_outliers = germancredit[~((germancredit < (Q1 - 1.5 * IQR)) |
                                          (germancredit > (Q3 + 1.5 * IQR))).any(axis = 1)]


# In[21]:


#Create a boxplot to view outliers
plt.figure(figsize=(15,7))
sns.boxplot(data = germancredit_no_outliers)
plt.xticks(rotation = 45)
plt.show()


# THE NUMERICAL VALUES ARE NORMALIZED USING THE STANDARD SCALER

# In[22]:


#Normalize the data using Standard Scaler
Sc = StandardScaler()
germancredit_sc = Sc.fit_transform(germancredit_no_outliers)


# In[23]:


germancredit_sc


# FEATURE EXTRACTION

# In[24]:


#Perform Feature Extraction using PCA
pca = PCA(n_components = 2)
pca_germancredit = pca.fit_transform(germancredit_sc)

pca_germ = pd.DataFrame(data = pca_germancredit, columns = ['PC1', 'PC2'])


# TO IMPLEMENT THE KMEANS CLUSTERING, THE VALUE OF K(THE NUMBER OF CLUSTERS) IS FIRST DETERMINED. IN THIS CASE, THE SILHOUETTE METHOD IS USED FOR THIS

# In[25]:


#Determine the optimal number of clusters using the silhouette method
scores = []
cluster_range = range(2, 12)
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(pca_germ)
    scores.append(silhouette_score(pca_germ, kmeans.labels_))
   
best_n_clusters = cluster_range[scores.index(max(scores))]

print(f'The optimal number of clusters is {best_n_clusters}.')


# RUN K-MEANS ALGORITHM ON THE DATA

# In[26]:


#Run Kmeans algorithm on the data
kmeans = KMeans(n_clusters = best_n_clusters, init = 'k-means++', max_iter= 300, n_init= 'auto', random_state= 42)
kmeans.fit(pca_germ)


# ADD THE CLUSTERS TO THE DATASET

# In[27]:


#Add clusters to the dataset
germancredit_no_outliers['KmeansCluster'] = kmeans.labels_


# In[28]:


germancredit_no_outliers.head()


# THE CLUSTERS ARE VISUALIZED BY USING THE PCA COMPONENTS AS SELECT DATA POINTS

# In[29]:


germancredit_no_outliers.KmeansCluster.value_counts()


# In[30]:


#Visualize the clusters
centroids = kmeans.cluster_centers_
sns.scatterplot(x = 'PC1', y = 'PC2', hue = kmeans.labels_, data = pca_germ, palette= 'viridis')
plt.scatter(centroids[:, 0], centroids[:,1], c = 'red', marker = 'X', s = 200, label = 'Centroids')
plt.title('Kmeans Clusters with Centroids')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# OBTAIN A BRIEF REPRESENTATION OF THE CLUSTERS TO GAIN INSIGHTS OF THE CLUSTERS

# In[31]:


#Calculate the cluster statistics
grouped_data_kmeans = germancredit_no_outliers.groupby('KmeansCluster')
cluster_means_kmeans = grouped_data_kmeans.mean()
print(cluster_means_kmeans)


# RUN HIERARCHICAL (AGGLOMERATIVE) CLUSTERING ON THE DATASET

# USE DENDROGRAM TO DETERMINE THE OPTIMAL NUMBER OF CLUSTERS

# In[32]:


linkage_mat = linkage(germancredit_sc, method = 'ward')


# In[33]:


#Plot the dendrogram
plt.figure(figsize = (10,6))
dendrogram(linkage_mat)
plt.show()


# In[34]:


#Perform heirarchical clustering using Agglomerative Clustering and add the clusters to the dataset
agglo = AgglomerativeClustering(n_clusters = 6).fit(pca_germ)
germancredit_no_outliers['AggloCluster'] = agglo.labels_


# In[35]:


germancredit_no_outliers.head()


# In[36]:


germancredit_no_outliers.AggloCluster.value_counts()


# VISUALIZE THE AGGLOMERATIVE CLUSTERS USING THE DURATION AND CREDIT AMOUNT AS SELECT DATA POINTS

# In[37]:


#visualize the agglomerative clusters
sns.scatterplot(data = pca_germ, x = 'PC1', y = 'PC2', hue = agglo.labels_, palette= 'viridis')
plt.show()


# In[38]:


#Calculate the cluster statistics
grouped_data_agglo = germancredit_no_outliers.groupby('AggloCluster')
cluster_means_agglo = grouped_data_agglo.mean()
print(cluster_means_agglo)


# COMPARE THE PERFORMANCE OF THE CLUSTERING ALGORITHMS USING SILHOUETTE SCORE AND DAVIES-BOULDING INDEX

# In[39]:


# Silhouette Score
silhouette_kmeans = silhouette_score(pca_germ, kmeans.labels_)
silhouette_agg = silhouette_score(pca_germ, agglo.labels_)

# Davies-Bouldin Score
db_kmeans = davies_bouldin_score(pca_germ, kmeans.labels_)
db_agg = davies_bouldin_score(pca_germ, agglo.labels_)

print("KMeans: Silhouette Score :", silhouette_kmeans, "Davies-Bouldin Score :", db_kmeans)
print("Agglomerative: Silhouette Score :", silhouette_agg, "Davies-Bouldin Score :", db_agg)


# In[ ]:





# In[ ]:




