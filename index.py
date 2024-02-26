# Importing the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

data = pd.read_excel("Online Retail.xlsx")

# Handle missing values
# Exploring the data
data.head()
data.info()
data.describe()

# Checking for missing values
data.isnull().sum()

# Dropping the rows with missing values
data.dropna(inplace=True)

# Creating a new column for total amount spent by each customer
data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

# Creating a new dataframe with only CustomerID and TotalAmount
customer_data = data.groupby("CustomerID")["TotalAmount"].sum().reset_index()

# Plotting the distribution of TotalAmount
plt.figure(figsize=(10,6))
sns.distplot(customer_data["TotalAmount"])
plt.title("Distribution of Total Amount Spent by Customers")
plt.xlabel("Total Amount")
plt.ylabel("Frequency")
plt.show()

# Applying log transformation to normalize the data
customer_data["LogTotalAmount"] = np.log(customer_data["TotalAmount"] + 1)

# Plotting the distribution of LogTotalAmount
plt.figure(figsize=(10,6))
sns.distplot(customer_data["LogTotalAmount"])
plt.title("Distribution of Log Transformed Total Amount Spent by Customers")
plt.xlabel("Log Total Amount")
plt.ylabel("Frequency")
plt.show()

# Scaling the data using StandardScaler
scaler = StandardScaler()
customer_data["ScaledLogTotalAmount"] = scaler.fit_transform(customer_data[["LogTotalAmount"]])
imputer = SimpleImputer(strategy='mean')  # or median, or most_frequent for mode
customer_data['ScaledLogTotalAmount'] = imputer.fit_transform(customer_data[['ScaledLogTotalAmount']])

# Finding the optimal number of clusters using the elbow method
wcss = [] # Within cluster sum of squares
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(customer_data[["ScaledLogTotalAmount"]])
    wcss.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Based on the elbow curve, the optimal number of clusters is 4
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
kmeans.fit(customer_data[["ScaledLogTotalAmount"]])

# Assigning the cluster labels to the customer data
customer_data["Cluster"] = kmeans.labels_

# Plotting the clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x="CustomerID", y="ScaledLogTotalAmount", hue="Cluster", data=customer_data, palette="rainbow")
plt.title("Customer Segmentation Based on Total Amount Spent")
plt.xlabel("Customer ID")
plt.ylabel("Scaled Log Total Amount")
plt.legend(title="Cluster")
plt.show()

# Analyzing the clusters
customer_data.groupby("Cluster")["TotalAmount"].agg(["count", "mean", "min", "max"])
