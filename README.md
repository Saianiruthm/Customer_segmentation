# Customer Segmentation Analysis

This Python script performs customer segmentation analysis on an e-commerce dataset using K-Means clustering. The goal is to segment customers based on their purchase behavior, which is represented by the total amount spent.

## Dataset

The dataset used is "Online Retail.xlsx", which contains transactional data, including customer IDs, invoice numbers, stock codes, descriptions, quantities, invoice dates, unit prices, and countries.

## Features

The script includes the following features:

- Data loading and exploration
- Handling missing values
- Creating a new feature for total amount spent
- Data visualization
- Log transformation for normalization
- Data scaling
- K-Means clustering
- Elbow method for determining the optimal number of clusters
- Cluster visualization
- Cluster analysis

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Ensure that the required packages are installed.
2. Place the "Online Retail.xlsx" file in the same directory as the script.
3. Run the script to perform the analysis.

## Output

The script will output several visualizations:

- Distribution of the total amount spent by customers
- Distribution of the log-transformed total amount
- Elbow curve for the optimal number of clusters
- Scatter plot of customers segmented into clusters

Additionally, the script will output the cluster analysis, showing the count, mean, minimum, and maximum total amount spent for each cluster.

## Contributing

Contributions to improve the script or analysis are welcome. Please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
