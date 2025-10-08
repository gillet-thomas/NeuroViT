# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# This code is contributed by Amiya Rout
path = '/mnt/data/iai/datasets/fMRI_marian/behavioural_data.csv'
df = pd.read_csv(path)

# correlation_matrix = df.corr(method='spearman')
# print(correlation_matrix)
# excel = correlation_matrix.to_excel('behavioural_correlation_matrix2.xlsx')


# Drop non-numeric columns (if any)
df_numeric = df.select_dtypes(include=[np.number])

# Initialize correlation & p-value matrices
corr_matrix = pd.DataFrame(index=df_numeric.columns, columns=df_numeric.columns)
pval_matrix = pd.DataFrame(index=df_numeric.columns, columns=df_numeric.columns)

# Compute Pearson correlation & p-values
for col1 in df_numeric.columns:
    for col2 in df_numeric.columns:
        if col1 == col2:
            corr_matrix.loc[col1, col2] = 1  # Correlation of a feature with itself is 1
            pval_matrix.loc[col1, col2] = 0  # P-value for self-correlation is 0
        else:
            corr, p_val = pearsonr(df_numeric[col1], df_numeric[col2])
            corr_matrix.loc[col1, col2] = corr
            pval_matrix.loc[col1, col2] = p_val

# Convert to float type for better visualization
corr_matrix = corr_matrix.astype(float)
pval_matrix = pval_matrix.astype(float)

# Save the correlation and p-value matrices
corr_matrix.to_excel('correlation_matrix.xlsx')
pval_matrix.to_excel('p_value_matrix.xlsx')

# Print the first few results
print("\n📊 Pearson Correlation Matrix:")
print(corr_matrix.head())

print("\n📊 P-Value Matrix:")
print(pval_matrix.head())

# Plot P-value Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pval_matrix, annot=True, cmap="coolwarm", fmt=".2g")
plt.title("P-value Heatmap")
plt.show()