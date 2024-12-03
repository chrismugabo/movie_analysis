 # Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

print("Script is running...")

# ===== Load Datasets =====
# Load IMDb and Netflix datasets
imdb_data = pd.read_csv('imdb_movies_cleaned.csv')
netflix_data = pd.read_csv('netflix_cleaned.csv')

# Standardize column names
imdb_data.columns = imdb_data.columns.str.strip().str.replace(" ", "_").str.lower()
netflix_data.columns = netflix_data.columns.str.strip().str.replace(" ", "_").str.lower()

# Add platform column
imdb_data['platform'] = 'IMDb'
netflix_data['platform'] = 'Netflix'

# Combine datasets
merged_data = pd.concat([imdb_data, netflix_data], ignore_index=True)

# Clean and standardize columns
# Parse release_date and extract release_year
merged_data['release_date'] = pd.to_datetime(
    merged_data['release_date'], format='%m/%d/%Y', errors='coerce'
).fillna(
    pd.to_datetime(merged_data['release_date'], format='%d-%b-%y', errors='coerce')
)
merged_data['release_year'] = merged_data['release_date'].dt.year

# Clean revenue column (remove $ symbols and commas, convert to numeric)
merged_data['revenue'] = pd.to_numeric(
    merged_data['revenue'].replace('[\$,]', '', regex=True), errors='coerce'
)

# One-hot encode genres for regression
merged_data_expanded = merged_data.copy()
merged_data_expanded = merged_data_expanded.join(
    merged_data_expanded['genre'].str.get_dummies(sep=',')
)

# Clean up column names
merged_data_expanded.columns = merged_data_expanded.columns.str.strip()

# ===== Prepare the Data =====
# Use release_year, revenue, and genres as predictors
predictors = ['release_year', 'revenue'] + [
    col for col in merged_data_expanded.columns if col.startswith('genre_')
]
target = 'score'

# Drop rows with missing values in predictors or target
regression_data = merged_data_expanded.dropna(subset=[target] + predictors)

# Split data into features (X) and target (y)
X = regression_data[predictors]
y = regression_data[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Multiple Linear Regression =====
# Fit the model using statsmodels for detailed analysis
X_train_sm = sm.add_constant(X_train)  # Add intercept
model = sm.OLS(y_train, X_train_sm).fit()

# Print the summary of the regression model
print("=== Regression Model Summary ===")
print(model.summary())

# ===== Generate Visualizations =====

# 1. Relationship Between Revenue and Score
plt.figure(figsize=(8, 5))
sns.scatterplot(x=regression_data['revenue'], y=regression_data['score'], alpha=0.5)
plt.title('Relationship Between Revenue and Score')
plt.xlabel('Revenue')
plt.ylabel('Score')
plt.savefig('revenue_vs_score.png')
plt.show()

# 2. Relationship Between Release Year and Score
plt.figure(figsize=(8, 5))
sns.scatterplot(x=regression_data['release_year'], y=regression_data['score'], alpha=0.5)
plt.title('Relationship Between Release Year and Score')
plt.xlabel('Release Year')
plt.ylabel('Score')
plt.savefig('release_year_vs_score.png')
plt.show()

# 3. Distribution of Scores
plt.figure(figsize=(8, 5))
sns.histplot(regression_data['score'], bins=30, kde=True)
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.savefig('score_distribution.png')
plt.show()

# ===== Hypothesis Testing =====
# Hypothesis tests are included in the regression summary (p-values for predictors)
# Significant predictors have p-values < 0.05
