 # Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns

print("Script is running...")

# Load datasets
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

# ===== STEP 1: Descriptive Statistics =====
# Calculate descriptive statistics for ratings
ratings_stats = merged_data.groupby('platform')['score'].agg(['mean', 'median', 'std'])

# Print descriptive statistics
print("=== Descriptive Statistics for Ratings ===")
print(ratings_stats)

# ===== STEP 2: Visualize Ratings Distribution =====
plt.figure(figsize=(10, 5))
for platform in merged_data['platform'].unique():
    subset = merged_data[merged_data['platform'] == platform]
    plt.hist(subset['score'], bins=20, alpha=0.5, label=platform)

plt.title('Ratings Distribution by Platform')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('ratings_distribution.png')
plt.show()

# ===== STEP 3: Chi-Square Test for Genre Distribution =====
# Create a contingency table for genres
genre_counts = (
    merged_data['genre']
    .str.split(',')
    .explode()
    .str.strip()
    .groupby(merged_data['platform'])
    .value_counts()
    .unstack(fill_value=0)
)

# Perform Chi-Square Test
chi2, p_value, dof, expected = chi2_contingency(genre_counts)

# Print results
print("=== Chi-Square Test Results for Genre Distribution ===")
print(f"Chi-Square Value: {chi2}")
print(f"Degrees of Freedom: {dof}")
print(f"P-Value: {p_value}")

# Optional: Visualize Genre Distribution
top_genres = genre_counts.sum(axis=0).sort_values(ascending=False).head(10)
top_genres.plot(kind='bar', figsize=(10, 5), title='Top 10 Genres by Frequency')
plt.xlabel('Genres')
plt.ylabel('Frequency')
plt.savefig('top_genres_distribution.png')
plt.show()

# ===== STEP 4: Enhanced Correlation Analysis =====
# One-hot encode genres for correlation
merged_data_expanded = merged_data.copy()
merged_data_expanded = merged_data_expanded.join(
    merged_data_expanded['genre'].str.get_dummies(sep=',')
)

# Select numerical columns for correlation analysis
numerical_cols = merged_data_expanded.select_dtypes(include=['float64', 'int64']).columns

# Compute correlation matrices for IMDb and Netflix
imdb_corr_matrix = merged_data_expanded[merged_data_expanded['platform'] == 'IMDb'][numerical_cols].corr()
netflix_corr_matrix = merged_data_expanded[merged_data_expanded['platform'] == 'Netflix'][numerical_cols].corr()

# Visualize Correlation Matrices
plt.figure(figsize=(12, 6))

# IMDb Correlation Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(imdb_corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('IMDb Correlation Matrix')

# Netflix Correlation Heatmap
plt.subplot(1, 2, 2)
sns.heatmap(netflix_corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Netflix Correlation Matrix')

plt.tight_layout()
plt.savefig('enhanced_correlation_matrices.png')
plt.show()
