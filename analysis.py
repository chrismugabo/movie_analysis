 # Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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

# ===== STEP 4: Correlation Analysis =====
# Compute correlation for ratings
# Since we removed runtime, we focus only on score
correlation_matrix = merged_data.groupby('platform')[['score']].corr()

# Print correlation analysis results
print("=== Correlation Analysis ===")
print(correlation_matrix)

# Optional: Visualize Genre Distribution
top_genres = genre_counts.sum(axis=0).sort_values(ascending=False).head(10)
top_genres.plot(kind='bar', figsize=(10, 5), title='Top 10 Genres by Frequency')
plt.xlabel('Genres')
plt.ylabel('Frequency')
plt.savefig('top_genres_distribution.png')
plt.show()
