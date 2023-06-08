import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the user-item ratings data from a CSV file
data = pd.read_csv('ratings_data.csv')

# Create a user-item matrix
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')

# Calculate item-item similarity using cosine similarity
item_similarities = cosine_similarity(user_item_matrix.T)

# Function to recommend items to a user
def recommend_items(user_id, top_n):
    # Get the user's item ratings
    user_ratings = user_item_matrix.loc[user_id].dropna()

    # Calculate weighted average of item ratings based on item-item similarities
    item_scores = item_similarities.dot(user_ratings)

    # Sort items by score in descending order
    sorted_items = item_scores.sort_values(ascending=False)

    # Get top n recommended items
    top_items = sorted_items.index[:top_n]

    return top_items

# Test the recommender system
user_id = 123  # Example user ID
top_n = 5  # Number of items to recommend

recommended_items = recommend_items(user_id, top_n)

print('--- Recommended Items ---')
print(recommended_items)