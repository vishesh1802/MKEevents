import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from geopy.distance import geodesic


event_data = pd.read_csv("Event_data.csv")
user_data = pd.read_csv("user_data.csv")


event_data['genre'] = event_data['genre'].fillna('Other').str.strip()
event_data['venue_name'] = event_data['venue_name'].fillna('Unknown').str.strip()
event_data['description'] = event_data['description'].fillna('')

# Combine text for semantic representation
event_data['text_features'] = (
    event_data['genre'] + ' ' +
    event_data['venue_name'] + ' ' +
    event_data['description']
)



# Convert text to numerical embeddings
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(event_data['text_features'])

# Training K-NearestNeighbors model
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

def ai_recommend_by_genre(event_name, top_n=5):
    """AI/ML-based semantic genre recommender."""
    idx = event_data[event_data['event_name'].str.lower() == event_name.lower()].index
    if len(idx) == 0:
        return f" Event '{event_name}' not found"
    idx = idx[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    rec_indices = indices.flatten()[1:]
    recs = event_data.iloc[rec_indices][['event_name', 'genre', 'venue_name', 'date']]
    recs['similarity_score'] = 1 - distances.flatten()[1:]
    return recs.sort_values('similarity_score', ascending=False)

print("Genre Recommendations for 'Art Fest 2025':")
print(ai_recommend_by_genre("Art Fest 2025"))




# K-Means clustering model to group events by region
kmeans = KMeans(n_clusters=6, random_state=42)
event_data['region_cluster'] = kmeans.fit_predict(event_data[['latitude', 'longitude']])

def ai_recommend_by_region(event_id, top_n=5):
    """AI-based regional recommendation using clustering + proximity."""
    base_event = event_data[event_data['event_id'] == event_id]
    if base_event.empty:
        return f" Event ID '{event_id}' not found"
    base_cluster = base_event.iloc[0]['region_cluster']
    base_coords = (base_event.iloc[0]['latitude'], base_event.iloc[0]['longitude'])
    
    # Get events in the same cluster (local region)
    nearby_events = event_data[event_data['region_cluster'] == base_cluster].copy()
    nearby_events['distance_km'] = nearby_events.apply(
        lambda x: geodesic(base_coords, (x['latitude'], x['longitude'])).km, axis=1
    )
    nearby_events = nearby_events[event_data['event_id'] != event_id]
    return nearby_events.sort_values('distance_km').head(top_n)[
        ['event_name', 'genre', 'venue_name', 'distance_km', 'region_cluster']
    ]

print("\nRegion-based Recommendations:")
sample_event_id = user_data.iloc[0]['event_id']
print(ai_recommend_by_region(sample_event_id))
