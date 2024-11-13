# SocialVerse Recommendation System

A recommendation system for SocialVerse, combining content-based and collaborative filtering to suggest posts.

## Features
- **Content-Based Filtering**: Recommends similar posts based on categories and visual elements.
- **Collaborative Filtering**: Recommends posts based on similar user interactions.
- **Hybrid Model**: Combines content and collaborative filtering for more accurate recommendations.
- **Evaluation**: Uses click-through rate (CTR) to measure effectiveness.

## Setup and Installation

1. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn requests

2. **API Access**
Add Flic-Token in the code for API authentication.
Base URL: https://api.socialverseapp.com

## Usage
1. **Fetch Data**
    ```bash

    import requests
    base_url = "https://api.socialverseapp.com"
    headers = {"Flic-Token": "your_flic_token_here"}
    viewed_posts = requests.get(f"{base_url}/posts/view?page=1&page_size=1000", headers=headers).json()
    liked_posts = requests.get(f"{base_url}/posts/like?page=1&page_size=5", headers=headers).json()
    Hybrid Recommendation Example

1. **Hybrid Recommendation Example**  
    ```bash 
    def hybrid_recommendations(post_id, user_id, alpha=0.5):
        content_similarities = cosine_sim[post_id]
        user_similarities = user_similarity[user_id]
        hybrid_scores = alpha * content_similarities + (1 - alpha) * user_similarities
        return posts_df_clean.iloc[hybrid_scores.argsort()[-5:][::-1]]
    Evaluate with Click-Through Rate (CTR)

1. **Evaluate with Click-Through Rate (CTR)**  
    ```bash  
    import numpy as np
    def click_through_rate(recommended_posts, actual_clicks):
        clicked = np.isin(recommended_posts['id'], actual_clicks)
        return np.sum(clicked) / len(recommended_posts)

## Future Improvements
-**Tune Hyperparameters**
-**Add More Interaction Data**
-**Explore Deep Learning Models**