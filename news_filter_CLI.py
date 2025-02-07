#!/usr/bin/env python Aqeel Bohoran
# coding: utf-8

# In[1]:


try:
    from sentence_transformers import SentenceTransformer, util
except:
    get_ipython().system('pip install sentence-transformers')
try:
    import pandas as pd
except:
    get_ipython().system('pip install pandas')
try:
    import pytz
except:
    get_ipython().system('pip install pytz')
try:
    import argparse
except:
    get_ipython().system('pip install argparse')


from datetime import datetime

utc = pytz.UTC # Define timezone-aware start and end dates


# In[2]:


# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and efficient


# In[3]:
def main(path, query):
    
    # Load the dataset from a JSON Lines file
    data = pd.read_json(path, lines=True)  
    data_org = data
    data.head()
    
    
    # In[4]:
    
    
    # Cleaning data
    data['content'] = data['content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    data['title'] = data['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    data['source'] = data['source'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    data['published'] = pd.to_datetime(data['published'])
    
    
    # In[5]:
    
    
    data.head()
    
    
    # In[6]:
    
    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define reference query
    reference_query = str(query)
    
    # Generate embeddings for articles and the query
    data['combined_text'] = data['title'] + ". " + data['content']
    article_embeddings = model.encode(data['combined_text'].tolist(), convert_to_tensor=True)  # Ensure input is a list
    query_embedding = model.encode(reference_query.lower(), convert_to_tensor=True)
    
    # Compute cosine similarity
    data['similarity'] = util.cos_sim(query_embedding, article_embeddings).squeeze().cpu().numpy()
    
    # Filter and sort by relevance
    relevant_articles = data[data['similarity'] > 0.5].sort_values(by='similarity', ascending=False)
    
    # Display results
    relevant_articles[['id', 'title', 'content', 'published', 'similarity']]
    
    
    # In[7]:
    
    
    # Save relevant articles to a CSV file
    output_csv_path = "relevant_articles.csv"
    
    # Use the original 'title' and 'content' columns before cleaning
    relevant_articles[['id', 'title', 'content', 'published', 'similarity']].to_csv(output_csv_path, index=False)
    
    # Print original titles, content, and dates in sequence
    for _, row in relevant_articles.sort_values(by='published', ascending=True).iterrows():
        print(f"Date: {row['published']}")
        print(f"Title: {data_org.loc[row.name, 'title']}")  # Access the original title
        print(f"Content: {data_org.loc[row.name, 'content']}")  # Access the original content
        print("-" * 80)
    
    print(f"Relevant articles have been saved to {output_csv_path}.")
    

# In[ ]:

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Process fold and epoch values.")
    parser.add_argument('--path', type=str, required=True, help="Path to input .jsonl file.")
    parser.add_argument('--query', type=str, required=True, help="Query word.  e.g, Volkswagen")

    args = parser.parse_args()
    
    main(args.path, args.query)
# --path signal-1m-vw_volkswagen.jsonl --s_date 2015_9_1 --e_date 2015_9_15 --query Volkswagen