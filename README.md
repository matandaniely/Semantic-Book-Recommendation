# Project Description 

Book Recommendation System using semantic embeddings and a generative chatbot interface. 
The application allows users to search for books based on two different and complimentary services:

1) A reccomendation system that genrates 20 books based on how probabale they given a search prompt, a category, rating, and publication year filters. 

2) A chatbot that given a book's title it generates a description using a large language model that is pretrained and fed more accurate info based on tabular data. 



### Choosing a Model
There are two complementry models that were used:
1) SentenceTransformer (all-MiniLM-L6-v2). A lightweight and efficient model for generating embeddings from book descriptions and user queries. These embeddings are stored and queried using FAISS for fast nearest-neighbor search. 

2. Generative Model (google). Used to answer questions about the recommended books by generating responses based on contextually relevant results. The prompt includes book descriptions retrieved via MiniLM-based semantic search.



### How I Pretrained 
1. SentenceTransformer (MiniLM) is pre-trained and loaded directly via Hugging Face (all-MiniLM-L6-v2). No additional fine-tuning is applied; only inference (encoding) is used.

2. google/flan-t5-base : These models are based on pretrained T5 (Raffel et al., 2020) and fine-tuned with instructions for better zero-shot and few-shot performance. There is one fine-tuned Flan model per T5 model size.

#### Context Length 

1. MiniLM: Processes each book’s full metadata (title, author, categories, description, year, rating, pages) into a single input string, typically well under the 512 token limit.

2. google/flan-t5-base: a maximum sequence length of 512 tokens, which means, anything beyond that will probably give bad results

#### What information is used from the book?

The following fields are combined into a single text string used for embedding generation:

- title
- authors
- categories
- description
- published_year
- average_rating
- num_pages

#### Aditional Filtering:
- Categories: Top 20 most common book categories are extracted and presented as multi-select options.

- Minimum Rating: A slider allows users to filter by Goodreads-style ratings.

- Publication Year: A dropdown menu allows filtering by year of publication.


#### Chatbot
1. You write: "Tell me about James and the Giant Peach"
2. The model finds that title via embeddings (generate_embeddings)
3. The app builds a prompt using that book’s metadata
4. The chatbot LLM (e.g., Flan-T5) generates the printed paragraph


#### Code Structure

main.py
-Loads the dataset and builds or loads the FAISS index.
-Prepares the Gradio app with two tabs:
    Recommendations tab for search + filtering.
    Chatbot tab for natural language Q&A.

model.py
- Loads the SentenceTransformer.
- Encodes text into embeddings.
- Extracts the embedding dimension dynamically.

indexer.py
- Creates, saves, and loads a FAISS index.
- Uses IndexFlatL2 for fast and accurate similarity queries.

search.py
- Converts user prompts to embeddings.
- Searches in FAISS for top-k similar books.
- Returns matching rows and their similarity scores.

utils.py
- Defines a helper function text_to_string() that converts a book’s metadata into a single string for embedding.

test_and_exploration.py
- Test script for embedding generation and similarity search.
- Useful for debugging, exploration, and verifying results.