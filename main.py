# Creating a Book Recommender System using Python and Gradio 
# Using LLM to generate recommendations based on user input


from model import get_model, generate_embeddings
from utils import text_to_string, find_title_in_question
from indexer import create_faiss_index, save_index, load_index
from search import search_books

import gradio as gr
import pandas as pd
import os
from collections import Counter
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



# Load data
df = pd.read_csv("books.csv")

# Format each row as a full descriptive string
df["full_text"] = df.apply(text_to_string, axis=1)

# Load sentence-transformer model
model = get_model()
# Path to save the FAISS index
index_path = "book_recommender.index"
# Check if the index already exists
if os.path.exists(index_path):
    print(f"Loading existing index from {index_path}")
    index = load_index(index_path)
else:
    print(f"Creating a new index and saving it to {index_path}")
    # Generate embeddings
    embeddings = generate_embeddings(model, df["full_text"])
    # Create and save FAISS index
    index = create_faiss_index(embeddings, model.get_sentence_embedding_dimension())
    save_index(index, index_path)


# Use this:
# gpt_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# gpt_model = AutoModelForCausalLM.from_pretrained("distilgpt2")


gpt_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gpt_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")



def recommend_books(prompt, selected_categories, min_rating, year):
    results, descriptions = search_books(prompt, model, index, df, top_k=len(df))
    results = results.copy()
    results["description"] = descriptions

    # Filter by category, rating, year
    if selected_categories:
        results = results[results['categories'].str.contains('|'.join(selected_categories), case=False, na=False)]
    if min_rating is not None:
        results = results[results["average_rating"] >= min_rating]
    if year:
        results = results[results["published_year"] == year]

    # Take top 20 after filtering
    results = results.head(20)

    # Build HTML cards
    cards = []
    for _, row in results.iterrows():
        card = f"""
        <div style="min-width: 250px; max-width: 250px; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 10px;">
            <img src="{row['thumbnail']}" alt="Cover" style="width:100%; border-radius: 8px;" />
            <h3>{row['title']}</h3>
            <p><strong>Author:</strong> {row['authors']}</p>
            <p><strong>Rating:</strong> {row['average_rating']}</p>
            <p style="font-size: 0.9em;">{str(row['description'])[:300]}...</p>
        </div>
        """
        cards.append(card)

    # Wrap all cards in a horizontal scroll container
    html_output = f"""
    <div style="display: flex; overflow-x: auto; padding: 10px;">
        {''.join(cards)}
    </div>
    """
    return html_output




# Prepare unique filter values
all_categories = [cat for cats in df["categories"].dropna() for cat in cats.split(', ')]
top_categories = [cat for cat, _ in Counter(all_categories).most_common(10)]
unique_categories = sorted(top_categories)
unique_years = sorted(df["published_year"].dropna().astype(int).unique())


# def answer_question(question):
#     matched_title = find_title_in_question(question, df['title'].tolist(), threshold=0.85)

#     if matched_title:
#         row = df[df['title'].str.lower() == matched_title].iloc[0]

#         # Build context from row
#         context = f"""
# Title: {row['title']}
# Author: {row['authors']}
# Categories: {row['categories']}
# Published Year: {int(row['published_year']) if pd.notna(row['published_year']) else "Unknown"}
# Rating: {round(row['average_rating'], 2) if pd.notna(row['average_rating']) else "Unknown"}
# Pages: {int(row['num_pages']) if pd.notna(row['num_pages']) else "Unknown"}
# Description: {str(row['description']) if pd.notna(row['description']) else "No description available."}
# """

#         # Simple prompt
#         prompt = f"""You are a helpful assistant. Using only the book information below, write a short and friendly paragraph introducing the book to a reader. Do not add extra information.

# {context}

# Paragraph:"""

#         inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True)
#         output_ids = gpt_model.generate(
#             inputs["input_ids"],
#             max_length=inputs["input_ids"].shape[1] + 100,
#             pad_token_id=gpt_tokenizer.eos_token_id,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95
#         )
#         response = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         return response.split("Paragraph:")[-1].strip()

#     # If no match — basic message (no semantic fallback)
#     return "Sorry, I couldn't find an exact match for that book. Please check the title and try again."

def answer_question(question):
    matched_title = find_title_in_question(question, df['title'].tolist(), threshold=0.85)

    if matched_title:
        row = df[df['title'].str.lower() == matched_title].iloc[0]

        context = f"""Title: {row['title']}
Author: {row['authors']}
Categories: {row['categories']}
Published Year: {int(row['published_year']) if pd.notna(row['published_year']) else "Unknown"}
Rating: {round(row['average_rating'], 2) if pd.notna(row['average_rating']) else "Unknown"}
Pages: {int(row['num_pages']) if pd.notna(row['num_pages']) else "Unknown"}
Description: {str(row['description']) if pd.notna(row['description']) else "No description available."}"""

        prompt = f"""You are a helpful assistant. Using the book information below, write a warm and friendly paragraph that introduces the book to a reader. Do not use headings like "Title" or "Description" in your response.

BOOK INFO:
{context}

INTRODUCTION:"""

        inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True)

        # output_ids = gpt_model.generate(
        #     inputs["input_ids"],
        #     # max_length=inputs["input_ids"].shape[1] + 100,
        #     max_length=256,
        #     # pad_token_id=gpt_tokenizer.eos_token_id,
        #     # do_sample=True,
        #     top_k=50,
        #     top_p=0.95,
        #     temperature=0.85,
        #     repetition_penalty=1.2
        # )

        output_ids = gpt_model.generate(
            inputs["input_ids"],
            max_length=256,
            top_k=50,
            top_p=0.95,
            temperature=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.1
            )


        # raw_response = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # response = raw_response.split("INTRODUCTION:")[-1].strip()
        response = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()

        # Filter out unwanted patterns
        clean_lines = [
            line for line in response.split('\n')
            if not line.strip().lower().startswith(("title:", "description:", "length:", "keywords:", "type:", "reviewing:"))
        ]

        return "\n".join(clean_lines).strip()

    return "Sorry, I couldn't find an exact match for that book. Please check the title and try again."



# Define both interfaces
recommendation_interface = gr.Interface(
    fn=recommend_books,
    inputs=[
        gr.Textbox(lines=2, placeholder="Describe the kind of book you want to read...", label="Search Prompt"),
        gr.Dropdown(choices=unique_categories, label="Categories", multiselect=True),
        gr.Slider(minimum=0.0, maximum=5.0, step=0.1, label="Minimum Rating", value=3.5),
        gr.Dropdown(choices=unique_years, label="Publication Year", value=None)
    ],
    outputs=gr.HTML(label="Recommendations"),
    title="📚 Book Recommender",
    description="Enter a description and scroll through visually rich book recommendations!"
)

chatbot_interface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask something about books...", label="Your Question"),
    outputs=gr.Textbox(label="Answer"),
    title="📖 Book Chatbot",
    description="Ask about genres, topics, or content — based on known book info."
)


print("🚀 Starting Gradio app setup...")

# Combine them using TabbedInterface
gr.TabbedInterface(
    interface_list=[recommendation_interface, chatbot_interface],
    tab_names=["Recommendations", "Ask the Chatbot"]
).launch()
