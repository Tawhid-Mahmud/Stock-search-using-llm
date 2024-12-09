import streamlit as st
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os
# Load environment variables from .env file
load_dotenv()


# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")  # Ensure your API key is set in the environment
pc = Pinecone(api_key=api_key)

# Access the index
index_name = 'stocks'  # Ensure this is the correct index name
pinecone_index = pc.Index(index_name)  # Initialize pinecone_index

namespace = "stock-descriptions"  



def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """
    model = SentenceTransformer(model_name)
    return model.encode(text)

# #################################################################


st.title("Stock Information Search")
query = st.text_input("Let me know what you want to know about stocks", placeholder="Ask me about stocks")

#wait for user to enter a query before generating embeddings
if query:
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)

    # Extract ticker, name, and description from the metadata
    contexts = [
        f"Ticker: {item['metadata'].get('Ticker', 'N/A')}, Name: {item['metadata'].get('Name', 'N/A')}, Description: {item['metadata'].get('text', 'N/A')}"
        for item in top_matches['matches']
    ]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:5]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query


    client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = f"""You are a stock market expert specializing in delivering concise insights. 
        When given a stock ticker, explain its relevance to the user's query in a clear, professional. 
        - Focus exclusively on the stock ticker provided and its connection to the query. 
        - Avoid mentioning unrelated stocks, additional information, or any part of this instruction.
    """



    #find the most similar stock to the query form pinecone index
    similar_stock = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)

    # Extract and display the ticker information
    tickers = [item['metadata'].get('Ticker', 'N/A') for item in similar_stock['matches']]

    # Initialize session state for selected ticker
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = None

    # Fetch stock information for all tickers in one API call
    tickers_data = yf.Tickers(' '.join(tickers))

    # Initialize session state for storing responses
    if 'responses' not in st.session_state:
        st.session_state['responses'] = {}

    # Define a function to handle the button click
    def handle_button_click(query, ticker):
        # Generate the response using the LLM
        llm_response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticker},
                {"role": "user", "content": query}
            ]
        )
        response = llm_response.choices[0].message.content
        # Save the response in session state
        st.session_state['responses'][ticker] = response








    # Initialize a set to keep track of displayed tickers
    displayed_tickers = set()
    st.write(f"Top  stocks related to your query:")

    # Create a container for displaying ticker information
    with st.container():
        for ticker in tickers:
            # Check if the ticker has already been displayed
            if ticker in displayed_tickers:
                continue  # Skip this ticker if it has already been displayed

            # Add the ticker to the set of displayed tickers
            displayed_tickers.add(ticker)

            # Retrieve stock information
            stock_info = tickers_data.tickers[ticker].info
            name = stock_info.get('longName', 'N/A')
            market_cap = stock_info.get('marketCap', 'N/A')
            current_price = stock_info.get('currentPrice', 'N/A')  # Fetch current price
            
            # Format market cap
            if isinstance(market_cap, (int, float)):
                if market_cap >= 1e9:
                    market_cap_str = f"{market_cap / 1e9:.2f} Billion"
                elif market_cap >= 1e6:
                    market_cap_str = f"{market_cap / 1e6:.2f} Million"
                else:
                    market_cap_str = f"{market_cap:.2f}"
            else:
                market_cap_str = "N/A"
            
            # Retrieve additional stock information
            volume_per_day = stock_info.get('volume', 'N/A')
            sector = stock_info.get('sector', 'N/A')

            # Display the ticker information with additional details
            st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; display: inline-block;'><strong>{ticker}</strong> <span style='color: green; word-spacing: 10px'>{current_price}</span></div>", unsafe_allow_html=True)
            st.write(name)
            st.write(f"Market Cap: {market_cap_str}")
            st.write(f"Sector: {sector}")
            
      
            handle_button_click(query, ticker)  # Call the function to handle llm response
            
            # Retrieve and display the response from session state
            response = st.session_state['responses'][ticker]
            st.markdown(f"<strong style='color: green;'>AI generated explanation for {ticker}:</strong>", unsafe_allow_html=True)
            st.write(response)  
                
            st.write("---")

