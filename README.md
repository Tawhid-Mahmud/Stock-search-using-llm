# Stock Information Search Application

This application is a Streamlit-based web app that allows users to search for stock information using natural language queries. It leverages Pinecone for vector storage, Hugging Face for embeddings, and OpenAI's language model to generate insights about stocks.

## Features

- **Natural Language Query**: Enter queries about stocks and receive relevant information.
- **Embeddings**: Uses Hugging Face's SentenceTransformer to generate embeddings for queries.
- **Vector Search**: Utilizes Pinecone to find similar stocks based on the query.
- **Stock Information**: Fetches detailed stock information using Yahoo Finance.
- **AI Insights**: Provides AI-generated explanations for each stock using OpenAI's language model.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-info-search.git
   cd stock-info-search
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add your API keys:
     ```
     PINECONE_API_KEY=your_pinecone_api_key
     GROQ_API_KEY=your_groq_api_key
     ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the app**:
   - Enter your query in the text input field.
   - Click on "More Information about [Ticker]" to get AI-generated insights.

## Dependencies

- Streamlit
- Pinecone
- OpenAI
- Hugging Face Transformers
- Yahoo Finance
- dotenv

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Pinecone](https://www.pinecone.io/)
- [OpenAI](https://openai.com/)
- [Hugging Face](https://huggingface.co/)
- [Yahoo Finance](https://pypi.org/project/yfinance/)
