# DeepSeek Chat with Streamlit and Ollama

A real-time chat application that uses the DeepSeek model through Ollama to provide streaming responses. Built with Streamlit for a beautiful and responsive user interface.

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- DeepSeek model pulled in Ollama

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure Ollama is running and the DeepSeek model is pulled:
   ```bash
   ollama pull deepseek-coder
   ```

## Running the Application

1. Activate the virtual environment if not already activated:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Features

- Real-time streaming responses
- Chat-like interface
- Message history
- Clear chat functionality
- Responsive design

## Usage

1. Type your message in the chat input at the bottom of the screen
2. Press Enter to send your message
3. Watch as the AI response streams in real-time
4. Use the "Clear Chat" button to start a new conversation

## Note

Make sure Ollama is running locally before starting the application. The application connects to Ollama at the default address (http://localhost:11434). 