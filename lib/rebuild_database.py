from lib.index_builder import index_loader
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)

def main():
    print("Rebuilding database with hybrid retrieval...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Make sure you have a .env file with: OPENAI_API_KEY=your_key_here")
        return
    
    print(f"API Key loaded: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        retriever = index_loader(force_rebuild=True)
        print("Database rebuilt successfully with hybrid retrieval (Dense + Sparse)")
        print("Now you can run: streamlit run app.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()