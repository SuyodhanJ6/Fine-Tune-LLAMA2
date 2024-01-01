# Loading OPENAI_API_KEY
from dotenv import load_dotenv
import os


# take environment variables from .env.
load_dotenv()  
""" Workspace Constants """
GRADIENT_WORKSPACE_ID=os.getenv("GRADIENT_WORKSPACE_ID")

""" Access token of gradients"""
GRADIENT_ACCESS_TOKEN=os.getenv("GRADIENT_ACCESS_TOKEN")

