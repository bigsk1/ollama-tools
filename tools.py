import os
import base64
# import json
import logging
from typing import Dict, Any
# from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
# import requests
from dotenv import load_dotenv
from search_utils import perform_search, SEARCH_PROVIDER

# Load environment variables
load_dotenv()

# Setup rich console for beautiful terminal output
console = Console()

# Configure logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("rich")

# Constants
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


class AITools:
    @staticmethod
    def create_folder(path: str) -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            os.makedirs(full_path, exist_ok=True)
            log.info(f"Created folder: {full_path}")
            return {"success": True, "message": f"Folder created at {full_path}"}
        except Exception as e:
            log.error(f"Error creating folder: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def create_file(path: str, content: str = "") -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            with open(full_path, 'w') as f:
                f.write(content)
            log.info(f"Created file: {full_path}")
            return {"success": True, "message": f"File created at {full_path}"}
        except Exception as e:
            log.error(f"Error creating file: {str(e)}")
            return {"success": False, "error": str(e)}
        
    @staticmethod
    def write_to_file(path: str, content: str) -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            with open(full_path, 'w') as f:
                f.write(content)
            log.info(f"Wrote to file: {full_path}")
            return {"success": True, "message": f"Content written to {full_path}"}
        except Exception as e:
            log.error(f"Error writing to file: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def read_file(path: str) -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            log.info(f"Read file: {full_path}")
            return {"success": True, "content": content}
        except Exception as e:
            log.error(f"Error reading file: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def list_files(path: str = ".") -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            files = os.listdir(full_path)
            log.info(f"Listed files in: {full_path}")
            return {"success": True, "files": files}
        except Exception as e:
            log.error(f"Error listing files: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def delete_file(path: str) -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            os.remove(full_path)
            log.info(f"Deleted file: {full_path}")
            return {"success": True, "message": f"File deleted: {full_path}"}
        except Exception as e:
            log.error(f"Error deleting file: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def search(query: str) -> Dict[str, Any]:
        return await perform_search(query)


    @staticmethod
    def upload_file(file: str) -> Dict[str, Any]:
        """
        Upload a file (base64 encoded string).
        
        :param file: The file to upload (base64 encoded)
        :return: A dictionary with the result of the operation
        """
        try:
            # This is a placeholder. In a real implementation, you would handle the file upload.
            file_content = base64.b64decode(file)
            log.info("File uploaded successfully")
            return {"success": True, "message": "File uploaded successfully"}
        except Exception as e:
            log.error(f"Error uploading file: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_weather(city: str) -> Dict[str, Any]:
        """
        Get the current weather for a specified city.
        
        :param city: The name of the city
        :return: A dictionary with the weather information
        """
        # This is a placeholder. In a real implementation, you would integrate with a weather API.
        log.info(f"Retrieved weather for: {city}")
        return {"success": True, "weather": f"Weather information for {city}"}

    @staticmethod
    def get_news(topic: str) -> Dict[str, Any]:
        """
        Get the latest news on a specified topic.
        
        :param topic: The news topic
        :return: A dictionary with the news articles
        """
        # This is a placeholder. In a real implementation, you would integrate with a news API.
        log.info(f"Retrieved news for topic: {topic}")
        return {"success": True, "news": f"Latest news on {topic}"}

    @staticmethod
    def translate_text(text: str, target_language: str) -> Dict[str, Any]:
        """
        Translate text to a target language.
        
        :param text: The text to translate
        :param target_language: The target language code (e.g., 'es' for Spanish)
        :return: A dictionary with the translated text
        """
        # This is a placeholder. In a real implementation, you would integrate with a translation API.
        log.info(f"Translated text to {target_language}")
        return {"success": True, "translation": f"Translated '{text}' to {target_language}"}

    @staticmethod
    def summarize_text(text: str) -> Dict[str, Any]:
        """
        Summarize a given text.
        
        :param text: The text to summarize
        :return: A dictionary with the summarized text
        """
        # This is a placeholder. In a real implementation, you would use an AI model for summarization.
        log.info("Summarized text")
        return {"success": True, "summary": f"Summary of: {text[:50]}..."}

    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a given text.
        
        :param text: The text to analyze
        :return: A dictionary with the sentiment analysis result
        """
        # This is a placeholder. In a real implementation, you would use an AI model for sentiment analysis.
        log.info("Analyzed sentiment")
        return {"success": True, "sentiment": f"Sentiment analysis for: {text[:50]}..."}

# List of available tools
AVAILABLE_TOOLS = [
    {
        "name": "create_folder",
        "description": "Create a new folder at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path where the folder should be created"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file at the specified path with optional content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path where the file should be created"},
                "content": {"type": "string", "description": "The initial content of the file (optional)"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_to_file",
        "description": "Write content to a file at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the file to write to"},
                "content": {"type": "string", "description": "The full content to write to the file"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the file to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files and directories in the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the folder to list (optional, defaults to current directory)"}
            },
            "required": []
        }
    },
    {
        "name": "delete_file",
        "description": "Delete a file at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the file to delete"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "search",
        "description": f"Perform a web search using the {SEARCH_PROVIDER} search provider.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "upload_file",
        "description": "Upload a file to the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "The file to upload (base64 encoded)"}
            },
            "required": ["file"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a specified city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_news",
        "description": "Get the latest news on a specified topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The news topic"}
            },
            "required": ["topic"]
        }
    },
    {
        "name": "translate_text",
        "description": "Translate text to a target language.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate"},
                "target_language": {"type": "string", "description": "The target language code (e.g., 'es' for Spanish)"}
            },
            "required": ["text", "target_language"]
        }
    },
    {
        "name": "summarize_text",
        "description": "Summarize a given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to summarize"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "analyze_sentiment",
        "description": "Analyze the sentiment of a given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze"}
            },
            "required": ["text"]
        }
    }
]

def get_tool_by_name(name: str) -> Dict[str, Any]:
    for tool in AVAILABLE_TOOLS:
        if tool['name'] == name:
            return tool
    return None

def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    tool = getattr(AITools, tool_name, None)
    if tool is None:
        return {"success": False, "error": f"Tool '{tool_name}' not found"}
    
    try:
        result = tool(**kwargs)
        return result
    except Exception as e:
        log.error(f"Error executing tool '{tool_name}': {str(e)}")
        return {"success": False, "error": str(e)}