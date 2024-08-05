import os
# import base64
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


class AITools:
    @staticmethod
    async def create_folder(path: str) -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            os.makedirs(full_path, exist_ok=True)
            log.info(f"Created folder: {full_path}")
            return {"success": True, "message": f"Folder created at {full_path}"}
        except Exception as e:
            log.error(f"Error creating folder: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def create_file(path: str, content: str = "") -> Dict[str, Any]:
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
    async def write_to_file(path: str, content: str) -> Dict[str, Any]:
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
    async def read_file(path: str) -> Dict[str, Any]:
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
    async def list_files(path: str = ".") -> Dict[str, Any]:
        full_path = os.path.abspath(path)
        try:
            files = os.listdir(full_path)
            log.info(f"Listed files in: {full_path}")
            return {"success": True, "files": files}
        except Exception as e:
            log.error(f"Error listing files: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def delete_file(path: str) -> Dict[str, Any]:
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
    }
]

def get_tool_by_name(name: str) -> Dict[str, Any]:
    for tool in AVAILABLE_TOOLS:
        if tool['name'] == name:
            return tool
    return None

async def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    tool = getattr(AITools, tool_name, None)
    if tool is None:
        return {"success": False, "error": f"Tool '{tool_name}' not found"}
    
    try:
        result = await tool(**kwargs)
        return result
    except Exception as e:
        log.error(f"Error executing tool '{tool_name}': {str(e)}")
        return {"success": False, "error": str(e)}