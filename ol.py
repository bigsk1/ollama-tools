import os
import json
import asyncio
import signal
import traceback
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from typing import List, Dict, Any
from urllib.parse import urlparse
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from rich.text import Text
from rich.markup import escape
from textwrap import wrap
from dotenv import load_dotenv
from tools import AVAILABLE_TOOLS, execute_tool
from search_utils import SEARCH_PROVIDER
from db_utils import retrieve_context, add_to_vector_db

# Load environment variables
load_dotenv()

conversation_history = []
should_exit = False

# Setup rich console for beautiful terminal output
console = Console()

# Constants
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3-groq-tool-use")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Disable unwanted logging
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)  # Suppress ChromaDB warnings

def signal_handler(sig, frame):
    global should_exit
    should_exit = True
    console.print("\n[bold yellow]Gracefully shutting down...[/bold yellow]")

signal.signal(signal.SIGINT, signal_handler)

def create_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
    )

def print_debug(message):
    if DEBUG_MODE:
        console.print(f"[dim cyan]DEBUG: {message}[/dim cyan]")

def print_info(message):
    console.print(f"[bold blue]INFO: {message}[/bold blue]")

def print_warning(message):
    console.print(f"[bold yellow]WARNING: {message}[/bold yellow]")

async def ollama_chat(llm: ChatOllama, prompt: str, tools: List[Dict[str, Any]]) -> AIMessage:
    global conversation_history
    
    # Retrieve context from the database
    contexts = retrieve_context(prompt)
    
    if contexts:
        print_info("Retrieved relevant contexts:")
        for idx, context in enumerate(contexts, 1):
            console.print(f"  Context {idx} (similarity: {context['similarity']:.4f}):")
            console.print(f"    Prompt: {context['prompt']}")
            console.print(f"    Response: {context['response'][:50]}...")  # Truncate long responses
    else:
        print_info("No relevant contexts found.")
    
    # Format tools for the model
    tools_string = "<tools>\n" + "\n".join([json.dumps(tool["function"]) for tool in tools]) + "\n</tools>"
    
    # Prepare context information
    context_info = "\n".join([
        f"Context {idx + 1} (similarity: {context['similarity']:.4f}):\n"
        f"Prompt: {context['prompt']}\n"
        f"Response: {context['response']}\n"
        for idx, context in enumerate(contexts)
    ])
    
    system_message = f"""You are a helpful AI assistant with access to previous conversation contexts and various tools. 
Your responses should be informative, engaging, and tailored to the user's needs. 
It is CRUCIAL that you carefully review and utilize the information from the provided contexts in your responses.
The contexts are sorted by relevance, with the most relevant context listed first.
Always prefer information from these contexts over making assumptions or using general knowledge.

Here are the relevant contexts from previous conversations:
{context_info}

You MUST use this context information to inform your responses, especially for questions about personal information or previous interactions.

You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. 
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": <function-name>,"arguments": <args-dict>}}
</tool_call>

Here are the available tools:
{tools_string}

Remember, always prioritize the information from the provided contexts in your responses.
"""
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history,
        {"role": "user", "content": prompt}
    ]
    
    print_info("Sending request to Ollama")
    try:
        response = await llm.ainvoke(messages)
        
        # Process the response
        if isinstance(response, AIMessage):
            content = response.content
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": content})
            
            # Add the new interaction to the vector database
            print_info("Adding new interaction to vector DB")
            add_to_vector_db({
                "prompt": prompt,
                "response": content
            })
        else:
            console.print(Panel(f"[bold red]Error:[/bold red] Unexpected response type: {type(response)}", border_style="red"))
            content = "I'm sorry, I encountered an error and couldn't process your request."
        
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        return AIMessage(content=content)
    except Exception as e:
        print_warning(f"Error in ollama_chat: {str(e)}")
        if DEBUG_MODE:
            print_debug(f"Exception details: {traceback.format_exc()}")
        return AIMessage(content="I'm sorry, I encountered an error and couldn't process your request.")

async def process_tool_calls(content: str) -> str:
    while "<tool_call>" in content:
        tool_call_start = content.index("<tool_call>")
        tool_call_end = content.index("</tool_call>", tool_call_start)
        tool_call = content[tool_call_start + 11:tool_call_end].strip()
        
        try:
            tool_data = json.loads(tool_call)
            tool_name = tool_data["name"]
            arguments = tool_data["arguments"]
            
            print_info(f"Using tool: {tool_name}")
            
            if tool_name == "search":
                print_info(f"Search provider: {SEARCH_PROVIDER}")
            
            result = await execute_tool(tool_name, **arguments)

            if result["success"]:
                print_info(f"Tool executed successfully.")
                if tool_name == "search":
                    if result["results"]:
                        format_search_results(result["results"])
                    else:
                        print_info("No search results found.")
            else:
                print_warning(f"Error executing tool: {result.get('error', 'Unknown error')}")
            
            # Remove the tool call and response from the content
            tool_response_start = content.index("<tool_response>", tool_call_end)
            tool_response_end = content.index("</tool_response>", tool_response_start)
            content = content[:tool_call_start] + content[tool_response_end + 16:]
        
        except ValueError as e:
            if "substring not found" in str(e):
                # If <tool_response> is not found, just remove the tool call
                content = content[:tool_call_start] + content[tool_call_end + 12:]
            else:
                print_warning(f"Error: {str(e)}")
                break
        except Exception as e:
            print_warning(f"Error: {str(e)}")
            break
    
    return content

async def chat_loop():
    console.print(Panel(
        "[bold blue]Welcome to the Integrated Context-Aware Ollama AI Assistant![/bold blue]\n"
        f"[green]Using model: {OLLAMA_MODEL}[/green]\n"
        "[yellow]Type 'exit', 'quit', or 'bye' to end the conversation.[/yellow]",
        title="AI Assistant",
        border_style="cyan"
    ))
    
    llm = create_llm()
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool["input_schema"]["properties"],
                    "required": tool["input_schema"].get("required", [])
                }
            }
        } for tool in AVAILABLE_TOOLS
    ]
    
    while not should_exit:
        try:
            user_input = Prompt.ask("\n[bold green]You")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            print_debug(f"Received user input: {user_input}")
            
            console.print("\n[bold yellow]AI Assistant[/bold yellow]")
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response = await ollama_chat(llm, user_input, tools)
                
                # Process the response content
                content = await process_tool_calls(response.content)
                
                # Display the final response
                console.print("\n")
                console.print(Markdown(content))
                console.print("\n")

            print_debug("Finished processing user input")
        except Exception as e:
            print_warning(f"An error occurred during chat: {str(e)}")
            if DEBUG_MODE:
                print_debug(f"Exception details: {traceback.format_exc()}")


def format_search_results(results):
    table = Table(title="Search Results", show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Title", style="cyan", width=30, overflow="fold")
    table.add_column("Domain", style="blue", width=30, overflow="fold")
    table.add_column("Snippet", style="green", width=60, overflow="fold")

    if not results:
        table.add_row("No results found", "", "")
    else:
        for result in results:
            title = result.get("title", "N/A")
            url = result.get("url", "N/A")
            snippet = result.get("snippet", "N/A")
            
            # Extract domain from URL
            domain = urlparse(url).netloc
            
            # Wrap the snippet text
            wrapped_snippet = "\n".join(wrap(snippet, width=58))
            
            # Escape any Rich markup characters in the URL
            escaped_url = escape(url)
            
            table.add_row(
                Text(title, style="cyan"),
                Text(f"[link={escaped_url}]{domain}[/link]", style="blue"),
                Text(wrapped_snippet, style="green")
            )

    console.print(table)
    
    # Print full URLs below the table
    console.print("\n[bold]Full URLs:[/bold]")
    for i, result in enumerate(results, 1):
        url = result.get("url", "N/A")
        escaped_url = escape(url)
        console.print(f"{i}. [link={escaped_url}]{escaped_url}[/link]")
    

if __name__ == "__main__":
    print_debug("Script started")
    try:
        asyncio.run(chat_loop())
    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
        if DEBUG_MODE:
            print_debug(f"Exception details: {traceback.format_exc()}")
    finally:
        console.print("\n[bold green]Thank you for using the Integrated Context-Aware Ollama AI Assistant. Goodbye![/bold green]")
    print_debug("Script ended")