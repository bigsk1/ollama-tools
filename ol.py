# ollama.py
import os
import json
import asyncio
import signal
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


# Load environment variables
load_dotenv()

conversation_history = []
should_exit = False

# Setup rich console for beautiful terminal output
console = Console()

# Constants
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

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


async def ollama_chat(llm: ChatOllama, prompt: str, tools: List[Dict[str, Any]]) -> AIMessage:
    global conversation_history
    
    # Format tools for the model
    tools_string = "<tools>\n" + "\n".join([json.dumps(tool["function"]) for tool in tools]) + "\n</tools>"
    
    system_message = f"""You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": <function-name>,"arguments": <args-dict>}}
</tool_call>

Here are the available tools:
{tools_string}
"""
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history,
        {"role": "user", "content": prompt}
    ]
    
    response = await llm.ainvoke(messages)
    
    # Process the response
    if isinstance(response, AIMessage):
        content = response.content
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": content})
    else:
        console.print(Panel(f"[bold red]Error:[/bold red] Unexpected response type: {type(response)}", border_style="red"))
        content = "I'm sorry, I encountered an error and couldn't process your request."
    
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    return AIMessage(content=content)

async def chat_loop():
    console.print(Panel(
        "[bold blue]Welcome to the Ollama AI Assistant![/bold blue]\n"
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
        user_input = Prompt.ask("\n[bold green]You")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        console.print("\n[bold yellow]AI Assistant[/bold yellow]")
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            response = await ollama_chat(llm, user_input, tools)
            
            # Process the response content
            content = response.content
            while "<tool_call>" in content:
                tool_call_start = content.index("<tool_call>")
                tool_call_end = content.index("</tool_call>", tool_call_start)
                tool_call = content[tool_call_start + 11:tool_call_end].strip()
                
                try:
                    tool_data = json.loads(tool_call)
                    tool_name = tool_data["name"]
                    arguments = tool_data["arguments"]
                    
                    console.print(f"[bold blue]Using tool:[/bold blue] {tool_name}")
                    
                    if tool_name == "search":
                        console.print(f"[bold cyan]Search provider:[/bold cyan] {SEARCH_PROVIDER}")
                    
                    result = await execute_tool(tool_name, **arguments)

                    if result["success"]:
                        console.print(f"[bold green]Tool executed successfully.[/bold green]")
                        if tool_name == "search":
                            console.print(f"[bold cyan]Search provider:[/bold cyan] {SEARCH_PROVIDER}")
                            if result["results"]:
                                format_search_results(result["results"])
                            else:
                                console.print("[yellow]No search results found.[/yellow]")
                    else:
                        console.print(f"[bold red]Error executing tool:[/bold red] {result.get('error', 'Unknown error')}")
                    
                    # Remove the tool call and response from the content
                    tool_response_start = content.index("<tool_response>", tool_call_end)
                    tool_response_end = content.index("</tool_response>", tool_response_start)
                    content = content[:tool_call_start] + content[tool_response_end + 16:]
                
                except ValueError as e:
                    if "substring not found" in str(e):
                        # If <tool_response> is not found, just remove the tool call
                        content = content[:tool_call_start] + content[tool_call_end + 12:]
                    else:
                        console.print(Panel(f"[bold red]Error:[/bold red] {str(e)}", border_style="red"))
                        break
                except Exception as e:
                    console.print(Panel(f"[bold red]Error:[/bold red] {str(e)}", border_style="red"))
                    break
            
            # Display the final response
            console.print(Markdown(content))


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
    try:
        asyncio.run(chat_loop())
    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
    finally:
        console.print("\n[bold green]Thank you for using the Ollama AI Assistant. Goodbye![/bold green]")