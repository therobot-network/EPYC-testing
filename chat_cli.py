#!/usr/bin/env python3
"""
Simple CLI for interacting with EPYC-testing models.
No more curl commands - just simple chat!
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
import httpx

console = Console()

# Model shortcuts and their configurations
MODEL_SHORTCUTS = {
    "llama31": {
        "name": "llama31",
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "local_path": "./models/llama-3.1-8b-instruct",
        "description": "Llama 3.1 8B Instruct - Fast and efficient"
    },
    "llama2": {
        "name": "llama2",
        "path": "meta-llama/Llama-2-13b-hf",
        "local_path": "./models/llama-2-13b",
        "description": "Llama 2 13B - Reliable and well-tested"
    },
    "llama33": {
        "name": "llama33",
        "path": "meta-llama/Llama-3.3-70B-Instruct",
        "local_path": "./models/llama-3.3-70b-instruct",
        "description": "Llama 3.3 70B - Most capable (requires lots of VRAM)"
    }
}

class ChatCLI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.current_model = None
        self.conversation_history = []
        
    async def check_server(self) -> bool:
        """Check if the server is running."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/health", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False
    
    async def list_loaded_models(self) -> List[Dict]:
        """List currently loaded models."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/models")
                if response.status_code == 200:
                    return response.json()
                return []
        except Exception as e:
            console.print(f"[red]Error listing models: {e}[/red]")
            return []
    
    async def load_model(self, model_shortcut: str) -> bool:
        """Load a model using shortcut."""
        if model_shortcut not in MODEL_SHORTCUTS:
            console.print(f"[red]Unknown model shortcut: {model_shortcut}[/red]")
            console.print("Available models:", ", ".join(MODEL_SHORTCUTS.keys()))
            return False
        
        model_config = MODEL_SHORTCUTS[model_shortcut]
        
        # Try local path first, then HuggingFace path
        model_path = model_config["local_path"]
        if not Path(model_path).exists():
            model_path = model_config["path"]
        
        with console.status(f"[bold green]Loading {model_config['description']}..."):
            try:
                timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/v1/models/load",
                        json={
                            "model_name": model_config["name"],
                            "model_path": model_path,
                            "force_reload": False
                        }
                    )
                    
                    if response.status_code == 200:
                        self.current_model = model_config["name"]
                        console.print(f"[green]✓ {model_config['description']} loaded successfully![/green]")
                        return True
                    else:
                        error_data = response.json()
                        console.print(f"[red]Failed to load model: {error_data.get('detail', 'Unknown error')}[/red]")
                        return False
                        
            except Exception as e:
                error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
                console.print(f"[red]Error loading model: {error_msg}[/red]")
                return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(f"{self.base_url}/api/v1/models/{model_name}")
                if response.status_code == 200:
                    console.print(f"[green]✓ Model {model_name} unloaded[/green]")
                    if self.current_model == model_name:
                        self.current_model = None
                    return True
                else:
                    console.print(f"[red]Failed to unload model: {response.json().get('detail', 'Unknown error')}[/red]")
                    return False
        except Exception as e:
            console.print(f"[red]Error unloading model: {e}[/red]")
            return False
    
    async def chat(self, message: str, **kwargs) -> Optional[str]:
        """Send a chat message to the current model."""
        if not self.current_model:
            console.print("[red]No model loaded. Use 'load <model>' first.[/red]")
            return None
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat",
                    json={
                        "messages": self.conversation_history,
                        "model_name": self.current_model,
                        "max_new_tokens": kwargs.get("max_tokens", 1024),
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "top_k": kwargs.get("top_k", 50)
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result["response"]
                    
                    # Add assistant response to history
                    self.conversation_history.append({"role": "assistant", "content": assistant_response})
                    
                    return assistant_response
                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    console.print(f"[red]Chat error: {error_msg}[/red]")
                    return None
                    
        except httpx.TimeoutException:
            console.print("[red]Request timed out. The model might be taking too long to respond.[/red]")
            return None
        except httpx.ConnectError:
            console.print("[red]Connection error. Is the server running?[/red]")
            return None
        except asyncio.CancelledError:
            console.print("[yellow]Request was cancelled.[/yellow]")
            return None
        except Exception as e:
            # Escape any markup in the error message
            error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
            console.print(f"[red]Error during chat: {error_msg}[/red]")
            return None
    
    def show_models(self):
        """Show available model shortcuts."""
        table = Table(title="Available Models")
        table.add_column("Shortcut", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        
        for shortcut, config in MODEL_SHORTCUTS.items():
            # Check if local path exists
            local_exists = Path(config["local_path"]).exists()
            status = "✓ Local" if local_exists else "🌐 Remote"
            table.add_row(shortcut, config["description"], status)
        
        console.print(table)
    
    async def show_loaded_models(self):
        """Show currently loaded models."""
        models = await self.list_loaded_models()
        if not models:
            console.print("[yellow]No models currently loaded[/yellow]")
            return
        
        table = Table(title="Loaded Models")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Memory", style="yellow")
        table.add_column("Loaded At", style="white")
        
        for model in models:
            memory = f"{model.get('memory_usage', 0):.1f}MB" if model.get('memory_usage') else "N/A"
            loaded_at = model.get('loaded_at', 'Unknown')[:19].replace('T', ' ')
            table.add_row(
                model['name'],
                model['status'],
                memory,
                loaded_at
            )
        
        console.print(table)
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        console.print("[green]✓ Conversation history cleared[/green]")
    
    def show_help(self):
        """Show help message."""
        help_text = """
[bold cyan]EPYC-testing Chat CLI[/bold cyan]

[bold]Quick Commands:[/bold]
  [cyan]load <model>[/cyan]     - Load a model (llama31, llama2, llama33)
  [cyan]models[/cyan]           - Show available models
  [cyan]loaded[/cyan]           - Show loaded models
  [cyan]unload <model>[/cyan]   - Unload a model
  [cyan]clear[/cyan]            - Clear conversation history
  [cyan]help[/cyan]             - Show this help
  [cyan]quit[/cyan] or [cyan]exit[/cyan]  - Exit the CLI

[bold]Chat:[/bold]
  Just type your message and press Enter to chat with the loaded model.

[bold]Examples:[/bold]
  [dim]> load llama31[/dim]
  [dim]> Hello, how are you?[/dim]
  [dim]> What is quantum computing?[/dim]

[bold]Parameters:[/bold]
  You can adjust generation parameters by typing:
  [cyan]set temperature 0.8[/cyan]
  [cyan]set max_tokens 512[/cyan]
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))

async def main():
    parser = argparse.ArgumentParser(description="EPYC-testing Chat CLI")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--model", help="Auto-load model on startup")
    args = parser.parse_args()
    
    cli = ChatCLI(args.server)
    
    # Show banner
    console.print(Panel.fit(
        "[bold cyan]EPYC-testing Chat CLI[/bold cyan]\n"
        "[dim]Simple interface for chatting with your models[/dim]",
        border_style="cyan"
    ))
    
    # Check server connection
    if not await cli.check_server():
        console.print(f"[red]❌ Cannot connect to server at {args.server}[/red]")
        console.print("[yellow]Make sure the server is running with: python -m app.main[/yellow]")
        return
    
    console.print(f"[green]✓ Connected to server at {args.server}[/green]")
    
    # Auto-load model if specified
    if args.model:
        await cli.load_model(args.model)
    
    # Show initial help
    console.print("\n[dim]Type 'help' for commands or just start chatting![/dim]")
    
    # Generation parameters
    gen_params = {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 50
    }
    
    # Main loop
    while True:
        try:
            # Show current model in prompt
            model_indicator = f"[{cli.current_model}]" if cli.current_model else "[no model]"
            prompt_text = f"{model_indicator} > "
            
            user_input = Prompt.ask(prompt_text, default="").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif user_input.lower() == "help":
                cli.show_help()
            elif user_input.lower() == "models":
                cli.show_models()
            elif user_input.lower() == "loaded":
                await cli.show_loaded_models()
            elif user_input.lower() == "clear":
                cli.clear_conversation()
            elif user_input.lower().startswith("load "):
                model_name = user_input[5:].strip()
                await cli.load_model(model_name)
            elif user_input.lower().startswith("unload "):
                model_name = user_input[7:].strip()
                await cli.unload_model(model_name)
            elif user_input.lower().startswith("set "):
                # Handle parameter setting
                parts = user_input[4:].strip().split()
                if len(parts) == 2:
                    param, value = parts
                    try:
                        if param in ["temperature", "top_p"]:
                            gen_params[param] = float(value)
                        elif param in ["max_tokens", "top_k"]:
                            gen_params[param] = int(value)
                        console.print(f"[green]✓ Set {param} = {value}[/green]")
                    except ValueError:
                        console.print(f"[red]Invalid value for {param}: {value}[/red]")
                else:
                    console.print("[red]Usage: set <parameter> <value>[/red]")
            else:
                # Regular chat message
                if cli.current_model:
                    with Live(Spinner("dots", text="Thinking..."), refresh_per_second=4):
                        response = await cli.chat(user_input, **gen_params)
                    
                    if response:
                        console.print(f"\n[bold blue]Assistant:[/bold blue] {response}\n")
                else:
                    console.print("[red]No model loaded. Use 'load <model>' first.[/red]")
                    console.print("Available models: " + ", ".join(MODEL_SHORTCUTS.keys()))
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit properly[/yellow]")
        except Exception as e:
            # Escape any markup in the error message to prevent Rich formatting errors
            error_msg = str(e).replace('[', '\\[').replace(']', '\\]')
            console.print(f"[red]Unexpected error: {error_msg}[/red]")

if __name__ == "__main__":
    asyncio.run(main()) 