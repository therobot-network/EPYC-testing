"""
Main CLI entry point for the EPYC-testing TUI interface.
"""

import asyncio
import click
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .tui import EPYCTestingTUI
from ..config.settings import get_settings


console = Console()


def show_banner():
    """Display the application banner."""
    banner_text = Text()
    banner_text.append("EPYC", style="bold red")
    banner_text.append("-", style="white")
    banner_text.append("TESTING", style="bold blue")
    banner_text.append(" TUI", style="bold green")
    
    subtitle = Text("LLaMA 3.3 70B Terminal Interface", style="italic cyan")
    
    panel = Panel.fit(
        Text.assemble(banner_text, "\n", subtitle),
        title="🦙 Welcome",
        border_style="bright_blue",
        padding=(1, 2)
    )
    
    console.print(panel)
    console.print()


@click.command()
@click.option(
    "--model-path", 
    "-m", 
    type=click.Path(exists=True, path_type=Path),
    help="Path to the LLaMA model directory"
)
@click.option(
    "--model-name",
    "-n",
    default="llama33",
    help="Name to assign to the loaded model"
)
@click.option(
    "--auto-load",
    "-a",
    is_flag=True,
    help="Automatically load the model on startup"
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Enable debug mode"
)
def main(
    model_path: Optional[Path] = None,
    model_name: str = "llama33",
    auto_load: bool = False,
    debug: bool = False
):
    """
    Launch the EPYC-testing TUI for LLaMA 3.3 70B model interaction.
    
    This terminal user interface provides:
    - Beautiful loading screens with progress tracking
    - Real-time chat interface with the model
    - System monitoring and performance metrics
    - Timestamped conversation history
    """
    show_banner()
    
    # Get settings and determine model path
    settings = get_settings()
    
    if not model_path:
        # Try to find model in default locations
        default_paths = [
            Path("./models/llama-3.3-70b-instruct"),
            Path(settings.model_path) / "llama-3.3-70b-instruct",
            Path("/home/ubuntu/EPYC-testing/models/llama-3.3-70b-instruct"),  # EC2 path
        ]
        
        for path in default_paths:
            if path.exists() and path.is_dir():
                model_path = path
                break
        
        if not model_path:
            console.print(
                Panel.fit(
                    "[red]❌ No model found![/red]\n\n"
                    "Please specify a model path with --model-path or ensure the model is in one of these locations:\n"
                    + "\n".join(f"• {path}" for path in default_paths),
                    title="Model Not Found",
                    border_style="red"
                )
            )
            return
    
    console.print(f"[green]✓[/green] Model found at: [cyan]{model_path}[/cyan]")
    console.print(f"[green]✓[/green] Model name: [cyan]{model_name}[/cyan]")
    
    if debug:
        console.print(f"[yellow]⚠[/yellow] Debug mode enabled")
    
    console.print()
    
    # Launch the TUI
    try:
        tui = EPYCTestingTUI(
            model_path=str(model_path),
            model_name=model_name,
            auto_load=auto_load,
            debug=debug
        )
        tui.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        if debug:
            console.print_exception()


if __name__ == "__main__":
    main() 