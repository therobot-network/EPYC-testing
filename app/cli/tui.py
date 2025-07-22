"""
TUI (Terminal User Interface) for EPYC-testing using Textual.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import psutil
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, ProgressBar, Label, Input, Button, 
    RichLog, Collapsible, DataTable, TabbedContent, TabPane
)
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.console import Console

from ..models.manager import ModelManager
from ..config.settings import get_settings
from .progress_tracker import StreamingProgressTracker, format_generation_stats


class LoadingScreen(Container):
    """Loading screen with progress bars and system information."""
    
    def __init__(self, model_path: str, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model_name = model_name
        self.loading_progress = 0.0
        self.loading_stage = "Initializing..."
        self.start_time = time.time()
        
    def compose(self) -> ComposeResult:
        """Compose the loading screen."""
        yield Static(self._create_header(), id="loading-header")
        yield Container(
            Static(id="system-info"),
            Static(id="loading-progress"),
            Static(id="loading-logs"),
            id="loading-content"
        )
        
    def on_mount(self) -> None:
        """Initialize loading screen."""
        self.update_system_info()
        self.start_loading_animation()
        
    def _create_header(self) -> Panel:
        """Create the loading header."""
        header_text = Text()
        header_text.append("🦙 Loading LLaMA 3.3 70B", style="bold cyan")
        header_text.append(f"\n📁 {self.model_path}", style="dim white")
        header_text.append(f"\n🏷️  {self.model_name}", style="dim white")
        
        return Panel.fit(
            header_text,
            title="Model Loading",
            border_style="bright_blue",
            padding=(1, 2)
        )
    
    def update_system_info(self) -> None:
        """Update system information display."""
        # Get system information
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Create system info table
        table = Table(title="System Information", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_column("Details", style="dim white")
        
        table.add_row("CPU Cores", str(cpu_count), f"Physical cores available")
        table.add_row("Memory", f"{memory.total // (1024**3)} GB", f"{memory.percent}% used")
        table.add_row("Disk Space", f"{disk.total // (1024**3)} GB", f"{disk.percent}% used")
        table.add_row("Model Size", "~140 GB", "Estimated memory usage")
        
        # Get EC2 specific info if available
        settings = get_settings()
        if hasattr(settings, 'ec2_instance_type') and settings.ec2_instance_type:
            table.add_row("Instance", settings.ec2_instance_type, "AWS EC2 instance")
            table.add_row("Region", getattr(settings, 'aws_region', 'N/A'), "AWS region")
        
        system_info_widget = self.query_one("#system-info", Static)
        system_info_widget.update(table)
    
    def update_progress(self, progress: float, stage: str, details: str = "") -> None:
        """Update loading progress."""
        self.loading_progress = progress
        self.loading_stage = stage
        
        # Create progress display
        elapsed = time.time() - self.start_time
        
        progress_text = Text()
        progress_text.append(f"Stage: {stage}\n", style="bold yellow")
        progress_text.append(f"Progress: {progress:.1f}%\n", style="cyan")
        progress_text.append(f"Elapsed: {elapsed:.1f}s\n", style="dim white")
        if details:
            progress_text.append(f"Details: {details}", style="dim cyan")
        
        # Create progress bar
        bar_width = 40
        filled = int(progress / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        progress_text.append(f"\n[{bar}] {progress:.1f}%", style="bright_green")
        
        progress_panel = Panel.fit(
            progress_text,
            title="Loading Progress",
            border_style="green",
            padding=(1, 1)
        )
        
        progress_widget = self.query_one("#loading-progress", Static)
        progress_widget.update(progress_panel)
    
    def add_log(self, message: str, level: str = "info") -> None:
        """Add a log message to the loading screen."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        style_map = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red"
        }
        style = style_map.get(level, "white")
        
        log_text = Text()
        log_text.append(f"[{timestamp}] ", style="dim white")
        log_text.append(message, style=style)
        
        logs_widget = self.query_one("#loading-logs", Static)
        current_content = logs_widget.renderable or Text()
        if isinstance(current_content, Text):
            current_content.append("\n")
            current_content.append(log_text)
        else:
            current_content = Text.assemble(current_content, "\n", log_text)
        
        logs_widget.update(current_content)
    
    def start_loading_animation(self) -> None:
        """Start the loading animation."""
        self.animation_step = 0
        self.set_timer(0.5, self._animate_loading)
    
    def _animate_loading(self) -> None:
        """Animate loading indicators."""
        if not hasattr(self, 'animation_step'):
            self.animation_step = 0
        
        # Create animated spinner
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = spinner_chars[self.animation_step % len(spinner_chars)]
        
        # Update progress display with spinner
        elapsed = time.time() - self.start_time
        
        progress_text = Text()
        progress_text.append(f"{spinner} Stage: {self.loading_stage}\n", style="bold yellow")
        progress_text.append(f"Progress: {self.loading_progress:.1f}%\n", style="cyan")
        progress_text.append(f"Elapsed: {elapsed:.1f}s\n", style="dim white")
        progress_text.append("Press ESC to cancel loading\n", style="dim red")
        
        # Create progress bar
        bar_width = 40
        filled = int(self.loading_progress / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        progress_text.append(f"\n[{bar}] {self.loading_progress:.1f}%", style="bright_green")
        
        progress_panel = Panel.fit(
            progress_text,
            title="Loading Progress",
            border_style="green",
            padding=(1, 1)
        )
        
        try:
            progress_widget = self.query_one("#loading-progress", Static)
            progress_widget.update(progress_panel)
        except Exception:
            pass  # Widget might not exist yet
        
        self.animation_step += 1
        
        # Continue animation
        self.set_timer(0.5, self._animate_loading)


class ChatMessage(Container):
    """A single chat message widget."""
    
    def __init__(self, role: str, content: str, timestamp: datetime, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.timestamp = timestamp
        
    def compose(self) -> ComposeResult:
        """Compose the chat message."""
        timestamp_str = self.timestamp.strftime("%H:%M:%S")
        
        # Style based on role
        if self.role == "user":
            style = "blue"
            icon = "👤"
            title = "You"
        elif self.role == "assistant":
            style = "green"
            icon = "🦙"
            title = "LLaMA 3.3"
        else:
            style = "yellow"
            icon = "ℹ️"
            title = "System"
        
        message_text = Text()
        message_text.append(f"{icon} {title} ", style=f"bold {style}")
        message_text.append(f"({timestamp_str})\n", style="dim white")
        message_text.append(self.content, style="white")
        
        yield Static(
            Panel.fit(
                message_text,
                border_style=style,
                padding=(0, 1)
            )
        )


class ChatInterface(Container):
    """Chat interface for interacting with the model."""
    
    def __init__(self, model_manager: ModelManager, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_manager = model_manager
        self.model_name = model_name
        self.messages: List[Dict[str, Any]] = []
        self.is_generating = False
        self.progress_tracker = StreamingProgressTracker(callback=self._on_progress_update)
        self.current_response = ""
        
    def compose(self) -> ComposeResult:
        """Compose the chat interface."""
        yield Container(
            ScrollableContainer(id="chat-messages"),
            Horizontal(
                Input(placeholder="Type your message here...", id="message-input"),
                Button("Send", variant="primary", id="send-button"),
                id="input-area"
            ),
            Static(id="generation-status"),
            id="chat-container"
        )
    
    def on_mount(self) -> None:
        """Initialize chat interface."""
        self.add_system_message("Chat interface ready. Start typing to chat with LLaMA 3.3!")
        
    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.add_message("system", content)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat."""
        timestamp = datetime.now()
        message_data = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.messages.append(message_data)
        
        # Add to UI
        chat_messages = self.query_one("#chat-messages", ScrollableContainer)
        chat_messages.mount(ChatMessage(role, content, timestamp))
        
        # Scroll to bottom
        chat_messages.scroll_end()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "send-button":
            await self.send_message()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "message-input":
            await self.send_message()
    
    async def send_message(self) -> None:
        """Send user message and get response."""
        if self.is_generating:
            return
            
        message_input = self.query_one("#message-input", Input)
        user_message = message_input.value.strip()
        
        if not user_message:
            return
        
        # Clear input
        message_input.value = ""
        
        # Add user message
        self.add_message("user", user_message)
        
        # Start generation
        self.is_generating = True
        self.update_generation_status("Generating response...", 0.0)
        
        try:
            # Prepare messages for the model
            model_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.messages
                if msg["role"] in ["user", "assistant"]
            ]
            
            # Generate response
            start_time = time.time()
            response, inference_time = await self.model_manager.predict(
                model_name=self.model_name,
                input_data=model_messages,
                parameters={
                    "max_new_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )
            
            # Add response
            self.add_message("assistant", response)
            
            # Update status
            self.update_generation_status(
                f"Response generated in {inference_time:.2f}s", 
                100.0
            )
            
        except Exception as e:
            self.add_message("system", f"Error generating response: {str(e)}")
            self.update_generation_status("Error occurred", 0.0)
        
        finally:
            self.is_generating = False
            # Clear status after a delay
            self.set_timer(3.0, lambda: self.update_generation_status("", 0.0))
    
    def update_generation_status(self, message: str, progress: float) -> None:
        """Update generation status."""
        if not message:
            status_text = ""
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            status_text = f"[{timestamp}] {message}"
            if progress > 0:
                status_text += f" ({progress:.1f}%)"
        
        status_widget = self.query_one("#generation-status", Static)
        status_widget.update(status_text)
    
    def _on_progress_update(self, progress_info: Dict[str, Any]) -> None:
        """Handle progress updates from the tracker."""
        tokens = progress_info.get("tokens_generated", 0)
        tokens_per_second = progress_info.get("tokens_per_second", 0.0)
        elapsed = progress_info.get("elapsed_time", 0.0)
        
        # Update status with detailed progress
        status_message = f"Generating... {tokens} tokens ({tokens_per_second:.1f} tok/s) - {elapsed:.1f}s"
        progress_percentage = min(100, (tokens / 1024) * 100)  # Assume max 1024 tokens
        
        self.update_generation_status(status_message, progress_percentage)


class EPYCTestingTUI(App):
    """Main TUI application."""
    
    CSS = """
    #loading-content {
        layout: vertical;
        height: 100%;
        padding: 1;
    }
    
    #system-info {
        height: auto;
        margin-bottom: 1;
    }
    
    #loading-progress {
        height: auto;
        margin-bottom: 1;
    }
    
    #loading-logs {
        height: 1fr;
    }
    
    #chat-container {
        layout: vertical;
        height: 100%;
    }
    
    #chat-messages {
        height: 1fr;
        border: solid $primary;
        margin-bottom: 1;
    }
    
    #input-area {
        height: auto;
        margin-bottom: 1;
    }
    
    #message-input {
        width: 1fr;
        margin-right: 1;
    }
    
    #generation-status {
        height: auto;
        color: $accent;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
        Binding("ctrl+r", "reload_model", "Reload Model"),
        Binding("escape", "cancel_loading", "Cancel Loading"),
    ]
    
    def __init__(
        self, 
        model_path: str, 
        model_name: str, 
        auto_load: bool = False,
        debug_mode: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model_name = model_name
        self.auto_load = auto_load
        self.debug_mode = debug_mode
        self.model_manager = ModelManager()
        self.model_loaded = False
        self.current_screen = "loading"
        self.loading_task = None
        self.loading_cancelled = False
        
    def compose(self) -> ComposeResult:
        """Compose the main application."""
        yield Header(show_clock=True)
        
        # Start with loading screen
        yield LoadingScreen(
            self.model_path, 
            self.model_name, 
            id="loading-screen"
        )
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the application."""
        self.title = "EPYC-testing TUI"
        self.sub_title = f"LLaMA 3.3 70B - {self.model_name}"
        
        if self.auto_load:
            # Use call_after_refresh to ensure UI is ready before starting loading
            self.call_after_refresh(self._start_loading_task)
    
    def _start_loading_task(self) -> None:
        """Start the model loading task after UI is initialized."""
        try:
            self.loading_task = self.load_model()
        except Exception as e:
            loading_screen = self.query_one("#loading-screen", LoadingScreen)
            loading_screen.add_log(f"Failed to start loading task: {str(e)}", "error")
            if self.debug_mode:
                import traceback
                loading_screen.add_log(f"Debug traceback: {traceback.format_exc()}", "error")
    
    @work(exclusive=True)
    async def load_model(self) -> None:
        """Load the model with progress updates and timeout handling."""
        loading_screen = self.query_one("#loading-screen", LoadingScreen)
        
        try:
            # Stage 1: Initialize model manager
            loading_screen.update_progress(5.0, "Initializing Model Manager")
            loading_screen.add_log("Initializing model manager...", "info")
            await asyncio.sleep(0.1)
            
            # Stage 2: Load tokenizer
            loading_screen.update_progress(15.0, "Loading Tokenizer")
            loading_screen.add_log("Loading tokenizer from model path...", "info")
            
            # Stage 3: Load model configuration
            loading_screen.update_progress(25.0, "Loading Model Configuration")
            loading_screen.add_log("Reading model configuration...", "info")
            
            # Stage 4: Load model weights (this is the heavy operation)
            loading_screen.update_progress(35.0, "Loading Model Weights", "This may take several minutes...")
            loading_screen.add_log("Loading 70B parameters... This will take time.", "warning")
            loading_screen.add_log("If this takes too long, try using quantization or restart with debug mode", "info")
            
            # Create a task for model loading with timeout
            model_loading_task = asyncio.create_task(
                self._load_model_with_progress(loading_screen)
            )
            
            # Wait for model loading with timeout (45 minutes for large models)
            try:
                await asyncio.wait_for(model_loading_task, timeout=2700.0)  # 45 minutes
            except asyncio.TimeoutError:
                loading_screen.add_log("Model loading timed out after 45 minutes", "error")
                loading_screen.add_log("This may indicate insufficient memory or slow disk I/O", "warning")
                loading_screen.add_log("Try using quantization to reduce memory usage", "info")
                loading_screen.add_log("Or restart with debug mode: ./cli.py --debug", "info")
                return
            except asyncio.CancelledError:
                loading_screen.add_log("Model loading was cancelled", "warning")
                return
            
            # Stage 5: Final setup
            loading_screen.update_progress(100.0, "Ready!")
            loading_screen.add_log("Model loaded successfully!", "success")
            await asyncio.sleep(1.0)
            
            self.model_loaded = True
            self.switch_to_chat()
            
        except Exception as e:
            loading_screen.add_log(f"Error loading model: {str(e)}", "error")
            loading_screen.add_log("Common solutions:", "info")
            loading_screen.add_log("1. Restart with quantization: edit configs/ec2.yaml", "info")
            loading_screen.add_log("2. Check available memory: free -h", "info")
            loading_screen.add_log("3. Try debug mode: ./cli.py --debug", "info")
            if self.debug_mode:
                import traceback
                loading_screen.add_log(f"Debug traceback: {traceback.format_exc()}", "error")
    
    async def _load_model_with_progress(self, loading_screen) -> None:
        """Load model with detailed progress tracking."""
        try:
            # Check if cancelled before starting
            if self.loading_cancelled:
                raise asyncio.CancelledError("Loading cancelled")
            
            # Check available memory before loading
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            loading_screen.add_log(f"Available memory: {available_gb:.1f}GB", "info")
            
            if available_gb < 140:
                loading_screen.add_log(f"Warning: Only {available_gb:.1f}GB available, model needs ~140GB", "warning")
                loading_screen.add_log("Consider using quantization or freeing up memory", "warning")
            
            # Create progress update callback
            def progress_callback(stage: str, progress: float):
                # Map internal progress to UI progress (35% to 95%)
                ui_progress = 35.0 + (progress * 0.6)  # Scale to 35-95%
                loading_screen.update_progress(ui_progress, stage)
                loading_screen.add_log(f"{stage}: {progress:.1f}%", "info")
            
            # Actually load the model with timeout protection
            loading_screen.update_progress(40.0, "Loading Model Weights")
            loading_screen.add_log("Starting model weight loading...", "info")
            
            # Wrap the model loading in a timeout
            try:
                await asyncio.wait_for(
                    self.model_manager.load_model(
                        model_name=self.model_name,
                        model_path=self.model_path,
                        model_type="llama33"
                    ),
                    timeout=2400.0  # 40 minutes for just the model loading
                )
                loading_screen.add_log("Model weights loaded successfully", "success")
            except asyncio.TimeoutError:
                loading_screen.add_log("Model weight loading timed out", "error")
                loading_screen.add_log("The model is too large for available resources", "error")
                raise RuntimeError("Model loading timeout - insufficient resources")
            
            # Check if cancelled after loading
            if self.loading_cancelled:
                raise asyncio.CancelledError("Loading cancelled")
            
            # Final optimization stage
            loading_screen.update_progress(95.0, "Optimizing for Hardware")
            loading_screen.add_log("Applying hardware optimizations...", "info")
            await asyncio.sleep(0.5)
            
        except asyncio.CancelledError:
            loading_screen.add_log("Model loading was cancelled", "warning")
            raise
        except Exception as e:
            loading_screen.add_log(f"Model loading failed: {str(e)}", "error")
            if "out of memory" in str(e).lower() or "insufficient" in str(e).lower():
                loading_screen.add_log("Try using quantization to reduce memory usage", "info")
                loading_screen.add_log("Edit configs/ec2.yaml and set load_in_8bit: true", "info")
            elif "timeout" in str(e).lower():
                loading_screen.add_log("Model loading is taking too long", "error")
                loading_screen.add_log("This usually indicates memory pressure", "info")
            loading_screen.add_log("Press ESC to cancel and try again with different settings", "info")
            raise
    
    def switch_to_chat(self) -> None:
        """Switch from loading screen to chat interface."""
        # Remove loading screen
        loading_screen = self.query_one("#loading-screen")
        loading_screen.remove()
        
        # Add chat interface
        chat_interface = ChatInterface(self.model_manager, self.model_name)
        self.mount(chat_interface)
        
        self.current_screen = "chat"
        self.sub_title = f"Chat - {self.model_name}"
    
    def action_clear_chat(self) -> None:
        """Clear the chat history."""
        if self.current_screen == "chat":
            try:
                chat_interface = self.query_one(ChatInterface)
                chat_messages = chat_interface.query_one("#chat-messages", ScrollableContainer)
                chat_messages.remove_children()
                chat_interface.messages = []
                chat_interface.add_system_message("Chat cleared. Start a new conversation!")
            except Exception:
                pass
    
    def action_reload_model(self) -> None:
        """Reload the model."""
        if self.model_loaded or self.current_screen == "loading":
            if self.loading_task:
                self.loading_task.cancel()
            self.loading_task = self.load_model()
    
    def action_cancel_loading(self) -> None:
        """Cancel model loading."""
        if self.current_screen == "loading" and self.loading_task:
            self.loading_cancelled = True
            self.loading_task.cancel()
            loading_screen = self.query_one("#loading-screen", LoadingScreen)
            loading_screen.add_log("Loading cancelled by user", "warning")
    
    def action_quit(self) -> None:
        """Quit the application."""
        if self.loading_task:
            self.loading_task.cancel()
        self.exit() 