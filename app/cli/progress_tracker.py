"""
Progress tracking for model generation with real-time updates.
"""

import asyncio
import time
from datetime import datetime
from typing import Callable, Optional, Dict, Any
from threading import Event, Thread

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.text import Text


class GenerationProgressTracker:
    """Tracks and displays real-time progress during model generation."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = None
        self.task_id = None
        self.start_time = None
        self.is_active = False
        self.update_callback: Optional[Callable] = None
        self.generation_stats = {}
        
    def start_generation(
        self, 
        description: str = "Generating response...",
        total_tokens: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> None:
        """Start tracking generation progress."""
        self.start_time = time.time()
        self.is_active = True
        self.update_callback = callback
        self.generation_stats = {
            "tokens_generated": 0,
            "tokens_per_second": 0.0,
            "estimated_total": total_tokens or 1024,
            "stage": "initializing"
        }
        
        # Create progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        )
        
        self.task_id = self.progress.add_task(
            description, 
            total=self.generation_stats["estimated_total"]
        )
        
        # Start the progress display
        self.progress.start()
        
        # Start background update task
        self._start_background_updates()
    
    def update_progress(
        self, 
        tokens_generated: int, 
        stage: str = "generating",
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update generation progress."""
        if not self.is_active or not self.progress or self.task_id is None:
            return
        
        elapsed = time.time() - self.start_time
        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0
        
        self.generation_stats.update({
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "stage": stage,
            "elapsed_time": elapsed
        })
        
        if additional_info:
            self.generation_stats.update(additional_info)
        
        # Update progress bar
        progress_percentage = min(100, (tokens_generated / self.generation_stats["estimated_total"]) * 100)
        
        description = f"{stage.title()} - {tokens_generated} tokens ({tokens_per_second:.1f} tok/s)"
        
        self.progress.update(
            self.task_id,
            completed=tokens_generated,
            description=description
        )
        
        # Call update callback if provided
        if self.update_callback:
            self.update_callback(self.generation_stats)
    
    def finish_generation(self, final_tokens: int, response_text: str = "") -> Dict[str, Any]:
        """Finish tracking and return final statistics."""
        if not self.is_active:
            return {}
        
        self.is_active = False
        elapsed = time.time() - self.start_time
        tokens_per_second = final_tokens / elapsed if elapsed > 0 else 0
        
        final_stats = {
            "total_tokens": final_tokens,
            "total_time": elapsed,
            "average_tokens_per_second": tokens_per_second,
            "response_length": len(response_text),
            "words_generated": len(response_text.split()) if response_text else 0,
            "completion_time": datetime.now().isoformat()
        }
        
        # Update final progress
        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id,
                completed=final_tokens,
                description=f"Complete - {final_tokens} tokens ({tokens_per_second:.1f} tok/s)"
            )
            
            # Stop progress display after a brief pause
            asyncio.create_task(self._stop_progress_delayed())
        
        return final_stats
    
    def cancel_generation(self) -> None:
        """Cancel progress tracking."""
        self.is_active = False
        if self.progress:
            self.progress.stop()
            self.progress = None
    
    async def _stop_progress_delayed(self) -> None:
        """Stop progress display after a delay."""
        await asyncio.sleep(2.0)  # Show completion for 2 seconds
        if self.progress:
            self.progress.stop()
            self.progress = None
    
    def _start_background_updates(self) -> None:
        """Start background updates for elapsed time."""
        if not self.is_active:
            return
        
        # Schedule next update
        asyncio.create_task(self._background_update())
    
    async def _background_update(self) -> None:
        """Background update task."""
        while self.is_active:
            await asyncio.sleep(0.5)  # Update every 500ms
            
            if not self.is_active or not self.progress or self.task_id is None:
                break
            
            # Update elapsed time display
            elapsed = time.time() - self.start_time
            current_tokens = self.generation_stats.get("tokens_generated", 0)
            tokens_per_second = current_tokens / elapsed if elapsed > 0 else 0
            
            if current_tokens > 0:
                description = f"{self.generation_stats.get('stage', 'generating').title()} - {current_tokens} tokens ({tokens_per_second:.1f} tok/s)"
                self.progress.update(self.task_id, description=description)


class StreamingProgressTracker:
    """Enhanced progress tracker for streaming generation."""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.start_time = None
        self.tokens_generated = 0
        self.current_text = ""
        self.is_active = False
        
    def start_streaming(self) -> None:
        """Start streaming progress tracking."""
        self.start_time = time.time()
        self.tokens_generated = 0
        self.current_text = ""
        self.is_active = True
    
    def add_token(self, token_text: str) -> None:
        """Add a new token to the stream."""
        if not self.is_active:
            return
        
        self.tokens_generated += 1
        self.current_text += token_text
        
        elapsed = time.time() - self.start_time
        tokens_per_second = self.tokens_generated / elapsed if elapsed > 0 else 0
        
        progress_info = {
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": tokens_per_second,
            "current_text": self.current_text,
            "elapsed_time": elapsed,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.callback:
            self.callback(progress_info)
    
    def finish_streaming(self) -> Dict[str, Any]:
        """Finish streaming and return final stats."""
        if not self.is_active:
            return {}
        
        self.is_active = False
        elapsed = time.time() - self.start_time
        
        return {
            "total_tokens": self.tokens_generated,
            "total_time": elapsed,
            "average_tokens_per_second": self.tokens_generated / elapsed if elapsed > 0 else 0,
            "final_text": self.current_text,
            "word_count": len(self.current_text.split()),
            "character_count": len(self.current_text)
        }


def create_progress_display(title: str = "Generation Progress") -> Text:
    """Create a rich text display for progress information."""
    text = Text()
    text.append(f"🔄 {title}\n", style="bold cyan")
    text.append("━" * 50 + "\n", style="dim white")
    return text


def format_generation_stats(stats: Dict[str, Any]) -> Text:
    """Format generation statistics as rich text."""
    text = Text()
    
    # Header
    text.append("📊 Generation Statistics\n", style="bold green")
    text.append("━" * 30 + "\n", style="dim white")
    
    # Statistics
    if "tokens_generated" in stats:
        text.append(f"Tokens: {stats['tokens_generated']}\n", style="cyan")
    
    if "tokens_per_second" in stats:
        text.append(f"Speed: {stats['tokens_per_second']:.1f} tok/s\n", style="yellow")
    
    if "elapsed_time" in stats:
        text.append(f"Time: {stats['elapsed_time']:.1f}s\n", style="white")
    
    if "stage" in stats:
        text.append(f"Stage: {stats['stage'].title()}\n", style="magenta")
    
    return text 