"""
Global logger for the memory system.
Only prints messages when debug mode is enabled.
"""


class Logger:
    """Simple logger that only outputs when debug mode is enabled."""
    
    _debug_mode: bool = False
    
    @classmethod
    def set_debug(cls, enabled: bool):
        """
        Enable or disable debug mode.
        
        Args:
            enabled: True to enable debug messages, False to disable
        """
        cls._debug_mode = enabled
    
    @classmethod
    def is_debug(cls) -> bool:
        """
        Check if debug mode is enabled.
        
        Returns:
            True if debug mode is enabled, False otherwise
        """
        return cls._debug_mode
    
    @classmethod
    def debug(cls, message: str, prefix: str = ""):
        """
        Print a debug message if debug mode is enabled.
        
        Args:
            message: Message to print
            prefix: Optional prefix (e.g., "[MemoryAPI]")
        """
        if cls._debug_mode:
            if prefix:
                print(f"{prefix} {message}")
            else:
                print(message)
    
    @classmethod
    def info(cls, message: str, prefix: str = ""):
        """
        Print an info message (always shown, regardless of debug mode).
        
        Args:
            message: Message to print
            prefix: Optional prefix (e.g., "[MemoryAPI]")
        """
        if prefix:
            print(f"{prefix} {message}")
        else:
            print(message)
    
    @classmethod
    def error(cls, message: str, prefix: str = ""):
        """
        Print an error message (always shown, regardless of debug mode).
        
        Args:
            message: Message to print
            prefix: Optional prefix (e.g., "[MemoryAPI]")
        """
        if prefix:
            print(f"{prefix} ✗ {message}")
        else:
            print(f"✗ {message}")
    
    @classmethod
    def success(cls, message: str, prefix: str = ""):
        """
        Print a success message (always shown, regardless of debug mode).
        
        Args:
            message: Message to print
            prefix: Optional prefix (e.g., "[MemoryAPI]")
        """
        if prefix:
            print(f"{prefix} ✓ {message}")
        else:
            print(f"✓ {message}")
    
    @classmethod
    def warning(cls, message: str, prefix: str = ""):
        """
        Print a warning message (always shown, regardless of debug mode).
        
        Args:
            message: Message to print
            prefix: Optional prefix (e.g., "[MemoryAPI]")
        """
        if prefix:
            print(f"{prefix} ⚠ {message}")
        else:
            print(f"⚠ {message}")
