import sys
import importlib

def check_module_path(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            print(f"Module '{module_name}' found at: {spec.origin}")
            if module_name == 'moviepy.audio.fx.all':
                # Try importing it directly
                from moviepy.audio.fx.all import volumex
                print("Successfully imported 'volumex' from 'moviepy.audio.fx.all'")
        elif spec:
            print(f"Module '{module_name}' found, but origin is not available (e.g., built-in or namespace package).")
        else:
            print(f"Module '{module_name}' NOT found.")
    except ImportError as e:
        print(f"ImportError for '{module_name}': {e}")
    except Exception as e:
        print(f"Other error for '{module_name}': {e}")

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

if __name__ == "__main__":
    import moviepy
    print(f"moviepy version: {moviepy.__version__}")
    print(f"moviepy package path: {moviepy.__file__}")
    
    print("\nChecking specific module paths:")
    check_module_path('moviepy')
    check_module_path('moviepy.audio')
    check_module_path('moviepy.audio.fx')
    check_module_path('moviepy.audio.fx.all')
