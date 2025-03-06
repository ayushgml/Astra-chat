import streamlit as st
import ollama
from typing import List, Dict
import time
import json
import re
import os
import datetime
from pathlib import Path
import mimetypes
from PIL import Image
import io

# Add new imports for metrics
import psutil

# Add new imports for file handling
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, TextLexer
from pygments.formatters import HtmlFormatter

# Page configuration
st.set_page_config(
    page_title="Astra Chat",
    page_icon="🤖",
    layout="wide"
)

# Add metrics tracking
class ModelMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_requests = 0
        self.avg_response_time = 0
        self.total_response_time = 0
    
    def update_metrics(self, tokens: int, response_time: float):
        self.total_tokens += tokens
        self.total_requests += 1
        self.total_response_time += response_time
        self.avg_response_time = self.total_response_time / self.total_requests
    
    def get_metrics(self) -> Dict:
        return {
            "Total Tokens": self.total_tokens,
            "Total Requests": self.total_requests,
            "Average Response Time": f"{self.avg_response_time:.2f}s",
            "Uptime": f"{(time.time() - self.start_time) / 3600:.1f}h"
        }

# Initialize metrics in session state
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = ModelMetrics()

# Function to get system metrics
def get_system_metrics() -> Dict:
    metrics = {}
    
    # CPU metrics
    try:
        metrics["CPU Usage"] = f"{psutil.cpu_percent()}%"
    except Exception as e:
        metrics["CPU Usage"] = "N/A"
        print(f"Error getting CPU metrics: {e}")
    
    # Memory metrics
    try:
        memory = psutil.virtual_memory()
        metrics["Memory Usage"] = f"{memory.percent}%"
        metrics["Available Memory"] = f"{memory.available / (1024 * 1024 * 1024):.1f} GB"
    except Exception as e:
        metrics["Memory Usage"] = "N/A"
        metrics["Available Memory"] = "N/A"
        print(f"Error getting memory metrics: {e}")
    
    # GPU metrics (optional)
    try:
        # Only import GPUtil when needed
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            metrics["GPU Usage"] = f"{gpu.load * 100:.1f}%"
            metrics["GPU Memory"] = f"{gpu.memoryUsed}MB / {gpu.memoryTotal}MB"
        else:
            metrics["GPU Status"] = "No GPUs detected"
    except ImportError:
        metrics["GPU Status"] = "GPU monitoring not available"
    except Exception as e:
        metrics["GPU Status"] = "Error getting GPU metrics"
        print(f"Error getting GPU metrics: {e}")
    
    return metrics

# Function to save model settings
def save_model_settings(settings: Dict, filename: str = "model_settings.json"):
    try:
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        
        settings["timestamp"] = datetime.datetime.now().isoformat()
        with open(settings_dir / filename, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save settings: {str(e)}")
        return False

# Function to load model settings
def load_model_settings(filename: str = "model_settings.json") -> Dict:
    try:
        settings_dir = Path("settings")
        settings_file = settings_dir / filename
        if settings_file.exists():
            with open(settings_file, "r") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load settings: {str(e)}")
    return None

# Add version information to available models
AVAILABLE_MODELS = [
    {"name": "deepseek-coder", "versions": ["latest", "v1.5", "v1.0"]},
    {"name": "deepseek-r1:1.5b", "versions": ["latest"]},
    {"name": "llama3.2", "versions": ["latest", "v2", "v1"]},
    {"name": "llama3.2:7b", "versions": ["latest"]},
    {"name": "qwen2.5:3b", "versions": ["latest"]}
]

# Cache for installed models and server status
if "installed_models" not in st.session_state:
    st.session_state.installed_models = set()
if "server_checked" not in st.session_state:
    st.session_state.server_checked = False

# Modified ensure_model_available function with improved caching
def ensure_model_available(model_name: str) -> bool:
    try:
        # If model is in our cache and server was checked before, return True immediately
        if model_name in st.session_state.installed_models and st.session_state.server_checked:
            return True

        # Check if Ollama server is running (only if not checked before)
        if not st.session_state.server_checked:
            try:
                models = ollama.list()
                st.session_state.server_checked = True
                
                # Update cache with all installed models
                if isinstance(models, dict) and 'models' in models:
                    installed_models = [m.get('name', '') for m in models['models'] if isinstance(m, dict)]
                    st.session_state.installed_models.update(installed_models)
                    
                    # If model is now in cache, return True
                    if model_name in st.session_state.installed_models:
                        return True
            except Exception as server_error:
                st.error("⚠️ Error: Cannot connect to Ollama server. Please make sure Ollama is running.")
                return False
        
        # If model not in cache, we need to pull it
        if model_name not in st.session_state.installed_models:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(f"📥 Pulling {model_name}..."):
                try:
                    ollama.pull(model_name)
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"✅ Successfully pulled {model_name}")
                    # Add to cache after successful pull
                    st.session_state.installed_models.add(model_name)
                    return True
                except Exception as pull_error:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Failed to pull {model_name}. Error: {str(pull_error)}")
                    return False
        
        return True
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return False

# Custom CSS for think blocks and thinking animation
st.markdown("""
<style>
    .think-block {
        background-color: #2d3436;
        border-left: 5px solid #00b894;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .thinking {
        color: #00b894;
        animation: blink 1s infinite;
        font-weight: bold;
        margin: 10px 0;
    }
    .math-block {
        background-color: #2d3436;
        border-left: 5px solid #0984e3;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 1.1em;
    }
    .katex-display {
        margin: 0.5em 0 !important;
        overflow-x: auto;
        overflow-y: hidden;
        padding: 2px 0;
    }
    .katex {
        font-size: 1.1em !important;
    }
    .math-inline {
        padding: 0 2px;
    }
</style>
""", unsafe_allow_html=True)

# LaTeX command mappings
LATEX_COMMANDS = {
    # Basic math operations
    'sqrt': '\\sqrt',
    'frac': '\\frac',
    'pm': '\\pm',
    'div': '\\div',
    'times': '\\times',
    'cdot': '\\cdot',
    
    # Greek letters
    'alpha': '\\alpha',
    'beta': '\\beta',
    'gamma': '\\gamma',
    'delta': '\\delta',
    'theta': '\\theta',
    'lambda': '\\lambda',
    'mu': '\\mu',
    'pi': '\\pi',
    'sigma': '\\sigma',
    'omega': '\\omega',
    
    # Operators and symbols
    'sum': '\\sum',
    'prod': '\\prod',
    'int': '\\int',
    'infty': '\\infty',
    'partial': '\\partial',
    'nabla': '\\nabla',
    
    # Arrows and relations
    'rightarrow': '\\rightarrow',
    'leftarrow': '\\leftarrow',
    'Rightarrow': '\\Rightarrow',
    'Leftarrow': '\\Leftarrow',
    'leftrightarrow': '\\leftrightarrow',
    'Leftrightarrow': '\\Leftrightarrow',
    
    # Sets and logic
    'in': '\\in',
    'notin': '\\notin',
    'subset': '\\subset',
    'subseteq': '\\subseteq',
    'cup': '\\cup',
    'cap': '\\cap',
    'emptyset': '\\emptyset',
    'exists': '\\exists',
    'forall': '\\forall',
    
    # Brackets and delimiters
    'left': '\\left',
    'right': '\\right',
    'langle': '\\langle',
    'rangle': '\\rangle',
    'lceil': '\\lceil',
    'rceil': '\\rceil',
    'lfloor': '\\lfloor',
    'rfloor': '\\rfloor',
    
    # Formatting and spacing
    'text': '\\text',
    'textbf': '\\textbf',
    'textit': '\\textit',
    'quad': '\\quad',
    'qquad': '\\qquad',
    
    # Special functions
    'sin': '\\sin',
    'cos': '\\cos',
    'tan': '\\tan',
    'log': '\\log',
    'ln': '\\ln',
    'exp': '\\exp',
    'lim': '\\lim',
    
    # Matrices and alignment
    'matrix': '\\matrix',
    'pmatrix': '\\pmatrix',
    'bmatrix': '\\bmatrix',
    'vmatrix': '\\vmatrix',
    'begin': '\\begin',
    'end': '\\end',
    'align': '\\align',
    
    # Decorations
    'hat': '\\hat',
    'bar': '\\bar',
    'vec': '\\vec',
    'dot': '\\dot',
    'ddot': '\\ddot',
    'overline': '\\overline',
    'underline': '\\underline',
    
    # Boxes and spacing
    'boxed': '\\boxed',
    'space': '\\;',
    'quad': '\\quad',
    'qquad': '\\qquad',
}

def is_math_expression(text: str) -> bool:
    """Check if text contains mathematical expressions."""
    math_patterns = [
        r'\^', r'_', r'\\', r'\$', r'\{', r'\}',  # LaTeX control characters
        r'\b(?:' + '|'.join(map(re.escape, LATEX_COMMANDS.keys())) + r')\b',  # LaTeX commands
        r'[+\-*/=<>≤≥≠]',                         # Mathematical operators
        r'\d+\s*[+\-*/]\s*\d+',                   # Arithmetic expressions
        r'\\begin\{.*?\}',                         # Environment begins
        r'\\end\{.*?\}',                          # Environment ends
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in math_patterns)

def clean_math_expression(expr: str) -> str:
    """Clean and format math expression."""
    # First, protect existing LaTeX commands
    expr = re.sub(r'\\[a-zA-Z]+', lambda m: f'LATEXCMD{hash(m.group(0))}', expr)
    
    # Convert basic superscripts to LaTeX format
    expr = re.sub(r'(\d+|\w+)\^(\d+|\w+)', r'\1^{\2}', expr)
    expr = re.sub(r'(\d+|\w+)²', r'\1^{2}', expr)
    expr = re.sub(r'(\d+|\w+)³', r'\1^{3}', expr)
    
    # Handle special characters that need escaping in LaTeX
    special_chars = {
        '_': '_',  # No need to escape in math mode
        '^': '^',  # No need to escape in math mode
        '\\': '\\',  # Keep backslashes as is in math mode
        '{': '{',  # No need to escape in math mode
        '}': '}',  # No need to escape in math mode
        '#': '\\#',
        '$': '\\$',
        '%': '\\%',
        '&': '\\&',
        '~': '\\sim',
        '|': '\\vert'
    }
    
    # Escape special characters that aren't part of LaTeX commands
    for char, escaped in special_chars.items():
        expr = re.sub(r'(?<!\\)' + re.escape(char), escaped, expr)
    
    # Restore LaTeX commands
    expr = re.sub(r'LATEXCMD-?\d+', lambda m: re.sub(r'LATEXCMD(-?\d+)', 
                 lambda n: [cmd for cmd in LATEX_COMMANDS.values() 
                          if hash(cmd) == int(n.group(1))][0], m.group(0)), expr)
    
    # Replace LaTeX commands
    for cmd, latex_cmd in LATEX_COMMANDS.items():
        expr = re.sub(r'(?<!\\)\b' + re.escape(cmd) + r'\b', latex_cmd, expr, flags=re.IGNORECASE)
    
    # Add proper spacing around operators
    expr = re.sub(r'(?<!\\)([+\-*/=<>])', r' \1 ', expr)
    expr = re.sub(r'\s+', ' ', expr)  # Remove multiple spaces
    
    return expr.strip()

def process_inline_math(content: str) -> str:
    """Process inline math expressions."""
    try:
        def replace_math_parens(match):
            try:
                inner = match.group(1)
                if is_math_expression(inner):
                    cleaned = clean_math_expression(inner)
                    return f'<span class="math-inline">${cleaned}$</span>'
                return f'({inner})'
            except:
                return match.group(0)
        
        # Handle parenthetical expressions
        content = re.sub(r'\((.*?)\)', replace_math_parens, content)
        return content
    except Exception as e:
        # If anything fails, return the original content
        return content

def format_math(content: str) -> str:
    """Convert math expressions to LaTeX math mode."""
    try:
        if not content:
            return content

        # If content already contains LaTeX delimiters, preserve them
        if '\\(' in content or '\\)' in content:
            try:
                # Replace LaTeX delimiters with HTML spans
                content = re.sub(r'\\\((.*?)\\\)', 
                               lambda m: f'<span class="math-inline">${m.group(1)}$</span>',
                               content, flags=re.DOTALL)
            except:
                # If regex fails, return original content
                pass
            return content

        if '$' in content:
            try:
                # Handle existing dollar sign delimiters
                content = re.sub(r'\$\$(.*?)\$\$',
                               lambda m: f'<div class="math-block">$${m.group(1)}$$</div>',
                               content, flags=re.DOTALL)
                content = re.sub(r'\$(.*?)\$',
                               lambda m: f'<span class="math-inline">${m.group(1)}$</span>',
                               content)
            except:
                # If regex fails, return original content
                pass
            return content

        # Process potential math expressions in text
        def process_text(text):
            try:
                # Convert basic math expressions to LaTeX
                text = re.sub(r'(\d+|\w+)\^(\d+|\w+)', r'$\1^{\2}$', text)
                text = re.sub(r'(\d+|\w+)²', r'$\1^{2}$', text)
                text = re.sub(r'(\d+|\w+)³', r'$\1^{3}$', text)
                return text
            except:
                return text

        parts = []
        current_pos = 0
        
        try:
            # Process display math (between square brackets)
            for match in re.finditer(r'\[(.*?)\]', content, re.DOTALL):
                # Add text before the match with inline math processing
                if match.start() > current_pos:
                    text_part = process_text(content[current_pos:match.start()])
                    parts.append(process_inline_math(text_part))
                
                # Process the display math content
                math_content = match.group(1).strip()
                if math_content:
                    parts.append(f'<div class="math-block">$${math_content}$$</div>')
                
                current_pos = match.end()
        except:
            # If regex fails, treat remaining content as text
            current_pos = 0
        
        # Add remaining text with inline math processing
        if current_pos < len(content):
            remaining = process_text(content[current_pos:])
            parts.append(process_inline_math(remaining))
        
        result = ''.join(parts)
        
        # Handle equation environments
        try:
            result = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', 
                          lambda m: f'<div class="math-block">$${m.group(1)}$$</div>', 
                          result, 
                          flags=re.DOTALL)
        except:
            pass
        
        return result
    except Exception as e:
        # If anything fails, return the original content
        return content

def format_message(content: str) -> str:
    """Format message content with think blocks, code blocks, and math expressions."""
    try:
        if not content:
            return content
            
        # Handle streaming cursor
        has_cursor = content.endswith('▌')
        if has_cursor:
            content = content[:-1]
        
        # First handle code blocks with improved pattern matching
        parts = []
        current_pos = 0
        # Updated pattern to be more strict about code block endings
        code_pattern = r'```(?:(\w+)\n)?(.*?)```(?:\n|$)'
        
        for match in re.finditer(code_pattern, content, re.DOTALL):
            # Add text before code block with normal formatting
            if match.start() > current_pos:
                pre_text = content[current_pos:match.start()]
                parts.append(format_non_code_content(pre_text))
            
            # Add code block without math formatting
            lang = match.group(1) or ''
            code = match.group(2).rstrip()  # Remove trailing whitespace
            formatted_code = f'```{lang}\n{code}\n```\n'  # Ensure proper block formatting
            parts.append(formatted_code)
            
            current_pos = match.end()
        
        # Add remaining content with normal formatting
        if current_pos < len(content):
            remaining = content[current_pos:]
            if remaining.strip():  # Only add if there's actual content
                parts.append(format_non_code_content(remaining))
        
        result = ''.join(parts)
        
        # Add back the cursor if it was present
        if has_cursor:
            result += '▌'
        
        return result
    except Exception as e:
        # If anything fails, return the original content
        print(f"Error in format_message: {e}")  # Add debug logging
        return content

def format_non_code_content(content: str) -> str:
    """Format non-code content with think blocks and math expressions."""
    try:
        if not content:
            return content
        
        # Handle think blocks
        parts = re.split(r'(<think>.*?</think>)', content, flags=re.DOTALL)
        formatted_content = []
        
        for part in parts:
            try:
                if part.startswith('<think>') and part.endswith('</think>'):
                    # Format math within think blocks
                    think_content = part[7:-8].strip()
                    formatted_think = format_math(think_content)
                    formatted_content.append(f'<div class="think-block">💭 Thinking: {formatted_think}</div>')
                else:
                    # Handle LaTeX delimiters first
                    if '\\(' in part and '\\)' in part:
                        part = re.sub(r'\\\((.*?)\\\)', 
                                    lambda m: f'<span class="math-inline">${m.group(1)}$</span>',
                                    part, flags=re.DOTALL)
                    # Format remaining math expressions
                    formatted_content.append(format_math(part))
            except Exception as e:
                # If formatting fails, return the original content
                formatted_content.append(part)
        
        return ''.join(formatted_content)
    except Exception as e:
        # If anything fails, return the original content
        return content

# File handling configurations
ALLOWED_FILE_TYPES = {
    'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.h', '.c', '.rs', '.go', '.ts', '.jsx', '.tsx'],
    'document': ['.txt', '.md', '.pdf', '.doc', '.docx'],
    'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
}

def is_file_type_allowed(filename: str) -> bool:
    """Check if the file type is allowed."""
    ext = Path(filename).suffix.lower()
    return any(ext in types for types in ALLOWED_FILE_TYPES.values())

def get_file_category(filename: str) -> str:
    """Get the category of the file."""
    ext = Path(filename).suffix.lower()
    for category, extensions in ALLOWED_FILE_TYPES.items():
        if ext in extensions:
            return category
    return "unknown"

def preview_code(content: bytes, filename: str) -> str:
    """Generate HTML preview for code files."""
    try:
        lexer = get_lexer_for_filename(filename)
    except:
        lexer = TextLexer()
    
    formatter = HtmlFormatter(
        style='monokai',
        cssclass='syntax-highlight',
        linenos=True,
        lineanchors='line',
        anchorlinenos=True,
        noclasses=False
    )
    
    result = highlight(content.decode('utf-8'), lexer, formatter)
    
    # Add custom CSS for better code formatting
    custom_css = """
        <style>
            .syntax-highlight {
                background-color: #272822;
                padding: 1em;
                border-radius: 5px;
                margin: 10px 0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.5;
                overflow-x: auto;
            }
            .syntax-highlight pre {
                margin: 0;
                padding: 0;
                white-space: pre;
                background: transparent;
            }
            .syntax-highlight .linenos {
                color: #75715e;
                padding-right: 1em;
                border-right: 1px solid #49483e;
                -webkit-user-select: none;
                -moz-user-select: none;
                -ms-user-select: none;
                user-select: none;
            }
            .syntax-highlight .code {
                padding-left: 1em;
            }
            /* Preserve whitespace and line breaks */
            .syntax-highlight code {
                white-space: pre;
                word-wrap: normal;
                display: block;
            }
        </style>
    """
    
    return f"{custom_css}<div class='code-preview'>{result}</div>"

def handle_file_upload(uploaded_file) -> Dict:
    """Handle file upload and return file information."""
    if uploaded_file is None:
        return None
    
    try:
        # Get file info
        file_category = get_file_category(uploaded_file.name)
        file_size = uploaded_file.size
        
        # Create unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{uploaded_file.name}"
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = upload_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return {
            "filename": uploaded_file.name,
            "saved_as": unique_filename,
            "category": file_category,
            "size": file_size,
            "path": str(file_path)
        }
    except Exception as e:
        st.error(f"Error handling file upload: {str(e)}")
        return None

def render_file_upload_section():
    """Render the file upload section in the interface."""
    st.subheader("📎 File Upload")
    
    # Initialize last_uploaded_file in session state if it doesn't exist
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=[ext[1:] for types in ALLOWED_FILE_TYPES.values() for ext in types],
        help="Upload code, documents, or images to discuss them with the AI."
    )
    
    if uploaded_file:
        # Check if this is a new file upload
        current_file = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_uploaded_file != current_file:
            st.session_state.last_uploaded_file = current_file
            file_info = handle_file_upload(uploaded_file)
            
            if file_info:
                st.success(f"✅ File uploaded successfully: {file_info['filename']}")
                
                # Show file preview based on category
                with st.expander("📄 File Preview", expanded=True):
                    if file_info['category'] == 'image':
                        image = Image.open(file_info['path'])
                        st.image(image, caption=file_info['filename'])
                    elif file_info['category'] == 'code':
                        with open(file_info['path'], 'rb') as f:
                            content = f.read()
                        st.markdown(preview_code(content, file_info['filename']), unsafe_allow_html=True)
                    else:
                        try:
                            with open(file_info['path'], 'r') as f:
                                content = f.read()
                            st.text_area("File Content", content, height=200)
                        except:
                            st.warning("Preview not available for this file type")
                
                # Add file to chat context
                if "current_files" not in st.session_state:
                    st.session_state.current_files = []
                st.session_state.current_files.append(file_info)
                
                # Read file content for context
                try:
                    with open(file_info['path'], 'r') as f:
                        file_content = f.read()
                    
                    # Update chat context with file information and content
                    context_message = {
                        "role": "system",
                        "content": f"User has uploaded a file: {file_info['filename']} ({file_info['category']} file, {file_info['size']} bytes)\n\nFile content:\n\n{file_content}"
                    }
                    st.session_state.messages.append(context_message)
                except Exception as e:
                    # If we can't read the file content (e.g., binary file), just add the metadata
                    context_message = {
                        "role": "system",
                        "content": f"User has uploaded a file: {file_info['filename']} ({file_info['category']} file, {file_info['size']} bytes)"
                    }
                    st.session_state.messages.append(context_message)

# Initialize session state for chat history and model settings
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_settings" not in st.session_state:
    st.session_state.model_settings = {
        "model": "deepseek-coder",
        "temperature": 0.7,
        "context_window": 4096
    }

# Sidebar for model settings
with st.sidebar:
    st.title("⚙️ Model Settings")
    
    # Model selection with version
    try:
        # Check Ollama server status
        try:
            ollama.list()
            server_status = "🟢 Ollama server is running"
        except Exception:
            server_status = "🔴 Ollama server is not running"
        
        st.info(server_status)
        
        # Model selection
        selected_model_info = st.selectbox(
            "Select Model",
            options=AVAILABLE_MODELS,
            format_func=lambda x: x["name"],
            index=0,
            help="Choose a model to use for chat."
        )
        
        # Version selection
        selected_version = st.selectbox(
            "Select Version",
            options=selected_model_info["versions"],
            index=0,
            help="Choose the model version."
        )
        
        selected_model = selected_model_info["name"]
        if selected_version != "latest":
            selected_model = f"{selected_model}:{selected_version}"
        
        # Show model status
        if ensure_model_available(selected_model):
            st.success(f"✅ Model {selected_model} is ready")
        else:
            st.error(f"❌ Model {selected_model} is not available")
            selected_model = "deepseek-coder"
        
        # Model metrics
        st.subheader("📊 Model Metrics")
        metrics = st.session_state.model_metrics.get_metrics()
        for metric_name, metric_value in metrics.items():
            st.metric(metric_name, metric_value)
        
        # Settings export/import
        st.subheader("⚙️ Settings Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Settings"):
                current_settings = {
                    "model": selected_model,
                    "version": selected_version,
                    "temperature": st.session_state.model_settings["temperature"],
                    "context_window": st.session_state.model_settings["context_window"]
                }
                if save_model_settings(current_settings):
                    st.success("Settings exported successfully!")
        
        with col2:
            if st.button("Import Settings"):
                loaded_settings = load_model_settings()
                if loaded_settings:
                    st.session_state.model_settings.update(loaded_settings)
                    st.success("Settings imported successfully!")
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error with model selection: {str(e)}")
        st.info("ℹ️ Using default model: deepseek-coder")
        selected_model = "deepseek-coder"
    
    # Temperature control
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.model_settings["temperature"],
        step=0.1,
        help="Higher values (closer to 1.0) make the output more random, lower values (closer to 0.0) make it more deterministic"
    )
    
    # Context window size
    context_window = st.select_slider(
        "Context Window Size",
        options=[1024, 2048, 4096, 8192],
        value=st.session_state.model_settings["context_window"],
        help="Maximum number of tokens to consider for context. Larger values allow for more context but use more memory."
    )
    
    # Update model settings
    st.session_state.model_settings.update({
        "model": selected_model,
        "temperature": temperature,
        "context_window": context_window
    })
    
    # Display current settings
    st.markdown("### Current Settings")
    st.json(st.session_state.model_settings, expanded=False)

# Main chat interface
st.title("🤖 Astra Chat")
st.markdown("""
    Chat with AI models using Ollama. The responses will be streamed word by word.
    Type your message below and press Enter to start chatting.
""")

# Add file upload section
render_file_upload_section()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_message(message["content"]), unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add assistant message to chat history
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        start_time = time.time()  # Track response time

        try:
            # Check if model is available before chatting
            if ensure_model_available(st.session_state.model_settings["model"]):
                # Get the response stream with full conversation history
                stream = ollama.chat(
                    model=st.session_state.model_settings["model"],
                    messages=st.session_state.messages,  # Pass the entire conversation history
                    stream=True,
                    options={
                        "temperature": st.session_state.model_settings["temperature"],
                        "num_ctx": st.session_state.model_settings["context_window"]
                    }
                )

                token_count = 0  # Track tokens for this response
                # Process the stream
                for chunk in stream:
                    if chunk.get('message', {}).get('content'):
                        content = chunk['message']['content']
                        full_response += content
                        token_count += len(content.split())  # Approximate token count
                        # Format the response with think blocks
                        formatted_response = format_message(full_response + "▌")
                        message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                        time.sleep(0.01)  # Small delay for better readability
                
                # Update metrics and store response in session state
                response_time = time.time() - start_time
                st.session_state.model_metrics.update_metrics(token_count, response_time)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Display final response
                formatted_response = format_message(full_response)
                message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                
                # Force a rerun to update metrics in sidebar
                st.rerun()
            else:
                st.error("❌ Selected model is not available. Please choose a different model.")
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.info("ℹ️ Please try again or check your model settings.")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 