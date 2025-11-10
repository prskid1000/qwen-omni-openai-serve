"""
Main FastAPI Application
Qwen2.5-Omni Model Server
"""

import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .omni_manager import OmniModelManager
from .routes import omni_chat, mcp_servers
from .models import OmniHealthResponse
from .mcp_client_manager import MCPClientManager

# Global managers
omni_manager: Optional[OmniModelManager] = None
mcp_manager: Optional[MCPClientManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global omni_manager, mcp_manager
    
    try:
        print("🚀 Starting Omni Model Server...")
        
        # Initialize Omni model manager
        model_name = os.getenv("OMNI_MODEL_NAME", "wolfofbackstreet/Qwen2.5-Omni-3B-4Bit")  # 4-bit quantized model (bnb)
        use_flash_attention = os.getenv("OMNI_USE_FLASH_ATTENTION", "true").lower() == "true"
        use_cpu_offload = os.getenv("OMNI_USE_CPU_OFFLOAD", "false").lower() == "true"
        
        omni_manager = OmniModelManager(
            model_name=model_name,
            use_cpu_offload=use_cpu_offload,
            use_flash_attention=use_flash_attention
        )
        
        # Load model with talker disabled by default (like USE_TALKER=False in omni_bnb.py)
        omni_manager.load_model(use_talker=False)
        
        # Set manager in routes
        omni_chat.set_omni_manager(omni_manager)
        
        # Initialize MCP client manager
        mcp_manager = MCPClientManager()
        mcp_servers.set_mcp_manager(mcp_manager)
        
        # Update tool service with MCP manager
        from .tool_service import tool_service
        tool_service.mcp_manager = mcp_manager
        
        print(f"✅ Server ready at http://0.0.0.0:8665")
        print(f"📚 Model: {omni_manager.model_name}")
        print(f"🔌 MCP Server Manager initialized")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        raise
    
    yield
    
    # Cleanup
    if mcp_manager:
        print("🔄 Disconnecting MCP servers...")
        await mcp_manager.disconnect_all_servers()
    
    if omni_manager:
        print("🔄 Cleaning up...")
        # PyTorch models don't need explicit cleanup, but we can clear references
        omni_manager.model = None
        omni_manager.processor = None


# Create FastAPI app
app = FastAPI(
    title="Qwen2.5-Omni Server",
    description="FastAPI server for Qwen2.5-Omni multimodal model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(omni_chat.router, tags=["Omni Chat"])
app.include_router(mcp_servers.router, tags=["MCP Servers"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen2.5-Omni Server",
        "version": "1.0.0",
        "model": omni_manager.model_name if omni_manager else "Not loaded",
        "docs": "/docs"
    }


@app.get("/health")
async def health() -> OmniHealthResponse:
    """Health check endpoint"""
    if omni_manager and omni_manager.model:
        device = next(omni_manager.model.parameters()).device
        return OmniHealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=omni_manager.model_name,
            device=str(device),
            context_length=omni_manager.context_length
        )
    else:
        return OmniHealthResponse(
            status="unhealthy",
            model_loaded=False
        )


if __name__ == "__main__":
    import uvicorn
    import sys
    
    port = int(os.getenv("PORT", "8665"))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"🚀 Starting Qwen2.5-Omni Server...")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print()
    
    try:
        uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

