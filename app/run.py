#!/usr/bin/env python
"""
Start the Qwen2.5-Omni Server
Alternative entry point (can be run from app directory)
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8665"))
    
    # Get host from environment or use default
    host = os.getenv("HOST", "0.0.0.0")
    
    # Enable reload in development mode
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"🚀 Starting Qwen2.5-Omni Server...")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print(f"🔄 Reload mode: {'enabled' if reload else 'disabled'}")
    print()
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)

