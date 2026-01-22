#!/usr/bin/env python3
"""
llama-conductor CLI
Launcher for package structure
"""
import sys


def serve(host="0.0.0.0", port=9000):
    """Launch the router"""
    try:
        import uvicorn
        print(f"[llama-conductor] Starting router on {host}:{port}")
        uvicorn.run(
            "llama_conductor.router_fastapi:app",  # NOTE: package.module format
            host=host,
            port=port,
            reload=False,
        )
    except KeyboardInterrupt:
        print("\n[llama-conductor] Shutting down...")
    except Exception as e:
        print(f"[llama-conductor] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="llama-conductor: LLM harness",
        epilog="Example: llama-conductor serve --port 9000"
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    serve_parser = subparsers.add_parser("serve", help="Start the router")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=9000, help="Port (default: 9000)")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        success = serve(host=args.host, port=args.port)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
