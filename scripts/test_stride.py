import asyncio
import websockets
import json
import sys
import os

async def test_stride(repo_url):
    # Default to localhost:8001 as seen in main.py
    uri = "ws://localhost:8001/ws/chat"
    
    print(f"Connecting to {uri}...")
    
    # Request payload
    request = {
        "repo_url": repo_url,
        "type": "github",
        "messages": [
            {"role": "user", "content": "/stride Generate a STRIDE threat model for the entire application"}
        ],
        "provider": "openai",
        "model": "gpt-5.1",
        "language": "en"
    }

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected! Sending STRIDE request for {repo_url}...")
            
            # Send request
            await websocket.send(json.dumps(request))
            
            # Receive response
            print("\n--- Response Start ---\n")
            full_response = ""
            while True:
                try:
                    response = await websocket.recv()
                    print(response, end="", flush=True)
                    full_response += response
                except websockets.exceptions.ConnectionClosed:
                    print("\n\n--- Response End ---")
                    print("Connection closed.")
                    
                    # Write output to file
                    repo_name = repo_url.rstrip('/').split('/')[-1]
                    output_dir = "outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = os.path.join(output_dir, f"{repo_name}_stride_output.json")
                    
                    with open(output_filename, "w", encoding="utf-8") as f:
                        f.write(full_response)
                    print(f"Output written to {output_filename}")
                    break
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the server is running on port 8001 (default) or check your configuration.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_stride.py <repo_url>")
        print("Example: python test_stride.py https://github.com/krishtna999/deepwiki-open")
        sys.exit(1)
        
    repo_url = sys.argv[1]
    asyncio.run(test_stride(repo_url))
