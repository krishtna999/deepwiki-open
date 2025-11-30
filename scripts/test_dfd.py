import asyncio
import websockets
import json
import sys

async def test_dfd(repo_url):
    # Default to localhost:8001 as seen in main.py
    uri = "ws://localhost:8001/ws/chat"
    
    print(f"Connecting to {uri}...")
    
    # Request payload
    request = {
        "repo_url": repo_url,
        "type": "github",
        "messages": [
            {"role": "user", "content": "/dfd Generate a DFD for the entire application in Threagile YAML format"}
        ],
        "provider": "openai",
        "model": "gpt-5",
        "language": "en"
    }

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected! Sending DFD request for {repo_url}...")
            
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
                    output_filename = "dfd_output.yaml"
                    with open(output_filename, "w", encoding="utf-8") as f:
                        f.write(full_response)
                    print(f"Output written to {output_filename}")
                    break
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the server is running on port 8001 (default) or check your configuration.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_dfd.py <repo_url>")
        print("Example: python test_dfd.py https://github.com/krishtna999/deepwiki-open")
        sys.exit(1)
        
    repo_url = sys.argv[1]
    asyncio.run(test_dfd(repo_url))
