import asyncio
import json
import websockets
import time

async def test():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as ws:
            # 1. Send the loud bark chunk to trigger the capture mechanism
            bark_payload = {
                "pcm_samples": [3000] * 2000, 
                "rms": 3000
            }
            await ws.send(json.dumps(bark_payload))
            print("🚀 Loud bark chunk delivered...")
            
            # Tiny delay to mimic hardware timing gap
            await asyncio.sleep(0.1)
            
            # 2. Stream quiet chunks to fill out the 800ms silence holding window
            print("🤫 Streaming silent trailing frames to close out the buffer...")
            for i in range(9):
                silence_payload = {
                    "pcm_samples": [50] * 500,
                    "rms": 50
                }
                await ws.send(json.dumps(silence_payload))
                await asyncio.sleep(0.1)

            print("✅ Simulation sequence complete!")
            
    except Exception as e:
        print(f"❌ Connection error: {e}. Is bridge_server.py running?")

if __name__ == "__main__":
    asyncio.run(test())