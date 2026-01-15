import asyncio
import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from server.game.room import Player, RoomStatus
from server.ws.handlers import handle_message, broadcast, room_manager

app = FastAPI(title="Six Seven Gesture Game")

client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")

@app.get("/")
async def root():
    index_path = client_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Six Seven Gesture Game API"}

@app.get("/api/rooms")
async def list_rooms():
    rooms = []
    for room in room_manager.rooms.values():
        if room.status == RoomStatus.WAITING and len(room.players) < 2:
            rooms.append({
                "id": room.id,
                "players": len(room.players),
                "duration": room.duration,
                "best_of": room.best_of
            })
    return {"rooms": rooms}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    player_id = str(uuid.uuid4())
    player_name = f"Player_{player_id[:4]}"
    player = Player(id=player_id, name=player_name, websocket=websocket)
    room = None

    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "player_id": player_id,
            "player_name": player_name
        }))

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "set_name":
                player.name = message.get("name", player.name)
                await websocket.send_text(json.dumps({
                    "type": "name_set",
                    "name": player.name
                }))
                continue

            room = await handle_message(websocket, player, room, message)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if room:
            room.remove_player(player_id)
            if len(room.players) == 0:
                room_manager.delete_room(room.id)
            else:
                await broadcast(room, {
                    "type": "player_left",
                    "player": player.name
                })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
