import asyncio
import json
from typing import Any
from fastapi import WebSocket

from server.game.room import Room, RoomStatus, Player, RoomManager

room_manager = RoomManager()

async def broadcast(room: Room, message: dict, exclude: str = None):
    data = json.dumps(message)
    for player in room.players.values():
        if player.id != exclude:
            try:
                await player.websocket.send_text(data)
            except:
                pass

async def send(ws: WebSocket, message: dict):
    await ws.send_text(json.dumps(message))

async def handle_create(ws: WebSocket, player: Player, data: dict) -> Room:
    duration = data.get('duration', 60)
    best_of = data.get('best_of', 1)
    room = room_manager.create_room(player, duration, best_of)
    await send(ws, {
        'type': 'room_created',
        'room_id': room.id,
        'duration': room.duration,
        'best_of': room.best_of
    })
    return room

async def handle_join(ws: WebSocket, player: Player, data: dict) -> Room:
    room_id = data.get('room_id')
    room = room_manager.join_room(room_id, player)

    if room:
        await send(ws, {
            'type': 'joined',
            'room_id': room.id,
            'duration': room.duration,
            'best_of': room.best_of,
            'players': [p.name for p in room.players.values()]
        })
        await broadcast(room, {
            'type': 'player_joined',
            'player': player.name,
            'players': [p.name for p in room.players.values()]
        }, exclude=player.id)
        return room
    else:
        await send(ws, {'type': 'error', 'message': 'Room not found or full'})
        return None

async def handle_ready(room: Room, player: Player):
    player.ready = True
    await broadcast(room, {
        'type': 'player_ready',
        'player': player.name,
        'all_ready': room.all_ready()
    })

    if room.all_ready():
        await start_countdown(room)

async def start_countdown(room: Room):
    room.status = RoomStatus.COUNTDOWN
    for i in range(3, 0, -1):
        await broadcast(room, {'type': 'countdown', 'count': i})
        await asyncio.sleep(1)
    await start_match(room)

async def start_match(room: Room):
    room.status = RoomStatus.PLAYING
    room.start_time = asyncio.get_event_loop().time()
    import time
    room.start_time = time.time()

    await broadcast(room, {
        'type': 'match_start',
        'duration': room.duration,
        'round': room.round_num
    })

    asyncio.create_task(match_timer(room))

async def match_timer(room: Room):
    await asyncio.sleep(room.duration)
    if room.status == RoomStatus.PLAYING:
        await end_round(room)

async def end_round(room: Room):
    room.status = RoomStatus.FINISHED
    winner = room.get_winner()
    scores = room.get_scores()

    for player in room.players.values():
        if player.name == winner:
            room.round_wins[player.id] += 1

    await broadcast(room, {
        'type': 'round_end',
        'winner': winner,
        'scores': scores,
        'round': room.round_num,
        'round_wins': {p.name: room.round_wins[p.id] for p in room.players.values()}
    })

    wins_needed = (room.best_of // 2) + 1
    match_winner = None
    for player in room.players.values():
        if room.round_wins[player.id] >= wins_needed:
            match_winner = player.name
            break

    if match_winner or room.round_num >= room.best_of:
        await broadcast(room, {
            'type': 'match_end',
            'winner': match_winner or winner,
            'final_scores': {p.name: room.round_wins[p.id] for p in room.players.values()}
        })
    else:
        room.round_num += 1
        room.reset_round()
        room.status = RoomStatus.WAITING
        await broadcast(room, {
            'type': 'next_round',
            'round': room.round_num
        })

async def handle_score(room: Room, player: Player, data: dict):
    if room.status != RoomStatus.PLAYING:
        return

    score = data.get('score', 0)
    player.score = score

    await broadcast(room, {
        'type': 'score_update',
        'player': player.name,
        'score': score,
        'scores': room.get_scores(),
        'time_remaining': room.time_remaining()
    })

async def handle_message(ws: WebSocket, player: Player, room: Room, data: dict) -> Room:
    msg_type = data.get('type')

    if msg_type == 'create':
        return await handle_create(ws, player, data)
    elif msg_type == 'join':
        return await handle_join(ws, player, data)
    elif msg_type == 'ready' and room:
        await handle_ready(room, player)
    elif msg_type == 'score' and room:
        await handle_score(room, player, data)

    return room
