import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid

class RoomStatus(Enum):
    WAITING = "waiting"
    READY = "ready"
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    FINISHED = "finished"

@dataclass
class Player:
    id: str
    name: str
    websocket: any
    score: int = 0
    ready: bool = False

@dataclass
class Room:
    id: str
    creator_id: str
    duration: int = 60
    best_of: int = 1
    status: RoomStatus = RoomStatus.WAITING
    players: dict = field(default_factory=dict)
    round_num: int = 1
    round_wins: dict = field(default_factory=dict)
    start_time: Optional[float] = None

    def add_player(self, player: Player) -> bool:
        if len(self.players) >= 2:
            return False
        self.players[player.id] = player
        self.round_wins[player.id] = 0
        return True

    def remove_player(self, player_id: str):
        if player_id in self.players:
            del self.players[player_id]
        if player_id in self.round_wins:
            del self.round_wins[player_id]

    def all_ready(self) -> bool:
        return len(self.players) == 2 and all(p.ready for p in self.players.values())

    def reset_round(self):
        for player in self.players.values():
            player.score = 0
            player.ready = False
        self.start_time = None

    def get_scores(self) -> dict:
        return {p.name: p.score for p in self.players.values()}

    def get_winner(self) -> Optional[str]:
        if len(self.players) != 2:
            return None
        players = list(self.players.values())
        if players[0].score > players[1].score:
            return players[0].name
        elif players[1].score > players[0].score:
            return players[1].name
        return "Tie"

    def time_remaining(self) -> float:
        if self.start_time is None:
            return self.duration
        elapsed = time.time() - self.start_time
        return max(0, self.duration - elapsed)

class RoomManager:
    def __init__(self):
        self.rooms: dict[str, Room] = {}

    def create_room(self, creator: Player, duration: int = 60, best_of: int = 1) -> Room:
        room_id = str(uuid.uuid4())[:8]
        room = Room(id=room_id, creator_id=creator.id, duration=duration, best_of=best_of)
        room.add_player(creator)
        self.rooms[room_id] = room
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)

    def join_room(self, room_id: str, player: Player) -> Optional[Room]:
        room = self.rooms.get(room_id)
        if room and room.add_player(player):
            return room
        return None

    def delete_room(self, room_id: str):
        if room_id in self.rooms:
            del self.rooms[room_id]
