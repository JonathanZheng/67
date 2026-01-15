let ws = null;
let myName = '';
let roomId = '';
let myScore = 0;
let gameActive = false;
let timerInterval = null;
let scores = {};

const MODEL_URL = '/model/model.onnx';
let detector = null;

function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => console.log('Connected');

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = () => {
        console.log('Disconnected');
        setTimeout(connect, 2000);
    };
}

function handleMessage(data) {
    console.log('Message:', data);

    switch (data.type) {
        case 'connected':
            myName = data.player_name;
            document.getElementById('playerName').value = myName;
            break;

        case 'room_created':
        case 'joined':
            roomId = data.room_id;
            document.getElementById('roomId').textContent = roomId;
            showPanel('waitingPanel');
            if (data.players) updatePlayerList(data.players);
            break;

        case 'player_joined':
            updatePlayerList(data.players);
            document.getElementById('readyBtn').disabled = false;
            break;

        case 'player_left':
            document.getElementById('readyBtn').disabled = true;
            break;

        case 'player_ready':
            break;

        case 'countdown':
            showCountdown(data.count);
            break;

        case 'match_start':
            startGame(data.duration);
            break;

        case 'score_update':
            updateScores(data.scores);
            break;

        case 'round_end':
        case 'match_end':
            endGame(data);
            break;

        case 'error':
            alert(data.message);
            break;
    }
}

function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
    }
}

function setName() {
    myName = document.getElementById('playerName').value || myName;
    send({ type: 'set_name', name: myName });
}

function createRoom() {
    setName();
    send({
        type: 'create',
        duration: parseInt(document.getElementById('duration').value),
        best_of: parseInt(document.getElementById('bestOf').value)
    });
}

function joinRoom() {
    setName();
    const code = document.getElementById('roomCode').value;
    if (code) {
        send({ type: 'join', room_id: code });
    }
}

async function refreshRooms() {
    const res = await fetch('/api/rooms');
    const data = await res.json();
    const list = document.getElementById('roomList');
    list.innerHTML = data.rooms.map(r => `
        <div class="room-item">
            <span>${r.id} - ${r.duration}s - ${r.players}/2 players</span>
            <button onclick="joinRoomById('${r.id}')">Join</button>
        </div>
    `).join('');
}

function joinRoomById(id) {
    document.getElementById('roomCode').value = id;
    joinRoom();
}

function updatePlayerList(players) {
    document.getElementById('playerList').textContent = `Players: ${players.join(', ')}`;
    if (players.length === 2) {
        document.getElementById('readyBtn').disabled = false;
    }
}

function ready() {
    send({ type: 'ready' });
    document.getElementById('readyBtn').disabled = true;
    document.getElementById('readyBtn').textContent = 'Waiting...';
}

function showCountdown(count) {
    const el = document.getElementById('countdown');
    el.textContent = count;
    el.classList.remove('hidden');
}

async function startGame(duration) {
    document.getElementById('countdown').classList.add('hidden');
    showPanel('gamePanel');
    gameActive = true;
    myScore = 0;
    scores = {};

    document.getElementById('p1Name').textContent = myName;

    let timeLeft = duration;
    document.getElementById('timer').textContent = timeLeft;

    timerInterval = setInterval(() => {
        timeLeft--;
        document.getElementById('timer').textContent = timeLeft;
        if (timeLeft <= 0) clearInterval(timerInterval);
    }, 1000);

    await startWebcam();
    startDetection();
}

async function startWebcam() {
    const video = document.getElementById('video');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        video.srcObject = stream;
    } catch (e) {
        console.error('Webcam error:', e);
        alert('Could not access webcam');
    }
}

let lastDetectionTime = 0;
const COOLDOWN_MS = 1500;
let frameBuffer = [];
const BUFFER_SIZE = 60;

function startDetection() {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');

    function processFrame() {
        if (!gameActive) return;

        ctx.drawImage(video, 0, 0, 224, 224);
        const imageData = ctx.getImageData(0, 0, 224, 224);
        frameBuffer.push(imageData);

        if (frameBuffer.length > BUFFER_SIZE) {
            frameBuffer.shift();
        }

        const now = Date.now();
        if (frameBuffer.length >= BUFFER_SIZE && now - lastDetectionTime > COOLDOWN_MS) {
            const detected = simulateDetection();
            if (detected) {
                lastDetectionTime = now;
                myScore++;
                updateMyScore();
                flashDetection();
                send({ type: 'score', score: myScore });
            }
        }

        requestAnimationFrame(processFrame);
    }

    processFrame();
}

function simulateDetection() {
    return Math.random() < 0.02;
}

function updateMyScore() {
    document.getElementById('myScore').textContent = myScore;
    document.getElementById('p1Score').textContent = myScore;
}

function updateScores(newScores) {
    scores = newScores;
    const names = Object.keys(scores);
    const opponent = names.find(n => n !== myName);

    if (opponent) {
        document.getElementById('p2Name').textContent = opponent;
        document.getElementById('p2Score').textContent = scores[opponent];
    }

    document.getElementById('p1Score').textContent = scores[myName] || 0;
}

function flashDetection() {
    const flash = document.getElementById('flash');
    flash.classList.add('active');
    setTimeout(() => flash.classList.remove('active'), 200);
}

function endGame(data) {
    gameActive = false;
    clearInterval(timerInterval);
    frameBuffer = [];

    const video = document.getElementById('video');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
    }

    document.getElementById('resultTitle').textContent =
        data.winner === myName ? 'You Win!' :
        data.winner === 'Tie' ? "It's a Tie!" : 'You Lose';

    const scoresHtml = Object.entries(data.scores || data.final_scores || {})
        .map(([name, score]) => `
            <div class="player">
                <div class="name">${name}</div>
                <div>${score}</div>
            </div>
        `).join('');

    document.getElementById('finalScores').innerHTML = scoresHtml;
    showPanel('resultPanel');
}

function playAgain() {
    showPanel('waitingPanel');
    document.getElementById('readyBtn').disabled = false;
    document.getElementById('readyBtn').textContent = 'Ready';
}

function backToLobby() {
    location.reload();
}

function showPanel(panelId) {
    ['lobbyPanel', 'waitingPanel', 'gamePanel', 'resultPanel'].forEach(id => {
        document.getElementById(id).classList.toggle('hidden', id !== panelId);
    });
}

connect();
