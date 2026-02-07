// ============================================
// CONFIG
// ============================================
const CONFIG = {
	strokeWidth: 8,
};

// ============================================
// STATE
// ============================================
let state = {
	round: 1,
	digit: null,
	humanWins: 0,
	genWins: 0,
	humanTotal: 0,
	genTotal: 0,
	hasDrawn: false,
	judging: false,
};

// ============================================
// SOCKET
// ============================================
const socket = io({
	path: window.location.pathname.startsWith("/gan_game") ? "/gan_game/socket.io" : "/socket.io",
	transports: ["websocket"]
});

socket.on("connect", () => {
	document.getElementById("connDot").classList.add("connected");
	document.getElementById("connText").textContent = "Connected";
	// Start first round when connected
	startNewRound();
});

socket.on("disconnect", () => {
	document.getElementById("connDot").classList.remove("connected");
	document.getElementById("connText").textContent = "Disconnected";
});

socket.on("round_ready", (data) => {
	// Generator has produced its digit - display it
	displayGenImage(data.gen_image);
	document.getElementById("genScoreDisplay").textContent = "Ready";
	document.getElementById("genScoreDisplay").classList.add("waiting");
});

socket.on("game_result", (data) => {
	showResult(data.human_score, data.gen_score);
});

// ============================================
// CANVAS
// ============================================
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
ctx.lineWidth = CONFIG.strokeWidth;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "#fff";

// Preview canvas (shows MNIST-like preprocessed image)
const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");

// Grid overlay canvas
const gridCanvas = document.getElementById("gridCanvas");
const gridCtx = gridCanvas.getContext("2d");

// Draw the 28x28 grid
function drawGrid() {
	const cell = 280 / 28;  // 10px cells
	gridCtx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
	gridCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
	gridCtx.lineWidth = 1;

	for (let i = 0; i <= 28; i++) {
		const pos = i * cell;
		gridCtx.beginPath();
		gridCtx.moveTo(pos, 0);
		gridCtx.lineTo(pos, 280);
		gridCtx.stroke();
		gridCtx.beginPath();
		gridCtx.moveTo(0, pos);
		gridCtx.lineTo(280, pos);
		gridCtx.stroke();
	}
}
drawGrid();

// MNIST view toggle
const mnistViewToggle = document.getElementById("mnistViewToggle");
const canvasContainer = document.querySelector(".canvas-container");

mnistViewToggle.addEventListener("change", (e) => {
	if (e.target.checked) {
		canvasContainer.classList.add("mnist-view");
		updatePreview();
	} else {
		canvasContainer.classList.remove("mnist-view");
	}
});

// MNIST Digit toggle - restart round to fetch real MNIST or Generator
const mnistDigitToggle = document.getElementById("mnistDigitToggle");
mnistDigitToggle.addEventListener("change", () => {
	const isMnist = mnistDigitToggle.checked;
	
	// Update all labels
	document.getElementById("genCanvasLabel").textContent = 
		isMnist ? "MNIST digit" : "AI-generated digit";
	document.getElementById("tugGenLabel").textContent = 
		isMnist ? "MNIST" : "Generator";
	document.getElementById("genWinsLabel").textContent = 
		isMnist ? "MNIST Wins" : "GAN Wins";
	document.getElementById("genPanelTitle").textContent = 
		isMnist ? "MNIST Sample" : "Generator Output";
	document.getElementById("genIcon").textContent = 
		isMnist ? "📊" : "⚡";
	
	// Restart current round with new source (keeps same digit)
	if (state.digit !== null && !state.judging) {
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		genCtx.clearRect(0, 0, genCanvas.width, genCanvas.height);
		previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
		state.hasDrawn = false;
		canvas.classList.remove("active");
		document.getElementById("judgeBtn").disabled = true;
		document.getElementById("humanScoreDisplay").textContent = "Draw to compete";
		document.getElementById("humanScoreDisplay").classList.add("waiting");
		document.getElementById("genScoreDisplay").textContent = "Generating...";
		document.getElementById("genScoreDisplay").classList.add("waiting");
		
		socket.emit("start_round", { digit: state.digit, use_mnist: isMnist });
	}
});

// Update preview canvas with MNIST-like rendering
function updatePreview() {
	if (!mnistViewToggle.checked) return;

	const processed = preprocess();
	
	// Debug: log value distribution
	const nonZero = processed.filter(v => v > 0.01);
	const gray = processed.filter(v => v > 0.01 && v < 0.99);
	console.log(`Preview stats: ${nonZero.length} non-zero pixels, ${gray.length} gray pixels (not pure black/white)`);
	if (gray.length > 0) {
		const grayVals = gray.slice(0, 10).map(v => v.toFixed(2));
		console.log(`Sample gray values: ${grayVals.join(', ')}`);
	}

	// Draw the 28x28 image scaled up to 280x280
	previewCtx.clearRect(0, 0, 280, 280);

	const imgData = previewCtx.createImageData(280, 280);
	for (let y = 0; y < 28; y++) {
		for (let x = 0; x < 28; x++) {
			const val = Math.round(processed[y * 28 + x] * 255);
			// Scale up 10x
			for (let dy = 0; dy < 10; dy++) {
				for (let dx = 0; dx < 10; dx++) {
					const idx = ((y * 10 + dy) * 280 + (x * 10 + dx)) * 4;
					imgData.data[idx] = val;
					imgData.data[idx + 1] = val;
					imgData.data[idx + 2] = val;
					imgData.data[idx + 3] = 255;
				}
			}
		}
	}
	previewCtx.putImageData(imgData, 0, 0);
}

// Show raw drawing without MNIST centering (used during active drawing)
function updatePreviewRaw() {
	previewCtx.clearRect(0, 0, 280, 280);
	previewCtx.drawImage(canvas, 0, 0);
}

let drawing = false;
let lastX = 0;
let lastY = 0;

function getPos(e) {
	const rect = canvas.getBoundingClientRect();
	const scaleX = canvas.width / rect.width;
	const scaleY = canvas.height / rect.height;
	const touch = e.touches?.[0];
	return {
		x: ((touch?.clientX ?? e.clientX) - rect.left) * scaleX,
		y: ((touch?.clientY ?? e.clientY) - rect.top) * scaleY
	};
}

function startDraw(e) {
	if (state.judging) return;
	drawing = true;
	const p = getPos(e);
	lastX = p.x;
	lastY = p.y;
	ctx.beginPath();
	ctx.moveTo(p.x, p.y);
}

function endDraw() {
	drawing = false;
	if (state.hasDrawn) {
		canvas.classList.add("active");
		document.getElementById("judgeBtn").disabled = false;
		// Keep showing raw preview - MNIST view only shown after judging
	}
}

function draw(e) {
	if (!drawing || state.judging) return;
	const p = getPos(e);

	// Interpolate for smoother strokes (MNIST-like)
	const dx = p.x - lastX;
	const dy = p.y - lastY;
	const dist = Math.sqrt(dx * dx + dy * dy);

	if (dist > 5) {
		const steps = Math.ceil(dist / 5);
		const stepX = dx / steps;
		const stepY = dy / steps;

		for (let i = 1; i <= steps; i++) {
			ctx.lineTo(lastX + stepX * i, lastY + stepY * i);
		}
	} else {
		ctx.lineTo(p.x, p.y);
	}

	ctx.stroke();
	ctx.beginPath();
	ctx.moveTo(p.x, p.y);

	lastX = p.x;
	lastY = p.y;
	state.hasDrawn = true;

	// During drawing: show raw strokes (not centered) for better UX
	if (mnistViewToggle.checked) {
		updatePreviewRaw();
	}
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("touchstart", (e) => { e.preventDefault(); startDraw(e); }, { passive: false });
canvas.addEventListener("touchend", (e) => { e.preventDefault(); endDraw(); }, { passive: false });
canvas.addEventListener("touchcancel", (e) => { e.preventDefault(); endDraw(); }, { passive: false });
canvas.addEventListener("touchmove", (e) => { e.preventDefault(); draw(e); }, { passive: false });

// ============================================
// DIGIT SELECTOR (toggleable - click to select/deselect)
// ============================================
const digitContainer = document.getElementById("digitButtons");
let selectedDigit = null;  // null = random mode

for (let i = 0; i <= 9; i++) {
	const btn = document.createElement("button");
	btn.className = "digit-btn";
	btn.textContent = i;
	btn.addEventListener("click", () => {
		if (selectedDigit === i) {
			// Deselect - go back to random mode
			selectedDigit = null;
			btn.classList.remove("selected");
			document.querySelector(".digit-selector-header").textContent = "Target Digit (randomly selected)";
		} else {
			// Select this digit
			document.querySelectorAll(".digit-btn").forEach(b => b.classList.remove("selected"));
			selectedDigit = i;
			btn.classList.add("selected");
			document.querySelector(".digit-selector-header").textContent = "Target Digit (manually selected)";
		}
		
		// Trigger new round with selected/random digit
		if (!state.judging) {
			triggerNewDigit();
		}
	});
	digitContainer.appendChild(btn);
}

function triggerNewDigit() {
	// Use selected digit or pick random
	state.digit = selectedDigit !== null ? selectedDigit : Math.floor(Math.random() * 10);
	
	// Update button highlight for random selection
	if (selectedDigit === null) {
		document.querySelectorAll(".digit-btn").forEach((b, i) => b.classList.toggle("selected", i === state.digit));
	}
	
	// Clear canvases
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	genCtx.clearRect(0, 0, genCanvas.width, genCanvas.height);
	previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
	
	// Reset state
	state.hasDrawn = false;
	canvas.classList.remove("active");
	
	// Reset UI
	document.getElementById("judgeBtn").disabled = true;
	document.getElementById("humanScoreDisplay").textContent = "Draw to compete";
	document.getElementById("humanScoreDisplay").classList.add("waiting");
	document.getElementById("genScoreDisplay").textContent = "Generating...";
	document.getElementById("genScoreDisplay").classList.add("waiting");
	
	// Clear result indicator and panel colors
	const resultEl = document.getElementById("roundResult");
	resultEl.className = "round-result";
	resultEl.textContent = "";
	document.querySelector(".player-panel.human").classList.remove("winner", "loser", "tie");
	document.querySelector(".player-panel.gen").classList.remove("winner", "loser", "tie");
	
	// Request new digit
	const useMnist = document.getElementById("mnistDigitToggle").checked;
	socket.emit("start_round", { digit: state.digit, use_mnist: useMnist });
}

// ============================================
// PREPROCESSING
// ============================================
function preprocess() {
	// Step 1: Apply blur at full resolution for soft edges
	const blurCanvas = document.createElement("canvas");
	blurCanvas.width = canvas.width;
	blurCanvas.height = canvas.height;
	const blurCtx = blurCanvas.getContext("2d");
	blurCtx.filter = "blur(1.5px)";
	blurCtx.drawImage(canvas, 0, 0);

	// Get bounding box from blurred canvas
	const imageData = blurCtx.getImageData(0, 0, blurCanvas.width, blurCanvas.height);
	const data = imageData.data;

	let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
	let sumI = 0;

	for (let y = 0; y < canvas.height; y++) {
		for (let x = 0; x < canvas.width; x++) {
			const idx = (y * canvas.width + x) * 4;
			const intensity = data[idx];
			if (intensity > 10) {
				minX = Math.min(minX, x);
				minY = Math.min(minY, y);
				maxX = Math.max(maxX, x);
				maxY = Math.max(maxY, y);
				sumI += intensity;
			}
		}
	}

	if (minX > maxX || sumI === 0) return new Float32Array(784);

	const bw = maxX - minX + 1;
	const bh = maxY - minY + 1;

	// Step 2: Scale blurred image to 28x28
	const tmp = document.createElement("canvas");
	tmp.width = 28;
	tmp.height = 28;
	const tctx = tmp.getContext("2d");
	tctx.imageSmoothingEnabled = true;
	tctx.imageSmoothingQuality = "high";

	const scale = Math.min(20 / bw, 20 / bh);
	const sw = bw * scale;
	const sh = bh * scale;
	const ox = (28 - sw) / 2;
	const oy = (28 - sh) / 2;

	tctx.drawImage(blurCanvas, minX, minY, bw, bh, ox, oy, sw, sh);

	const img = tctx.getImageData(0, 0, 28, 28);
	const out = new Float32Array(784);
	for (let i = 0; i < 784; i++) out[i] = img.data[i * 4] / 255;
	return out;
}

// ============================================
// GENERATOR DISPLAY
// ============================================
const genCanvas = document.getElementById("genCanvas");
const genCtx = genCanvas.getContext("2d");

function displayGenImage(imageData) {
	// imageData is ArrayBuffer (784 bytes) - convert to Uint8Array
	const bytes = new Uint8Array(imageData);
	const img = genCtx.createImageData(280, 280);

	for (let y = 0; y < 28; y++) {
		for (let x = 0; x < 28; x++) {
			const v = bytes[y * 28 + x];
			// Scale up 10x
			for (let dy = 0; dy < 10; dy++) {
				for (let dx = 0; dx < 10; dx++) {
					const idx = ((y * 10 + dy) * 280 + (x * 10 + dx)) * 4;
					img.data[idx] = v;
					img.data[idx + 1] = v;
					img.data[idx + 2] = v;
					img.data[idx + 3] = 255;
				}
			}
		}
	}
	genCtx.putImageData(img, 0, 0);
}

// ============================================
// TUG OF WAR
// ============================================
function updateTugOfWar() {
	const total = state.humanTotal + state.genTotal;
	if (total === 0) {
		document.getElementById("tugHumanScore").textContent = "50%";
		document.getElementById("tugGenScore").textContent = "50%";
		return;
	}

	const humanPct = (state.humanTotal / total) * 100;
	const genPct = (state.genTotal / total) * 100;

	document.getElementById("tugHumanScore").textContent = humanPct.toFixed(0) + "%";
	document.getElementById("tugGenScore").textContent = genPct.toFixed(0) + "%";
}

// ============================================
// RESULT
// ============================================
function showResult(humanScore, genScore) {
	state.judging = false;

	const hPct = humanScore * 100;
	const gPct = genScore * 100;

	// Update totals
	state.humanTotal += hPct;
	state.genTotal += gPct;

	// Compare with full precision
	const margin = hPct - gPct;
	
	// Choose decimal places based on margin size
	let decimals;
	if (Math.abs(margin) < 0.01) {
		decimals = 4;
	} else if (Math.abs(margin) < 0.1) {
		decimals = 2;
	} else {
		decimals = 1;
	}

	// Display scores with appropriate precision
	document.getElementById("humanScoreDisplay").textContent = hPct.toFixed(decimals) + "%";
	document.getElementById("humanScoreDisplay").classList.remove("waiting");
	document.getElementById("genScoreDisplay").textContent = gPct.toFixed(decimals) + "%";
	document.getElementById("genScoreDisplay").classList.remove("waiting");

	// Show result in inline indicator
	const resultEl = document.getElementById("roundResult");
	const humanPanel = document.querySelector(".player-panel.human");
	const genPanel = document.querySelector(".player-panel.gen");
	
	// Clear previous result classes
	humanPanel.classList.remove("winner", "loser", "tie");
	genPanel.classList.remove("winner", "loser", "tie");
	
	// Get opponent name based on mode
	const opponentName = document.getElementById("mnistDigitToggle").checked ? "MNIST" : "GAN";
	
	if (hPct > gPct) {
		state.humanWins++;
		resultEl.className = "round-result win";
		resultEl.textContent = `You Win! (+${margin.toFixed(decimals)}%)`;
		humanPanel.classList.add("winner");
		genPanel.classList.add("loser");
	} else if (hPct < gPct) {
		state.genWins++;
		resultEl.className = "round-result lose";
		resultEl.textContent = `${opponentName} Wins (${margin.toFixed(decimals)}%)`;
		humanPanel.classList.add("loser");
		genPanel.classList.add("winner");
	} else {
		resultEl.className = "round-result tie";
		resultEl.textContent = "Tie!";
		humanPanel.classList.add("tie");
		genPanel.classList.add("tie");
	}

	// Update UI
	document.getElementById("humanWins").textContent = state.humanWins;
	document.getElementById("genWins").textContent = state.genWins;
	updateTugOfWar();

	// Show Next button
	document.getElementById("judgeBtn").style.display = "none";
	document.getElementById("nextBtn").style.display = "block";
}

// ============================================
// ROUND MANAGEMENT
// ============================================
function startNewRound() {
	// Use selected digit or pick random
	state.digit = selectedDigit !== null ? selectedDigit : Math.floor(Math.random() * 10);

	// Update digit buttons to show selected (for random mode)
	if (selectedDigit === null) {
		document.querySelectorAll(".digit-btn").forEach((b, i) => b.classList.toggle("selected", i === state.digit));
	}

	// Clear canvases
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	genCtx.clearRect(0, 0, genCanvas.width, genCanvas.height);
	previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

	// Reset state
	state.hasDrawn = false;
	state.judging = false;
	canvas.classList.remove("active");

	// Reset UI
	document.getElementById("judgeBtn").disabled = true;
	document.getElementById("humanScoreDisplay").textContent = "Draw to compete";
	document.getElementById("humanScoreDisplay").classList.add("waiting");
	document.getElementById("genScoreDisplay").textContent = "Generating...";
	document.getElementById("genScoreDisplay").classList.add("waiting");
	
	// Clear round result
	const resultEl = document.getElementById("roundResult");
	resultEl.className = "round-result";
	resultEl.textContent = "";
	
	// Clear panel result colors
	document.querySelector(".player-panel.human").classList.remove("winner", "loser", "tie");
	document.querySelector(".player-panel.gen").classList.remove("winner", "loser", "tie");

	// Check if MNIST Digit mode is enabled
	const useMnist = document.getElementById("mnistDigitToggle").checked;

	// Request generator (or MNIST) to produce digit
	socket.emit("start_round", { digit: state.digit, use_mnist: useMnist });
}

function nextRound() {
	// Hide Next, show Judge
	document.getElementById("nextBtn").style.display = "none";
	document.getElementById("judgeBtn").style.display = "block";

	state.round++;
	document.getElementById("roundDisplay").textContent = state.round;
	startNewRound();
}

function resetRound() {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	genCtx.clearRect(0, 0, genCanvas.width, genCanvas.height);

	state.hasDrawn = false;

	canvas.classList.remove("active");
	document.getElementById("judgeBtn").disabled = true;
	document.getElementById("humanScoreDisplay").textContent = "Draw to compete";
	document.getElementById("humanScoreDisplay").classList.add("waiting");
	document.getElementById("genScoreDisplay").textContent = "Ready";
	document.getElementById("genScoreDisplay").classList.add("waiting");
	
	// Clear panel result colors
	document.querySelector(".player-panel.human").classList.remove("winner", "loser", "tie");
	document.querySelector(".player-panel.gen").classList.remove("winner", "loser", "tie");
	
	// Clear round result indicator
	const resultEl = document.getElementById("roundResult");
	resultEl.className = "round-result";
	resultEl.textContent = "";
}

function resetGame() {
	state = {
		round: 1,
		digit: null,
		humanWins: 0,
		genWins: 0,
		humanTotal: 0,
		genTotal: 0,
		hasDrawn: false,
		judging: false,
	};

	document.getElementById("roundDisplay").textContent = "1";
	document.getElementById("humanWins").textContent = "0";
	document.getElementById("genWins").textContent = "0";
	document.getElementById("tugHumanScore").textContent = "50%";
	document.getElementById("tugGenScore").textContent = "50%";
	
	// Reset button visibility
	document.getElementById("nextBtn").style.display = "none";
	document.getElementById("judgeBtn").style.display = "block";

	startNewRound();
}

// ============================================
// BUTTONS
// ============================================
document.getElementById("judgeBtn").onclick = () => {
	if (state.digit === null || !state.hasDrawn || state.judging) return;

	state.judging = true;
	document.getElementById("judgeBtn").disabled = true;
	document.getElementById("humanScoreDisplay").innerHTML = '<span class="waiting-dots">Judging</span>';

	// Send drawing as binary Float32Array
	const imageData = preprocess();
	const buffer = imageData.buffer;
	
	// Show MNIST-preprocessed view when judging
	if (mnistViewToggle.checked) {
		updatePreview();
	}

	socket.emit("judge_drawing", { image: buffer });
};

document.getElementById("clearBtn").onclick = () => {
	if (!state.judging) resetRound();
};

document.getElementById("nextBtn").onclick = nextRound;
document.getElementById("resultNextBtn").onclick = nextRound;
