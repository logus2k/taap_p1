// ============================================
// CONFIG
// ============================================
const CONFIG = {
	trialsPerRound: 10,
	difficulty: {
		basic: 2,
		average: 1,
		pro: 0.5
	}
};

// MNIST preview parameters (adjustable via debug sliders)
const mnistParams = {
	sigma: 0.7,
	gamma: 2.00,
	contrast: 1.9,
	brightness: 0.15,
	strokeWidth: 8
};

// ============================================
// STATE
// ============================================
let selectedDigit = null;  // null = random mode

let state = {
	round: 1,
	trial: 1,
	digit: null,
	humanMargin: 0,    // Cumulative margin for human
	genMargin: 0,      // Cumulative margin for GAN
	humanTrialWins: 0, // Trial wins this round
	genTrialWins: 0,   // Trial wins this round
	humanRounds: 0,    // Round wins
	genRounds: 0,      // Round wins
	humanPct: 50.0,    // Tug-of-war percentage
	genPct: 50.0,
	hasDrawn: false,
	judging: false,
	currentDifficulty: 'basic',
	timeLimit: CONFIG.difficulty.basic,
	timerInterval: null,
	timeRemaining: CONFIG.difficulty.basic,
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
	// Set initial tug-of-war colors (tied = yellow)
	updateTugOfWar();
	// Start first round when connected
	startNewRound();
});

socket.on("disconnect", () => {
	document.getElementById("connDot").classList.remove("connected");
	document.getElementById("connText").textContent = "Disconnected";
});

socket.on("round_ready", (data) => {
	// Generator has produced its digit - display it
	console.log("[game] round_ready received, gen_image length:", data.gen_image ? data.gen_image.byteLength : "null");
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
ctx.lineWidth = mnistParams.strokeWidth;
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
const genCanvasContainer = document.querySelector(".gen-canvas-container");

// Generator grid canvas
const genGridCanvas = document.getElementById("genGridCanvas");
const genGridCtx = genGridCanvas.getContext("2d");

function drawGenGrid() {
	const cell = 280 / 28;
	genGridCtx.clearRect(0, 0, genGridCanvas.width, genGridCanvas.height);
	genGridCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
	genGridCtx.lineWidth = 1;

	for (let i = 0; i <= 28; i++) {
		const pos = i * cell;
		genGridCtx.beginPath();
		genGridCtx.moveTo(pos, 0);
		genGridCtx.lineTo(pos, 280);
		genGridCtx.stroke();
		genGridCtx.beginPath();
		genGridCtx.moveTo(0, pos);
		genGridCtx.lineTo(280, pos);
		genGridCtx.stroke();
	}
}
drawGenGrid();

mnistViewToggle.addEventListener("change", (e) => {
	if (e.target.checked) {
		canvasContainer.classList.add("mnist-view");
		genCanvasContainer.classList.add("mnist-view-active");
		updatePreview();
	} else {
		canvasContainer.classList.remove("mnist-view");
		genCanvasContainer.classList.remove("mnist-view-active");
	}
});

// Set MNIST View as default
mnistViewToggle.checked = true;
canvasContainer.classList.add("mnist-view");
genCanvasContainer.classList.add("mnist-view-active");

// MNIST Digit toggle - restart round to fetch real MNIST or Generator
const mnistDigitToggle = document.getElementById("mnistDigitToggle");
mnistDigitToggle.addEventListener("change", () => {
	const isMnist = mnistDigitToggle.checked;
	
	// Update panel title
	document.getElementById("genPanelTitle").textContent = 
		isMnist ? "MNIST digit" : "AI-generated digit";
	document.getElementById("tugGenLabel").textContent = 
		isMnist ? "MNIST" : "Generator";
	document.getElementById("genWinsLabel").textContent = 
		isMnist ? "MNIST Margin" : "GAN Margin";
	
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

// Update preview canvas with MNIST-like rendering (no centering during drawing)
function updatePreview() {
	if (!mnistViewToggle.checked) return;

	const processed = preprocessLive();

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
	if (state.judging || state.timeRemaining <= 0) return;
	drawing = true;
	const p = getPos(e);
	lastX = p.x;
	lastY = p.y;
	ctx.beginPath();
	ctx.moveTo(p.x, p.y);
	
	// Start timer on first stroke
	if (!state.hasDrawn && !state.timerInterval) {
		startTimer();
	}
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
	if (!drawing || state.judging || state.timeRemaining <= 0) return;
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

	// During drawing: show pixelated preview if MNIST view is enabled
	if (mnistViewToggle.checked) {
		updatePreview();
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

// Live preview - no centering, with configurable antialiasing
function preprocessLive() {
	const tmp = document.createElement("canvas");
	tmp.width = 28;
	tmp.height = 28;
	const tctx = tmp.getContext("2d");
	tctx.imageSmoothingEnabled = true;
	tctx.imageSmoothingQuality = "high";
	
	// Scale to 28x28
	tctx.drawImage(canvas, 0, 0, 28, 28);
	
	// Get image data
	const img = tctx.getImageData(0, 0, 28, 28);
	
	// Apply blur if sigma > 0
	const processed = mnistParams.sigma > 0 
		? gaussianBlur(img.data, 28, 28, mnistParams.sigma)
		: img.data;
	
	const out = new Float32Array(784);
	for (let i = 0; i < 784; i++) {
		let val = processed[i * 4] / 255;
		
		// Apply gamma
		val = Math.pow(val, mnistParams.gamma);
		
		// Apply contrast (centered at 0.5)
		val = (val - 0.5) * mnistParams.contrast + 0.5;
		
		// Apply brightness
		val = val + mnistParams.brightness;
		
		out[i] = Math.max(0, Math.min(1.0, val));
	}
	
	return out;
}

// Settings panel handlers
function setupSettingsPanel() {
	const settingsGear = document.getElementById("settingsGear");
	const settingsPanel = document.getElementById("settingsPanel");
	const settingsClose = document.getElementById("settingsClose");
	
	settingsGear.addEventListener("click", () => {
		settingsPanel.classList.toggle("show");
	});
	
	settingsClose.addEventListener("click", () => {
		settingsPanel.classList.remove("show");
	});
	
	// Close panel when clicking outside
	document.addEventListener("click", (e) => {
		if (!settingsPanel.contains(e.target) && !settingsGear.contains(e.target)) {
			settingsPanel.classList.remove("show");
		}
	});
	
	// Sigma slider
	const sigmaSlider = document.getElementById("sigmaSlider");
	const sigmaValue = document.getElementById("sigmaValue");
	sigmaSlider.addEventListener("input", () => {
		mnistParams.sigma = parseFloat(sigmaSlider.value);
		sigmaValue.textContent = mnistParams.sigma.toFixed(1);
		if (mnistViewToggle.checked) updatePreview();
	});
	
	// Gamma slider
	const gammaSlider = document.getElementById("gammaSlider");
	const gammaValue = document.getElementById("gammaValue");
	gammaSlider.addEventListener("input", () => {
		mnistParams.gamma = parseFloat(gammaSlider.value);
		gammaValue.textContent = mnistParams.gamma.toFixed(2);
		if (mnistViewToggle.checked) updatePreview();
	});
	
	// Contrast slider
	const contrastSlider = document.getElementById("contrastSlider");
	const contrastValue = document.getElementById("contrastValue");
	contrastSlider.addEventListener("input", () => {
		mnistParams.contrast = parseFloat(contrastSlider.value);
		contrastValue.textContent = mnistParams.contrast.toFixed(1);
		if (mnistViewToggle.checked) updatePreview();
	});
	
	// Brightness slider
	const brightnessSlider = document.getElementById("brightnessSlider");
	const brightnessValue = document.getElementById("brightnessValue");
	brightnessSlider.addEventListener("input", () => {
		mnistParams.brightness = parseFloat(brightnessSlider.value);
		brightnessValue.textContent = mnistParams.brightness.toFixed(2);
		if (mnistViewToggle.checked) updatePreview();
	});
	
	// Stroke width slider
	const strokeWidthSlider = document.getElementById("strokeWidthSlider");
	const strokeWidthValue = document.getElementById("strokeWidthValue");
	strokeWidthSlider.addEventListener("input", () => {
		mnistParams.strokeWidth = parseInt(strokeWidthSlider.value);
		strokeWidthValue.textContent = mnistParams.strokeWidth;
		ctx.lineWidth = mnistParams.strokeWidth;
	});
}

setupSettingsPanel();

// Simple Gaussian blur implementation
function gaussianBlur(data, width, height, sigma) {
	const kernel = makeGaussianKernel(sigma);
	const kSize = kernel.length;
	const kHalf = Math.floor(kSize / 2);
	
	const result = new Uint8ClampedArray(data.length);
	
	// Horizontal pass
	const temp = new Uint8ClampedArray(data.length);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			let r = 0, g = 0, b = 0, a = 0, wSum = 0;
			for (let k = 0; k < kSize; k++) {
				const xi = Math.min(Math.max(x + k - kHalf, 0), width - 1);
				const idx = (y * width + xi) * 4;
				const w = kernel[k];
				r += data[idx] * w;
				g += data[idx + 1] * w;
				b += data[idx + 2] * w;
				a += data[idx + 3] * w;
				wSum += w;
			}
			const idx = (y * width + x) * 4;
			temp[idx] = r / wSum;
			temp[idx + 1] = g / wSum;
			temp[idx + 2] = b / wSum;
			temp[idx + 3] = a / wSum;
		}
	}
	
	// Vertical pass
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			let r = 0, g = 0, b = 0, a = 0, wSum = 0;
			for (let k = 0; k < kSize; k++) {
				const yi = Math.min(Math.max(y + k - kHalf, 0), height - 1);
				const idx = (yi * width + x) * 4;
				const w = kernel[k];
				r += temp[idx] * w;
				g += temp[idx + 1] * w;
				b += temp[idx + 2] * w;
				a += temp[idx + 3] * w;
				wSum += w;
			}
			const idx = (y * width + x) * 4;
			result[idx] = r / wSum;
			result[idx + 1] = g / wSum;
			result[idx + 2] = b / wSum;
			result[idx + 3] = a / wSum;
		}
	}
	
	return result;
}

function makeGaussianKernel(sigma) {
	const size = Math.ceil(sigma * 3) * 2 + 1;
	const kernel = new Array(size);
	const mean = Math.floor(size / 2);
	let sum = 0;
	
	for (let i = 0; i < size; i++) {
		kernel[i] = Math.exp(-0.5 * Math.pow((i - mean) / sigma, 2));
		sum += kernel[i];
	}
	
	// Normalize
	for (let i = 0; i < size; i++) {
		kernel[i] /= sum;
	}
	
	return kernel;
}

// Final preprocessing for submission - with centering (MNIST-style)
function preprocess() {
	// Get original image data
	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
	const data = imageData.data;
	
	// Find bounding box
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

	// Scale to 28x28 with smoothing
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

	tctx.drawImage(canvas, minX, minY, bw, bh, ox, oy, sw, sh);

	// Get scaled image - no blur, just like MNIST UI
	const img = tctx.getImageData(0, 0, 28, 28);
	const out = new Float32Array(784);
	for (let i = 0; i < 784; i++) {
		out[i] = img.data[i * 4] / 255;
	}
	
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
// TIMER
// ============================================
function startTimer() {
	state.timeRemaining = state.timeLimit;
	updateTimerDisplay();
	
	const timerDisplay = document.getElementById('timerDisplay');
	timerDisplay.classList.add('active');
	timerDisplay.classList.remove('warning');
	
	state.timerInterval = setInterval(() => {
		state.timeRemaining -= 0.1;
		updateTimerDisplay();
		
		// Warning when < 2 seconds
		if (state.timeRemaining <= 2) {
			timerDisplay.classList.add('warning');
		}
		
		// Auto-submit when time runs out
		if (state.timeRemaining <= 0) {
			stopTimer();
			autoSubmit();
		}
	}, 100);
}

function stopTimer() {
	if (state.timerInterval) {
		clearInterval(state.timerInterval);
		state.timerInterval = null;
	}
	const timerDisplay = document.getElementById('timerDisplay');
	timerDisplay.classList.remove('active', 'warning');
}

function updateTimerDisplay() {
	const display = Math.max(0, state.timeRemaining).toFixed(1);
	document.getElementById('timerValue').textContent = display;
}

function resetTimer() {
	stopTimer();
	state.timeRemaining = state.timeLimit;
	updateTimerDisplay();
}

function autoSubmit() {
	// Don't submit if already judging
	if (state.judging) return;
	
	// If user has drawn something, submit it
	// If user hasn't drawn anything, submit empty canvas (will lose)
	state.judging = true;
	stopTimer();
	document.getElementById("judgeBtn").disabled = true;
	document.getElementById("humanScoreDisplay").innerHTML = '<span class="waiting-dots">Time\'s up!</span>';

	// Send drawing as binary Float32Array
	const imageData = preprocess();
	const buffer = imageData.buffer;

	// Show MNIST-preprocessed view when judging
	if (mnistViewToggle.checked) {
		updatePreview();
	}

	socket.emit("judge_drawing", { image: buffer });
}

// ============================================
// MARGIN DISPLAY
// ============================================
function updateMarginDisplay() {
	const humanMarginEl = document.getElementById("humanMargin");
	const genMarginEl = document.getElementById("genMargin");
	
	// Format with sign and 2 decimal places
	const humanText = (state.humanMargin >= 0 ? "+" : "") + state.humanMargin.toFixed(2) + "%";
	const genText = (state.genMargin >= 0 ? "+" : "") + state.genMargin.toFixed(2) + "%";
	
	humanMarginEl.textContent = humanText;
	genMarginEl.textContent = genText;
	
	// Color based on who's ahead
	humanMarginEl.classList.remove("positive", "negative");
	genMarginEl.classList.remove("positive", "negative");
	
	if (state.humanMargin > state.genMargin) {
		humanMarginEl.classList.add("positive");
		genMarginEl.classList.add("negative");
	} else if (state.genMargin > state.humanMargin) {
		genMarginEl.classList.add("positive");
		humanMarginEl.classList.add("negative");
	}
}

// ============================================
// TUG OF WAR (now based on trial wins)
// ============================================
function updateTugOfWar() {
	const humanScoreEl = document.getElementById("tugHumanScore");
	const genScoreEl = document.getElementById("tugGenScore");
	
	// Display trial wins as the big numbers
	humanScoreEl.textContent = state.humanTrialWins;
	genScoreEl.textContent = state.genTrialWins;
	
	// Show trial count in precise display
	document.getElementById("tugHumanPrecise").textContent = `${state.humanTrialWins} wins`;
	document.getElementById("tugGenPrecise").textContent = `${state.genTrialWins} wins`;
	
	// Update colors based on who's winning
	humanScoreEl.classList.remove("winning", "losing", "tied");
	genScoreEl.classList.remove("winning", "losing", "tied");
	
	if (state.humanTrialWins > state.genTrialWins) {
		humanScoreEl.classList.add("winning");
		genScoreEl.classList.add("losing");
	} else if (state.genTrialWins > state.humanTrialWins) {
		genScoreEl.classList.add("winning");
		humanScoreEl.classList.add("losing");
	} else {
		// Tied
		humanScoreEl.classList.add("tied");
		genScoreEl.classList.add("tied");
	}
}

// ============================================
// RESULT
// ============================================
function showResult(humanScore, genScore) {
	state.judging = false;

	const hPct = humanScore * 100;
	const gPct = genScore * 100;

	// Calculate margin with full precision
	const margin = hPct - gPct;
	const absMargin = Math.abs(margin);
	
	console.log(`[game] Scores: Human=${hPct}%, Gen=${gPct}%, Margin=${margin}`);
	
	// Transfer margin from loser to winner (tug-of-war style)
	if (margin > 0) {
		// Human wins - transfer margin from gen to human
		state.humanPct += absMargin;
		state.genPct -= absMargin;
	} else if (margin < 0) {
		// Gen wins - transfer margin from human to gen
		state.genPct += absMargin;
		state.humanPct -= absMargin;
	}
	// Ties: no transfer
	
	// Clamp to valid range [0, 100]
	state.humanPct = Math.max(0, Math.min(100, state.humanPct));
	state.genPct = Math.max(0, Math.min(100, state.genPct));

	// Choose decimal places based on margin size
	let decimals;
	if (absMargin < 0.01) {
		decimals = 4;
	} else if (absMargin < 0.1) {
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
	
	// Treat very small margins (< 0.0001%) as ties
	const TIE_THRESHOLD = 0.0001;
	
	if (absMargin < TIE_THRESHOLD) {
		resultEl.className = "round-result tie";
		resultEl.textContent = "Tie!";
		humanPanel.classList.add("tie");
		genPanel.classList.add("tie");
	} else if (hPct > gPct) {
		state.humanMargin += absMargin;
		state.humanTrialWins++;
		resultEl.className = "round-result win";
		resultEl.textContent = `You Win! (+${absMargin.toFixed(decimals)}%)`;
		humanPanel.classList.add("winner");
		genPanel.classList.add("loser");
	} else {
		state.genMargin += absMargin;
		state.genTrialWins++;
		resultEl.className = "round-result lose";
		resultEl.textContent = `${opponentName} Wins (-${absMargin.toFixed(decimals)}%)`;
		humanPanel.classList.add("loser");
		genPanel.classList.add("winner");
	}

	// Update UI - show cumulative margins
	updateMarginDisplay();
	updateTugOfWar();

	// Check if round is complete
	if (state.trial >= CONFIG.trialsPerRound) {
		// Round complete - determine winner
		setTimeout(() => {
			endRound();
		}, 1500);
	} else {
		// Show Next button
		document.getElementById("judgeBtn").style.display = "none";
		document.getElementById("nextBtn").style.display = "block";
	}
}

// ============================================
// ROUND MANAGEMENT
// ============================================
function startNewTrial() {
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
	
	// Reset timer
	resetTimer();

	// Reset UI
	document.getElementById("judgeBtn").disabled = true;
	document.getElementById("humanScoreDisplay").textContent = "Draw to compete";
	document.getElementById("humanScoreDisplay").classList.add("waiting");
	document.getElementById("genScoreDisplay").textContent = "Generating...";
	document.getElementById("genScoreDisplay").classList.add("waiting");
	
	// Update trial display
	document.getElementById("trialDisplay").textContent = `${state.trial}/${CONFIG.trialsPerRound}`;
	
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

// Alias for backward compatibility
function startNewRound() {
	startNewTrial();
}

function nextTrial() {
	// Hide Next, show Judge
	document.getElementById("nextBtn").style.display = "none";
	document.getElementById("judgeBtn").style.display = "block";

	state.trial++;
	startNewTrial();
}

function endRound() {
	// Determine round winner based on trial wins
	let roundWinner;
	
	if (state.humanTrialWins > state.genTrialWins) {
		state.humanRounds++;
		roundWinner = 'human';
	} else if (state.genTrialWins > state.humanTrialWins) {
		state.genRounds++;
		roundWinner = 'gen';
	} else {
		roundWinner = 'tie';
	}
	
	// Show round result overlay
	showRoundEndOverlay(roundWinner);
}

function showRoundEndOverlay(winner) {
	const overlay = document.getElementById("resultOverlay");
	const badge = document.getElementById("resultBadge");
	const title = document.getElementById("resultTitle");
	const margin = document.getElementById("resultMargin");
	const humanScore = document.getElementById("resultHumanScore");
	const genScore = document.getElementById("resultGenScore");
	const genLabel = document.getElementById("resultGenLabel");
	
	const opponentName = document.getElementById("mnistDigitToggle").checked ? "MNIST" : "GAN";
	genLabel.textContent = `${opponentName} Score`;
	
	if (winner === 'human') {
		badge.className = "result-badge win";
		badge.textContent = "Victory";
		title.textContent = "You Won the Round!";
		margin.innerHTML = `Final: <span class="positive">${state.humanTrialWins} - ${state.genTrialWins}</span>`;
	} else if (winner === 'gen') {
		badge.className = "result-badge lose";
		badge.textContent = "Defeat";
		title.textContent = `${opponentName} Won the Round`;
		margin.innerHTML = `Final: <span class="negative">${state.humanTrialWins} - ${state.genTrialWins}</span>`;
	} else {
		badge.className = "result-badge tie";
		badge.textContent = "Draw";
		title.textContent = "Round Tied!";
		margin.innerHTML = `Final: ${state.humanTrialWins} - ${state.genTrialWins}`;
	}
	
	// Show trial wins
	genScore.textContent = `${state.genTrialWins} wins`;
	humanScore.textContent = `${state.humanTrialWins} wins`;
	
	overlay.classList.add("show");
}

function startNextRound() {
	// Hide overlay
	document.getElementById("resultOverlay").classList.remove("show");
	
	// Reset for new round
	state.round++;
	state.trial = 1;
	state.humanMargin = 0;
	state.genMargin = 0;
	state.humanTrialWins = 0;
	state.genTrialWins = 0;
	state.humanPct = 50.0;
	state.genPct = 50.0;
	
	// Update displays
	document.getElementById("humanMargin").textContent = "+0.00%";
	document.getElementById("genMargin").textContent = "+0.00%";
	document.getElementById("humanMargin").classList.remove("positive", "negative");
	document.getElementById("genMargin").classList.remove("positive", "negative");
	updateTugOfWar();
	
	// Reset button visibility
	document.getElementById("nextBtn").style.display = "none";
	document.getElementById("judgeBtn").style.display = "block";
	
	startNewTrial();
}

function nextRound() {
	// Alias - now calls nextTrial
	nextTrial();
}

function resetTrial() {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	genCtx.clearRect(0, 0, genCanvas.width, genCanvas.height);

	state.hasDrawn = false;
	
	// Reset timer
	resetTimer();

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

// Alias for backward compatibility  
function resetRound() {
	resetTrial();
}

function resetGame() {
	// Stop any running timer
	stopTimer();
	
	state = {
		round: 1,
		trial: 1,
		digit: null,
		humanMargin: 0,
		genMargin: 0,
		humanTrialWins: 0,
		genTrialWins: 0,
		humanRounds: 0,
		genRounds: 0,
		humanPct: 50.0,
		genPct: 50.0,
		hasDrawn: false,
		judging: false,
		currentDifficulty: state.currentDifficulty,
		timeLimit: state.timeLimit,
		timerInterval: null,
		timeRemaining: state.timeLimit,
	};

	document.getElementById("trialDisplay").textContent = `1/${CONFIG.trialsPerRound}`;
	document.getElementById("humanMargin").textContent = "+0.00%";
	document.getElementById("genMargin").textContent = "+0.00%";
	document.getElementById("humanMargin").classList.remove("positive", "negative");
	document.getElementById("genMargin").classList.remove("positive", "negative");
	document.getElementById("tugHumanScore").textContent = "50%";
	document.getElementById("tugGenScore").textContent = "50%";
	document.getElementById("tugHumanPrecise").textContent = "50.000000%";
	document.getElementById("tugGenPrecise").textContent = "50.000000%";
	
	// Reset timer display
	updateTimerDisplay();
	
	// Reset button visibility
	document.getElementById("nextBtn").style.display = "none";
	document.getElementById("judgeBtn").style.display = "block";
	
	// Hide overlay if showing
	document.getElementById("resultOverlay").classList.remove("show");

	startNewTrial();
}

// ============================================
// BUTTONS
// ============================================
document.getElementById("judgeBtn").onclick = () => {
	if (state.digit === null || !state.hasDrawn || state.judging) return;

	state.judging = true;
	stopTimer();
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
	if (!state.judging) resetTrial();
};

document.getElementById("nextBtn").onclick = nextTrial;
document.getElementById("resultNextBtn").onclick = startNextRound;

// ============================================
// DIFFICULTY SELECTOR
// ============================================
document.querySelectorAll(".difficulty-btn").forEach(btn => {
	btn.addEventListener("click", () => {
		// Update selection
		document.querySelectorAll(".difficulty-btn").forEach(b => b.classList.remove("selected"));
		btn.classList.add("selected");
		
		// Update state using CONFIG values
		const level = btn.dataset.level;
		state.currentDifficulty = level;
		state.timeLimit = CONFIG.difficulty[level];
		state.timeRemaining = state.timeLimit;
		
		// Reset game with new difficulty
		resetGame();
	});
});
