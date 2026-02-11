const isProxied = window.location.pathname.startsWith("/gan");
const socketPath = isProxied ? "/gan/socket.io" : "/socket.io";

const socket = io({
  path: socketPath,
  transports: ["websocket"]
});

// DOM elements
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const stepVal = document.getElementById("stepVal");
const gLossVal = document.getElementById("gLossVal");
const dLossVal = document.getElementById("dLossVal");
const fpsVal = document.getElementById("fpsVal");
const elapsedVal = document.getElementById("elapsedVal");
const gridContent = document.getElementById("gridContent");
const benchmarkProgress = document.getElementById("benchmarkProgress");
const currentStrategyName = document.getElementById("currentStrategyName");
const strategyTag = document.getElementById("strategyTag");
const strategyCount = document.getElementById("strategyCount");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const strategyPills = document.getElementById("strategyPills");
const resultsSection = document.getElementById("resultsSection");
const resultsBody = document.getElementById("resultsBody");
const emaToggle = document.getElementById("emaToggle");
const fidVal = document.getElementById("fidVal");
const fidStep = document.getElementById("fidStep");
const fidChartSection = document.getElementById("fidChartSection");

let grid = null;
let cells = [];
let lastFrameTime = 0;

// EMA state
let useEma = false;
let emaAlpha = 0.1; // Smoothing factor (lower = smoother)

// Timer state
let timerStart = null;
let timerInterval = null;
let timerRunning = false;

// Benchmark state
let benchmarkState = {
  strategies: [],
  numSteps: 0,
  currentStrategy: null,
  currentIndex: 0,
  results: []
};

// Chart setup
const chart = echarts.init(document.getElementById("lossChart"), null, { renderer: "svg" });
let gLossData = [];
let dLossData = [];
let gLossDataRaw = []; // Store raw data for toggle
let dLossDataRaw = [];

// FID Chart setup
let fidChart = null;
let fidData = [];

// EMA calculation
function ema(data, alpha = 0.1) {
  if (data.length === 0) return [];
  let smoothed = [];
  let prev = data[0][1];
  for (let [x, y] of data) {
    prev = alpha * y + (1 - alpha) * prev;
    smoothed.push([x, prev]);
  }
  return smoothed;
}

// Update chart with current EMA setting
function updateChartData() {
  if (useEma) {
    gLossData = ema(gLossDataRaw, emaAlpha);
    dLossData = ema(dLossDataRaw, emaAlpha);
  } else {
    gLossData = [...gLossDataRaw];
    dLossData = [...dLossDataRaw];
  }
  chart.setOption({ series: [{ data: gLossData }, { data: dLossData }] });
}

// EMA toggle handler
emaToggle.addEventListener('change', (e) => {
  useEma = e.target.checked;
  updateChartData();
});

const chartOption = {
  backgroundColor: "transparent",
  grid: { top: 40, right: 20, bottom: 36, left: 50 },
  legend: {
    data: ["G Loss", "D Loss"],
    top: 6,
    textStyle: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 10 }
  },
  xAxis: {
    type: "value",
    name: "Step",
    nameLocation: "center",
    nameGap: 24,
    nameTextStyle: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 10 },
    axisLine: { lineStyle: { color: "#1e1e26" } },
    axisLabel: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 9 },
    splitLine: { lineStyle: { color: "#1e1e26" } }
  },
  yAxis: {
    type: "value",
    name: "Loss",
    nameLocation: "center",
    nameGap: 36,
    nameTextStyle: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 10 },
    axisLine: { lineStyle: { color: "#1e1e26" } },
    axisLabel: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 9 },
    splitLine: { lineStyle: { color: "#1e1e26" } }
  },
  series: [
    {
      name: "G Loss",
      type: "line",
      data: gLossData,
      smooth: true,
      symbol: "none",
      lineStyle: { color: "#22c55e", width: 1.5 },
      itemStyle: { color: "#22c55e" }
    },
    {
      name: "D Loss",
      type: "line",
      data: dLossData,
      smooth: true,
      symbol: "none",
      lineStyle: { color: "#ef4444", width: 1.5 },
      itemStyle: { color: "#ef4444" }
    }
  ],
  animation: false
};

chart.setOption(chartOption);
window.addEventListener("resize", () => chart.resize());

// Timer functions
function formatTime(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return [h, m, s].map(v => String(v).padStart(2, "0")).join(":");
}

function startTimer() {
  if (timerRunning) return;
  timerStart = Date.now();
  timerRunning = true;
  timerInterval = setInterval(() => {
    elapsedVal.textContent = formatTime(Date.now() - timerStart);
  }, 200);
}

function stopTimer() {
  if (!timerRunning) return;
  timerRunning = false;
  clearInterval(timerInterval);
}

// Strategy colors
const strategyColors = {
  'bce': '#3b82f6',
  'lsgan': '#22c55e',
  'hinge': '#f97316',
  'wgan-gp': '#8b5cf6'
};

// Update strategy pills
function updateStrategyPills() {
  strategyPills.innerHTML = '';
  benchmarkState.strategies.forEach((s, i) => {
    const pill = document.createElement('span');
    pill.className = 'strategy-pill';
    pill.textContent = s.toUpperCase();
    
    if (i < benchmarkState.currentIndex) {
      pill.classList.add('complete');
    } else if (i === benchmarkState.currentIndex) {
      pill.classList.add('active');
    }
    
    strategyPills.appendChild(pill);
  });
}

// Update results table
function updateResultsTable() {
  if (benchmarkState.results.length === 0) {
    resultsSection.style.display = 'none';
    return;
  }
  
  resultsSection.style.display = 'block';
  
  // Sort by FID
  const sorted = [...benchmarkState.results].sort((a, b) => a.fid - b.fid);
  
  resultsBody.innerHTML = '';
  sorted.forEach((r, i) => {
    const row = document.createElement('tr');
    
    let rankBadge = '';
    if (i === 0) rankBadge = '<span class="rank-badge gold">1</span>';
    else if (i === 1) rankBadge = '<span class="rank-badge silver">2</span>';
    else if (i === 2) rankBadge = '<span class="rank-badge bronze">3</span>';
    
    const isBest = i === 0;
    
    row.innerHTML = `
      <td class="strategy-name">${rankBadge}${r.strategy.toUpperCase()}</td>
      <td class="${isBest ? 'best' : ''}">${r.fid.toFixed(2)}</td>
      <td>${r.kid_mean.toFixed(4)} ± ${r.kid_std.toFixed(4)}</td>
      <td>${r.training_time.toFixed(1)}s</td>
    `;
    
    resultsBody.appendChild(row);
  });
}

// Socket events
socket.on("connect", () => {
  statusDot.classList.add("connected");
  statusText.textContent = "connected";
});

socket.on("disconnect", () => {
  statusDot.classList.remove("connected");
  statusText.textContent = "disconnected";
});

socket.on("benchmark_start", (data) => {
  benchmarkState.strategies = data.strategies;
  benchmarkState.numSteps = data.num_steps;
  benchmarkState.results = [];
  
  benchmarkProgress.style.display = 'block';
  updateStrategyPills();
  startTimer();
});

socket.on("strategy_start", (data) => {
  benchmarkState.currentStrategy = data.strategy;
  benchmarkState.currentIndex = data.index;
  
  currentStrategyName.textContent = data.strategy.toUpperCase();
  strategyTag.textContent = data.strategy.toUpperCase();
  strategyTag.className = 'strategy-tag ' + data.strategy.replace('-', '');
  strategyCount.textContent = `Strategy ${data.index + 1} / ${data.total}`;
  
  // Reset chart data for new strategy
  gLossData = [];
  dLossData = [];
  gLossDataRaw = [];
  dLossDataRaw = [];
  chart.setOption({ series: [{ data: gLossData }, { data: dLossData }] });
  
  // Reset FID data for new strategy
  fidData = [];
  fidVal.textContent = "—";
  fidStep.textContent = "";
  if (fidChart) {
    fidChart.setOption({ series: [{ data: [] }] });
  }
  
  updateStrategyPills();
});

socket.on("strategy_end", (data) => {
  benchmarkState.results.push(data);
  updateResultsTable();
});

function ensureGrid(count) {
  if (grid && cells.length === count) return;

  grid = document.createElement("div");
  grid.className = "image-grid";

  // CSS controls grid-template-columns (4 columns)

  cells = [];
  for (let i = 0; i < count; i++) {
    const cell = document.createElement("div");
    cell.className = "image-cell";

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d");

    const label = document.createElement("span");
    label.className = "cell-label";

    cell.appendChild(canvas);
    cell.appendChild(label);
    grid.appendChild(cell);
    cells.push({ el: cell, canvas, ctx, label });
  }

  gridContent.innerHTML = "";
  gridContent.appendChild(grid);
}

socket.on("batch", (data) => {
  const { step, g_loss, d_loss, frames, num_steps, progress } = data;

  startTimer();

  // Stats
  stepVal.textContent = step.toLocaleString();
  gLossVal.textContent = g_loss.toFixed(4);
  dLossVal.textContent = d_loss.toFixed(4);

  // Progress - also show the progress section if hidden
  if (num_steps) {
    benchmarkProgress.style.display = 'block';
    progressBar.style.width = progress + '%';
    progressText.textContent = `Step ${step.toLocaleString()} / ${num_steps.toLocaleString()}`;
  }

  // FPS
  const now = performance.now();
  if (lastFrameTime > 0) {
    const fps = 1000 / (now - lastFrameTime);
    fpsVal.textContent = fps.toFixed(1);
  }
  lastFrameTime = now;

  // Chart data (downsample for performance)
  if (step % 10 === 0) {
    gLossDataRaw.push([step, g_loss]);
    dLossDataRaw.push([step, d_loss]);
    
    // Limit data points
    if (gLossDataRaw.length > 1500) {
      gLossDataRaw = gLossDataRaw.filter((_, i) => i % 2 === 0);
      dLossDataRaw = dLossDataRaw.filter((_, i) => i % 2 === 0);
    }
    
    updateChartData();
  }

  // Grid
  ensureGrid(frames.length);

  frames.forEach((frame) => {
    const { index, label, image } = frame;
    const cell = cells[index];
    if (!cell) return;

    const gray = new Uint8Array(image);
    const rgba = new Uint8ClampedArray(28 * 28 * 4);

    for (let i = 0; i < 784; i++) {
      const v = gray[i];
      rgba[i * 4 + 0] = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = 255;
    }

    const imgData = new ImageData(rgba, 28, 28);
    cell.ctx.putImageData(imgData, 0, 0);

    cell.label.textContent = label;

    cell.el.classList.add("fresh");
    setTimeout(() => cell.el.classList.remove("fresh"), 300);
  });
});

socket.on("done", () => {
  stopTimer();
  
  const badge = document.createElement("span");
  badge.className = "done-badge";
  badge.textContent = "Complete";
  elapsedVal.parentElement.appendChild(badge);
  
  progressBar.style.width = '100%';
  progressText.textContent = 'Benchmark Complete';
});

// FID update handler
socket.on("fid_update", (data) => {
  const { step, fid } = data;
  
  // Show FID display
  document.getElementById("fidDisplay").style.display = 'block';
  
  // Update display
  fidVal.textContent = fid.toFixed(2);
  fidStep.textContent = `@ step ${step.toLocaleString()}`;
  
  // Add to chart data
  fidData.push([step, fid]);
  
  // Initialize FID chart if needed
  if (!fidChart) {
    fidChartSection.style.display = 'block';
    fidChart = echarts.init(document.getElementById("fidChart"), null, { renderer: "svg" });
    
    fidChart.setOption({
      backgroundColor: "transparent",
      grid: { top: 20, right: 20, bottom: 36, left: 50 },
      xAxis: {
        type: "value",
        name: "Step",
        nameLocation: "center",
        nameGap: 24,
        nameTextStyle: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 10 },
        axisLine: { lineStyle: { color: "#1e1e26" } },
        axisLabel: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 9 },
        splitLine: { lineStyle: { color: "#1e1e26" } }
      },
      yAxis: {
        type: "value",
        name: "FID",
        nameLocation: "center",
        nameGap: 36,
        nameTextStyle: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 10 },
        axisLine: { lineStyle: { color: "#1e1e26" } },
        axisLabel: { color: "#5a5a6a", fontFamily: "JetBrains Mono", fontSize: 9 },
        splitLine: { lineStyle: { color: "#1e1e26" } }
      },
      series: [{
        type: "line",
        data: fidData,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        lineStyle: { color: "#06b6d4", width: 2 },
        itemStyle: { color: "#06b6d4" }
      }],
      animation: false
    });
    
    window.addEventListener("resize", () => fidChart.resize());
  } else {
    fidChart.setOption({ series: [{ data: fidData }] });
  }
});
