const API = "/api";
const CLASS_COLORS = ["#ff5fd1", "#35d9ff", "#ffd166", "#6cff9e"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const appState = {
  step: 1,
  maxStepReached: 1,
  datasetId: null,
  algorithm: null,
  trainResponse: null,
};

// ---------------------------------------------------------------------------
// Shared data-space <-> canvas-pixel transform
// ---------------------------------------------------------------------------

function makeTransform(bounds, width, height) {
  const scaleX = width / (bounds.x_max - bounds.x_min);
  const scaleY = height / (bounds.y_max - bounds.y_min);
  return {
    scaleX,
    scaleY,
    toCanvas(x, y) {
      return { cx: (x - bounds.x_min) * scaleX, cy: height - (y - bounds.y_min) * scaleY };
    },
    toData(cx, cy) {
      return { x: bounds.x_min + cx / scaleX, y: bounds.y_min + (height - cy) / scaleY };
    },
  };
}

function canvasClickToPixels(canvas, evt) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return { cx: (evt.clientX - rect.left) * sx, cy: (evt.clientY - rect.top) * sy };
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function boundsFromPoints(points, padRatio = 0.1) {
  const all = [...points.train, ...points.test];
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const padX = (xMax - xMin) * padRatio, padY = (yMax - yMin) * padRatio;
  return { x_min: xMin - padX, x_max: xMax + padX, y_min: yMin - padY, y_max: yMax + padY };
}

// ---------------------------------------------------------------------------
// Navigation between the 4 pages
// ---------------------------------------------------------------------------

function goToStep(n) {
  if (n > appState.maxStepReached + 1) return;
  appState.step = n;
  appState.maxStepReached = Math.max(appState.maxStepReached, n);

  document.querySelectorAll(".page").forEach((el) => el.classList.remove("active"));
  const ids = ["page-intro", "page-setup", "page-train", "page-results"];
  document.getElementById(ids[n - 1]).classList.add("active");

  document.querySelectorAll(".stepper .step").forEach((el) => {
    const step = Number(el.dataset.step);
    el.classList.toggle("active", step === n);
    el.classList.toggle("done", step < n || (step <= appState.maxStepReached && step !== n));
  });

  if (n === 1) startIntroLoop(); else stopIntroLoop();
}

function bindStepperClicks() {
  document.querySelectorAll(".stepper .step").forEach((el) => {
    el.addEventListener("click", () => {
      const step = Number(el.dataset.step);
      if (step <= appState.maxStepReached) goToStep(step);
    });
  });
}

// ---------------------------------------------------------------------------
// PAGE 1 — visual-only intro (attract-mode animation)
// ---------------------------------------------------------------------------

const INTRO_FUNCS = [
  { label: "f(x) = x", fn: (x) => 0.5 * x },
  { label: "f(x) = x²", fn: (x) => 0.09 * x * x - 5 },
  { label: "f(x) = sin(x)", fn: (x) => 5 * Math.sin(x * 0.4) },
];

let introFrameHandle = null;

function startIntroLoop() {
  if (introFrameHandle) return;
  const canvas = document.getElementById("introCanvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height;
  const captionEl = document.getElementById("introCaption");

  let funcIndex = 0;
  let points = [];
  let animStart = performance.now();
  const cycleMs = 4200;

  function toCanvas(x, y) {
    return { cx: width / 2 + x * ((width * 0.85) / 24), cy: height / 2 - y * ((height * 0.85) / 16) };
  }

  function regenerate() {
    const f = INTRO_FUNCS[funcIndex];
    points = [];
    for (let i = 0; i < 70; i++) {
      const x = (Math.random() - 0.5) * 24;
      const y = (Math.random() - 0.5) * 16;
      const label = y > f.fn(x) ? 0 : 1;
      points.push({ x, y, label });
    }
    captionEl.textContent = f.label;
  }
  regenerate();

  function draw(now) {
    const elapsed = now - animStart;
    const curveT = Math.min(1, elapsed / 900);
    const pointsT = Math.min(1, Math.max(0, elapsed - 500) / 900);
    const f = INTRO_FUNCS[funcIndex];

    ctx.clearRect(0, 0, width, height);

    ctx.beginPath();
    const steps = 200;
    const visible = Math.floor(steps * curveT);
    for (let i = 0; i <= visible; i++) {
      const x = -12 + (24 * i) / steps;
      const y = f.fn(x);
      const { cx, cy } = toCanvas(x, y);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.strokeStyle = "#35d9ff";
    ctx.lineWidth = 3;
    ctx.shadowColor = "#35d9ff";
    ctx.shadowBlur = 14;
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (pointsT > 0) {
      points.forEach((p, idx) => {
        const stagger = (idx / points.length) * 0.6;
        const localT = Math.min(1, Math.max(0, (pointsT - stagger) / 0.4));
        if (localT <= 0) return;
        const { cx, cy } = toCanvas(p.x, p.y);
        ctx.globalAlpha = localT;
        ctx.fillStyle = CLASS_COLORS[p.label];
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      });
    }

    if (elapsed >= cycleMs) {
      funcIndex = (funcIndex + 1) % INTRO_FUNCS.length;
      regenerate();
      animStart = now;
    }
    introFrameHandle = requestAnimationFrame(draw);
  }
  introFrameHandle = requestAnimationFrame(draw);
}

function stopIntroLoop() {
  if (introFrameHandle) cancelAnimationFrame(introFrameHandle);
  introFrameHandle = null;
}

// ---------------------------------------------------------------------------
// PAGE 2 — dataset preview + algorithm tiers + hyperparameters
// ---------------------------------------------------------------------------

async function loadDatasets() {
  const res = await fetch(`${API}/datasets`);
  const datasets = await res.json();
  const select = document.getElementById("dataset");
  select.innerHTML = "";
  datasets.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d.id;
    opt.textContent = d.description;
    select.appendChild(opt);
  });
  appState.datasetId = datasets[0].id;
  await refreshPreview();
}

async function refreshPreview() {
  const res = await fetch(`${API}/dataset_preview?dataset_id=${appState.datasetId}`);
  const data = await res.json();
  const bounds = boundsFromPoints(data.points);
  drawScatter(document.getElementById("previewCanvas"), data.points, data.class_names, bounds);

  const legend = document.getElementById("previewLegend");
  legend.innerHTML = "";
  data.class_names.forEach((name, i) => {
    const span = document.createElement("span");
    span.innerHTML = `<span class="swatch" style="background:${CLASS_COLORS[i]}"></span>${name}`;
    legend.appendChild(span);
  });
}

function drawScatter(canvas, points, classNames, bounds) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const t = makeTransform(bounds, canvas.width, canvas.height);
  const plot = (pts, shape) => {
    pts.forEach((p) => {
      const { cx, cy } = t.toCanvas(p.x, p.y);
      const idx = classNames.indexOf(p.label);
      ctx.fillStyle = CLASS_COLORS[idx % CLASS_COLORS.length];
      ctx.strokeStyle = "#05060f";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (shape === "circle") ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      else ctx.rect(cx - 3.5, cy - 3.5, 7, 7);
      ctx.fill();
      ctx.stroke();
    });
  };
  plot(points.train, "circle");
  plot(points.test, "square");
}

// Ordered from most basic to most powerful/complex — each algorithm carries
// its own "locked" flag (implemented vs. coming soon), independent of tier.
const ALGO_TIERS = [
  {
    tier: "Tier 1 — Instance-Based",
    algos: [
      { id: "knn", name: "K-Nearest Neighbors", locked: false, blurb: "Classifies a point by asking its k closest training points what class they are. No real training step at all." },
    ],
  },
  {
    tier: "Tier 2 — Rule-Based",
    algos: [
      { id: "decision_tree", name: "Decision Tree", locked: false, blurb: "Classifies by asking a sequence of yes/no questions about the features, learned from the data." },
    ],
  },
  {
    tier: "Tier 3 — Ensembles",
    algos: [
      { id: "random_forest", name: "Random Forest", locked: true, blurb: "Combines many decision trees and lets them vote together for a sturdier answer." },
      { id: "adaboost", name: "AdaBoost", locked: true, blurb: "Chains weak learners together, each one fixing the last one's mistakes." },
    ],
  },
  {
    tier: "Tier 4 — Optimization-Based",
    algos: [
      { id: "logreg", name: "Logistic / Polynomial Regression", locked: true, blurb: "Fits a curve by gradient descent on a loss surface instead of memorizing or splitting." },
      { id: "svm", name: "Support Vector Machine", locked: true, blurb: "Finds the boundary with the widest possible margin between classes." },
    ],
  },
  {
    tier: "Tier 5 — Deep Learning",
    algos: [
      { id: "neural_net", name: "Neural Network", locked: true, blurb: "Layers of tiny functions combine to warp the decision boundary into almost any shape." },
    ],
  },
];

function renderTiers() {
  const container = document.getElementById("tierContainer");
  container.innerHTML = "";
  ALGO_TIERS.forEach((group) => {
    const allLocked = group.algos.every((a) => a.locked);
    const groupEl = document.createElement("div");
    groupEl.className = "tier-group";
    groupEl.innerHTML = `<div class="tier-label">${group.tier}${allLocked ? " · 🔒 coming soon to this stall" : ""}</div>`;
    const cards = document.createElement("div");
    cards.className = "tier-cards";
    group.algos.forEach((algo) => {
      const card = document.createElement("div");
      card.className = "algo-card" + (algo.locked ? " locked" : "");
      card.dataset.algo = algo.id;
      card.innerHTML = `<h4>${algo.name}${algo.locked ? " 🔒" : ""}</h4><p>${algo.blurb}</p>`;
      if (!algo.locked) {
        card.addEventListener("click", () => selectAlgorithm(algo.id));
      } else {
        card.addEventListener("click", () => {
          card.classList.remove("shake");
          requestAnimationFrame(() => card.classList.add("shake"));
        });
      }
      cards.appendChild(card);
    });
    groupEl.appendChild(cards);
    container.appendChild(groupEl);
  });
}

function selectAlgorithm(algorithmId) {
  appState.algorithm = algorithmId;
  document.querySelectorAll(".algo-card").forEach((c) => c.classList.toggle("selected", c.dataset.algo === algorithmId));
  applyAlgorithmUI(algorithmId);
  document.getElementById("toTrainBtn").disabled = false;
}

function applyAlgorithmUI(algorithm) {
  document.getElementById("dtControls").style.display = algorithm === "decision_tree" ? "block" : "none";
  document.getElementById("knnControls").style.display = algorithm === "knn" ? "block" : "none";
  document.getElementById("secondaryTitle").textContent = algorithm === "decision_tree" ? "Tree Structure" : "Neighbor Radar (auto test-set scan)";
  document.getElementById("treeSvg").style.display = algorithm === "decision_tree" ? "block" : "none";
  document.getElementById("radarCanvas").style.display = algorithm === "knn" ? "block" : "none";
  document.getElementById("scanReadout").style.display = "none";
}

function bindRangeDisplay(rangeId, labelId) {
  const range = document.getElementById(rangeId);
  const label = document.getElementById(labelId);
  range.addEventListener("input", () => (label.textContent = range.value));
}

// ---------------------------------------------------------------------------
// Live math console (page 3)
// ---------------------------------------------------------------------------

let seenTreeNodeKeys = new Set();

function consoleReset() {
  document.getElementById("consoleBody").innerHTML = "";
  seenTreeNodeKeys = new Set();
}

function consoleLog(text, dim = false) {
  const body = document.getElementById("consoleBody");
  const line = document.createElement("div");
  line.className = "console-line" + (dim ? " dim" : "");
  line.textContent = text;
  body.appendChild(line);
  while (body.children.length > 60) body.removeChild(body.firstChild);
  body.scrollTop = body.scrollHeight;
}

function logNewTreeSplits(tree) {
  function walk(node) {
    if (node.is_leaf) return;
    const key = `${node.feature}|${node.threshold}|${node.samples}`;
    if (!seenTreeNodeKeys.has(key)) {
      seenTreeNodeKeys.add(key);
      const [left, right] = node.children;
      const weighted = (left.samples * left.impurity + right.samples * right.impurity) / node.samples;
      const gain = node.impurity - weighted;
      consoleLog(`split on '${node.feature}' <= ${node.threshold}   (n=${node.samples})`);
      consoleLog(`  impurity ${node.impurity.toFixed(3)} -> weighted ${weighted.toFixed(3)}  |  info gain = ${gain.toFixed(4)}`, true);
    }
    node.children.forEach(walk);
  }
  walk(tree);
}

// ---------------------------------------------------------------------------
// Decision boundary canvas (used on page 3 and page 4)
// ---------------------------------------------------------------------------

function drawBoundary(canvas, boundary, points, classNames) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  const res = boundary.resolution;
  const cellW = width / res;
  const cellH = height / res;

  for (let row = 0; row < res; row++) {
    for (let col = 0; col < res; col++) {
      const classIdx = boundary.grid[row][col];
      ctx.fillStyle = hexToRgba(CLASS_COLORS[classIdx % CLASS_COLORS.length], 0.28);
      const canvasRow = res - 1 - row;
      ctx.fillRect(col * cellW, canvasRow * cellH, cellW + 1, cellH + 1);
    }
  }

  const t = makeTransform(boundary, width, height);
  const plot = (pts, shape) => {
    pts.forEach((p) => {
      const { cx, cy } = t.toCanvas(p.x, p.y);
      const idx = classNames.indexOf(p.label);
      ctx.fillStyle = CLASS_COLORS[idx % CLASS_COLORS.length];
      ctx.strokeStyle = "#05060f";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (shape === "circle") ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      else ctx.rect(cx - 3.5, cy - 3.5, 7, 7);
      ctx.fill();
      ctx.stroke();
    });
  };
  plot(points.train, "circle");
  plot(points.test, "square");
}

// ---------------------------------------------------------------------------
// Tree SVG rendering
// ---------------------------------------------------------------------------

function layoutTree(root) {
  let leafCounter = 0;
  let maxDepth = 0;
  function assign(node, depth) {
    node._depth = depth;
    maxDepth = Math.max(maxDepth, depth);
    if (node.is_leaf) {
      node._x = leafCounter;
      leafCounter += 1;
    } else {
      assign(node.children[0], depth + 1);
      assign(node.children[1], depth + 1);
      node._x = (node.children[0]._x + node.children[1]._x) / 2;
    }
  }
  assign(root, 0);
  return { leafCount: leafCounter, maxDepth };
}

function drawTree(svg, root, classNames) {
  const { leafCount, maxDepth } = layoutTree(root);
  const width = svg.clientWidth || 440;
  const height = svg.clientHeight || 380;
  const marginX = 30, marginY = 24;
  const spacing = (width - 2 * marginX) / Math.max(leafCount - 1, 1);
  const levelHeight = (height - 2 * marginY) / Math.max(maxDepth, 1);

  const px = (node) => marginX + node._x * spacing;
  const py = (node) => marginY + node._depth * levelHeight;

  let svgContent = "";

  function edges(node) {
    if (node.is_leaf) return;
    node.children.forEach((child) => {
      svgContent += `<line x1="${px(node)}" y1="${py(node)}" x2="${px(child)}" y2="${py(child)}"
        stroke="rgba(255,255,255,0.25)" stroke-width="1.5" />`;
      edges(child);
    });
  }
  edges(root);

  function nodes(node) {
    const idx = classNames.indexOf(node.prediction);
    const color = CLASS_COLORS[idx % CLASS_COLORS.length];
    const r = node.is_leaf ? 9 : 11;
    svgContent += `<circle cx="${px(node)}" cy="${py(node)}" r="${r}"
      fill="${node.is_leaf ? color : "#161832"}" stroke="${color}" stroke-width="2.5" />`;
    if (!node.is_leaf) {
      svgContent += `<text x="${px(node)}" y="${py(node) - 15}" fill="#c9c1ff" font-size="9"
        text-anchor="middle">${node.feature}</text>`;
      svgContent += `<text x="${px(node)}" y="${py(node) - 5}" fill="#c9c1ff" font-size="9"
        text-anchor="middle">&le; ${node.threshold}</text>`;
      node.children.forEach(nodes);
    } else {
      svgContent += `<text x="${px(node)}" y="${py(node) + 22}" fill="${color}" font-size="9"
        text-anchor="middle">${node.prediction}</text>`;
    }
  }
  nodes(root);

  svg.innerHTML = svgContent;
}

// ---------------------------------------------------------------------------
// KNN "Neighbor Radar" — after training, automatically sweeps through the
// held-out test set, classifying each point by its k nearest neighbors and
// marking hits/misses live on the boundary canvas.
// ---------------------------------------------------------------------------

let trainedState = null;

function classify(queryX, queryY, trainPoints, k, weights, classNames) {
  const withDist = trainPoints.map((p) => {
    const dx = p.x - queryX, dy = p.y - queryY;
    return { ...p, dx, dy, dist: Math.sqrt(dx * dx + dy * dy) };
  });
  withDist.sort((a, b) => a.dist - b.dist);
  const neighbors = withDist.slice(0, k);

  const tally = {};
  classNames.forEach((c) => (tally[c] = 0));
  neighbors.forEach((n) => {
    tally[n.label] += weights === "distance" ? 1 / (n.dist + 1e-6) : 1;
  });

  let predicted = classNames[0], best = -Infinity;
  classNames.forEach((c) => { if (tally[c] > best) { best = tally[c]; predicted = c; } });

  return { neighbors, tally, predicted };
}

function getBaseCanvas(boundary, points, classNames) {
  const base = document.createElement("canvas");
  base.width = 440;
  base.height = 380;
  drawBoundary(base, boundary, points, classNames);
  return base;
}

async function runScanTestSet() {
  const readout = document.getElementById("scanReadout");
  readout.style.display = "block";

  const canvas = document.getElementById("radarCanvas");
  const ctx = canvas.getContext("2d");
  const t = makeTransform(trainedState.bounds, canvas.width, canvas.height);
  ctx.drawImage(trainedState.baseCanvas, 0, 0);
  consoleLog("scanning full test set...");

  let correct = 0;
  const testPoints = trainedState.points.test;
  for (let i = 0; i < testPoints.length; i++) {
    const p = testPoints[i];
    const result = classify(p.x, p.y, trainedState.points.train, trainedState.k, trainedState.weights, trainedState.classNames);
    const hit = result.predicted === p.label;
    if (hit) correct += 1;

    const { cx, cy } = t.toCanvas(p.x, p.y);
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.strokeStyle = hit ? "#6cff9e" : "#ff5f7a";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    const runningPct = ((correct / (i + 1)) * 100).toFixed(1);
    readout.innerHTML = `Scanned ${i + 1}/${testPoints.length} — ` +
      `<span class="${hit ? "hit" : "miss"}">${hit ? "hit" : "miss"}</span> — running accuracy: ${runningPct}%`;
    await sleep(70);
  }

  consoleLog(`scan complete: ${correct}/${testPoints.length} correct`);
  readout.innerHTML = `Scan complete: ${correct}/${testPoints.length} correct — ` +
    `official test accuracy: ${(trainedState.officialAccuracy * 100).toFixed(1)}%`;
}

// ---------------------------------------------------------------------------
// Training flow (page 3)
// ---------------------------------------------------------------------------

function animateNumber(el, from, to, ms) {
  const start = performance.now();
  function step(now) {
    const t = Math.min(1, (now - start) / ms);
    const val = from + (to - from) * t;
    el.textContent = `${(val * 100).toFixed(1)}%`;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function renderAdvice(advice) {
  const container = document.getElementById("adviceCards");
  container.innerHTML = "";
  advice.forEach((a) => {
    const div = document.createElement("div");
    div.className = `advice-card ${a.type}`;
    div.textContent = a.text;
    container.appendChild(div);
  });
}

async function runTraining() {
  document.getElementById("toResultsBtn").disabled = true;
  document.getElementById("adviceCards").innerHTML = "";
  document.getElementById("scanReadout").style.display = "none";
  consoleReset();
  trainedState = null;

  const algorithm = appState.algorithm;
  const datasetId = appState.datasetId;
  const payload = {
    dataset_id: datasetId,
    algorithm,
    max_depth: Number(document.getElementById("maxDepth").value),
    min_samples_split: Number(document.getElementById("minSplit").value),
    criterion: document.getElementById("criterion").value,
    n_neighbors: Number(document.getElementById("kNeighbors").value),
    weights: document.getElementById("weights").value,
    player_name: (document.getElementById("player").value || "AAA").toUpperCase(),
  };

  consoleLog(`training ${algorithm === "knn" ? "K-Nearest Neighbors" : "Decision Tree"} on '${datasetId}'...`);

  const res = await fetch(`${API}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  appState.trainResponse = data;

  const canvas = document.getElementById("boundaryCanvas");
  const svg = document.getElementById("treeSvg");
  const accuracyEl = document.getElementById("accuracyDisplay");
  const depthEl = document.getElementById("depthIndicator");
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  data.class_names.forEach((name, i) => {
    const span = document.createElement("span");
    span.innerHTML = `<span class="swatch" style="background:${CLASS_COLORS[i]}"></span>${name}`;
    legend.appendChild(span);
  });

  let prevAcc = 0;
  for (const snap of data.snapshots) {
    drawBoundary(canvas, snap.boundary, data.points, data.class_names);
    if (algorithm === "decision_tree") {
      drawTree(svg, snap.tree, data.class_names);
      logNewTreeSplits(snap.tree);
    } else {
      consoleLog(`k=${snap.k} -> recomputing distances for every point... test accuracy ${(snap.test_accuracy * 100).toFixed(1)}%`, true);
    }
    depthEl.textContent = `Step: ${snap.step_label}`;
    animateNumber(accuracyEl, prevAcc, snap.test_accuracy, 400);
    prevAcc = snap.test_accuracy;
    await sleep(algorithm === "knn" ? 220 : 700);
  }

  renderAdvice(data.advice);
  consoleLog(`training complete — final test accuracy ${(data.final.test_accuracy * 100).toFixed(1)}%`);

  if (algorithm === "knn") {
    const finalSnap = data.final;
    trainedState = {
      algorithm,
      classNames: data.class_names,
      points: data.points,
      bounds: finalSnap.boundary,
      k: finalSnap.k,
      weights: finalSnap.weights,
      officialAccuracy: finalSnap.test_accuracy,
      baseCanvas: getBaseCanvas(finalSnap.boundary, data.points, data.class_names),
    };
    const radarCanvas = document.getElementById("radarCanvas");
    radarCanvas.getContext("2d").drawImage(trainedState.baseCanvas, 0, 0);
    await runScanTestSet();
  }

  document.getElementById("toResultsBtn").disabled = false;
}

// ---------------------------------------------------------------------------
// PAGE 4 — results, leaderboard, random point dropper
// ---------------------------------------------------------------------------

function predictFromGrid(x, y, boundary) {
  const col = Math.round(((x - boundary.x_min) / (boundary.x_max - boundary.x_min)) * (boundary.resolution - 1));
  const row = Math.round(((y - boundary.y_min) / (boundary.y_max - boundary.y_min)) * (boundary.resolution - 1));
  const c = Math.min(boundary.resolution - 1, Math.max(0, col));
  const r = Math.min(boundary.resolution - 1, Math.max(0, row));
  return boundary.grid[r][c];
}

let droppedRandomPoints = [];

function redrawResultCanvas() {
  const data = appState.trainResponse;
  const canvas = document.getElementById("resultCanvas");
  drawBoundary(canvas, data.final.boundary, data.points, data.class_names);
  const t = makeTransform(data.final.boundary, canvas.width, canvas.height);
  const ctx = canvas.getContext("2d");
  droppedRandomPoints.forEach((p) => {
    const { cx, cy } = t.toCanvas(p.x, p.y);
    const color = CLASS_COLORS[data.class_names.indexOf(p.predictedClass) % CLASS_COLORS.length];
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - 5, cy);
    ctx.lineTo(cx + 5, cy);
    ctx.moveTo(cx, cy - 5);
    ctx.lineTo(cx, cy + 5);
    ctx.strokeStyle = "#05060f";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

function placePoint(x, y) {
  const data = appState.trainResponse;
  const boundary = data.final.boundary;
  const classIdx = predictFromGrid(x, y, boundary);
  const predictedClass = data.class_names[classIdx];

  droppedRandomPoints.push({ x, y, predictedClass });
  if (droppedRandomPoints.length > 12) droppedRandomPoints.shift();
  redrawResultCanvas();

  document.getElementById("randomPointReadout").innerHTML =
    `Point at (${x.toFixed(2)}, ${y.toFixed(2)}) → predicted <b style="color:${CLASS_COLORS[classIdx % CLASS_COLORS.length]}">${predictedClass}</b>`;
}

function dropRandomPoint() {
  const boundary = appState.trainResponse.final.boundary;
  const x = boundary.x_min + Math.random() * (boundary.x_max - boundary.x_min);
  const y = boundary.y_min + Math.random() * (boundary.y_max - boundary.y_min);
  placePoint(x, y);
}

function handleResultCanvasClick(evt) {
  if (!appState.trainResponse) return;
  const canvas = document.getElementById("resultCanvas");
  const { cx, cy } = canvasClickToPixels(canvas, evt);
  const t = makeTransform(appState.trainResponse.final.boundary, canvas.width, canvas.height);
  const { x, y } = t.toData(cx, cy);
  placePoint(x, y);
}

async function renderResultsPage() {
  const data = appState.trainResponse;
  document.getElementById("finalAcc").textContent = `${(data.final.test_accuracy * 100).toFixed(1)}%`;
  document.getElementById("rankLine").textContent =
    data.leaderboard_rank ? `Rank #${data.leaderboard_rank} of ${data.leaderboard_total} on this leaderboard` : "";

  droppedRandomPoints = [];
  redrawResultCanvas();
  document.getElementById("randomPointReadout").textContent = "";

  const params = new URLSearchParams({ dataset_id: appState.datasetId, algorithm: appState.algorithm });
  const res = await fetch(`${API}/leaderboard?${params.toString()}`);
  const rows = await res.json();
  const body = document.getElementById("leaderboardBody");
  body.innerHTML = "";
  rows.forEach((row, i) => {
    const tr = document.createElement("tr");
    const isOwn = row.player_name === data.entry.player_name &&
      Math.abs(row.timestamp - data.entry.timestamp) < 0.001;
    if (isOwn) tr.className = "own-row";
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${row.player_name}</td>
      <td>${row.dataset_id}</td>
      <td>${row.algorithm}</td>
      <td>${row.params_summary}</td>
      <td>${(row.test_accuracy * 100).toFixed(1)}%</td>`;
    body.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

async function init() {
  bindStepperClicks();
  await loadDatasets();
  renderTiers();
  applyAlgorithmUI(null);

  bindRangeDisplay("maxDepth", "maxDepthVal");
  bindRangeDisplay("minSplit", "minSplitVal");
  bindRangeDisplay("kNeighbors", "kNeighborsVal");

  document.getElementById("dataset").addEventListener("change", (e) => {
    appState.datasetId = e.target.value;
    refreshPreview();
  });

  document.getElementById("startBtn").addEventListener("click", () => goToStep(2));
  document.getElementById("backToIntro").addEventListener("click", () => goToStep(1));
  document.getElementById("toTrainBtn").addEventListener("click", async () => {
    goToStep(3);
    await runTraining();
  });
  document.getElementById("backToSetup").addEventListener("click", () => goToStep(2));
  document.getElementById("toResultsBtn").addEventListener("click", async () => {
    goToStep(4);
    await renderResultsPage();
  });
  document.getElementById("restartTutorial").addEventListener("click", () => {
    appState.maxStepReached = 1;
    goToStep(1);
  });
  document.getElementById("playAgainBtn").addEventListener("click", () => {
    appState.maxStepReached = Math.max(appState.maxStepReached, 2);
    goToStep(2);
  });

  document.getElementById("randomPointBtn").addEventListener("click", dropRandomPoint);
  document.getElementById("resultCanvas").addEventListener("click", handleResultCanvasClick);

  goToStep(1);
}

init();
