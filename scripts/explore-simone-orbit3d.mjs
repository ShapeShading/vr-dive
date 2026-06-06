#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const TWO_PI = Math.PI * 2;
const DEFAULT_OUTPUT = 'vr-dive/Demos/SimoneOrbit3D/preset-search-results.json';

const bourkePairs = [
  [3.69, 4.51],
  [5.51, 4.84],
  [3.64, 1.71],
  [5.46, 4.55],
  [0.47, 2.25],
  [0.29, 0.95],
  [2.59, 2.49],
  [0.54, 1.23],
  [0.40, 5.11],
  [2.31, 1.64],
  [0.29, 4.00],
  [5.90, 5.64],
  [3.61, 4.24],
  [2.70, 2.32],
  [2.55, 0.93],
];

function parseArgs(argv) {
  const options = {
    output: DEFAULT_OUTPUT,
    steps: 12000,
    warmup: 400,
    keepTop: 12,
    grid: 22,
    jitter: 0.18,
    cModes: ['half-delta', 'half-sum', 'neg-half-sum', 'a', 'b'],
    seeds: [
      [0.12, -0.09, 0.04],
      [-0.18, 0.06, 0.11],
    ],
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--output') options.output = argv[++i];
    else if (arg === '--steps') options.steps = Number(argv[++i]);
    else if (arg === '--warmup') options.warmup = Number(argv[++i]);
    else if (arg === '--keep-top') options.keepTop = Number(argv[++i]);
    else if (arg === '--grid') options.grid = Number(argv[++i]);
    else if (arg === '--jitter') options.jitter = Number(argv[++i]);
  }

  return options;
}

function hashNoise(index, axis) {
  const x = Math.sin(index * 127.1 + axis * 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

function makeCandidates(options) {
  const candidates = [];

  bourkePairs.forEach(([a, b], pairIndex) => {
    options.cModes.forEach((mode, modeIndex) => {
      const c = deriveC(a, b, mode);
      candidates.push({
        name: `bourke-${pairIndex + 1}-${mode}`,
        source: 'bourke-seed',
        a,
        b,
        c,
      });

      for (let k = 0; k < 3; k += 1) {
        const idx = pairIndex * 17 + modeIndex * 5 + k;
        candidates.push({
          name: `bourke-${pairIndex + 1}-${mode}-j${k + 1}`,
          source: 'bourke-jitter',
          a: wrapAngle(a + (hashNoise(idx, 0) - 0.5) * options.jitter * TWO_PI),
          b: wrapAngle(b + (hashNoise(idx, 1) - 0.5) * options.jitter * TWO_PI),
          c: wrapAngle(c + (hashNoise(idx, 2) - 0.5) * options.jitter * TWO_PI),
        });
      }
    });
  });

  const gridSize = options.grid;
  for (let ix = 0; ix < gridSize; ix += 1) {
    const fx = ix / Math.max(gridSize - 1, 1);
    for (let iy = 0; iy < gridSize; iy += 1) {
      const fy = iy / Math.max(gridSize - 1, 1);
      const a = fx * TWO_PI;
      const b = fy * TWO_PI;
      const c = wrapAngle(0.5 * (a - b));
      candidates.push({
        name: `grid-${ix + 1}-${iy + 1}`,
        source: 'grid-half-delta',
        a,
        b,
        c,
      });
    }
  }

  return candidates;
}

function deriveC(a, b, mode) {
  switch (mode) {
    case 'half-delta':
      return 0.5 * (a - b);
    case 'half-sum':
      return 0.5 * (a + b);
    case 'neg-half-sum':
      return -0.5 * (a + b);
    case 'a':
      return a;
    case 'b':
      return b;
    default:
      return 0.5 * (a - b);
  }
}

function wrapAngle(value) {
  let result = value % TWO_PI;
  if (result < 0) result += TWO_PI;
  return result;
}

function mapSimone3D([x, y, z], params) {
  const { a, b, c } = params;
  return [
    Math.sin(x * x - y * y - z * z + a),
    Math.cos(2 * x * y + b),
    Math.sin(2 * x * z + c),
  ];
}

function sampleTrajectory(params, options) {
  const samplePoints = [];
  const voxelSet = new Set();
  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];
  let sum = [0, 0, 0];
  let sqSum = [0, 0, 0];
  let stepSum = 0;
  let stepSqSum = 0;
  let radiusSum = 0;
  let radiusSqSum = 0;
  let minRadius = Infinity;
  let maxRadius = 0;
  let points = 0;
  let prev;

  const covarianceAcc = [0, 0, 0, 0, 0, 0];

  for (const baseSeed of options.seeds) {
    let state = [...baseSeed];
    for (let i = 0; i < options.warmup; i += 1) {
      state = mapSimone3D(state, params);
    }

    for (let i = 0; i < options.steps; i += 1) {
      state = mapSimone3D(state, params);
      const [x, y, z] = state;
      const radius = Math.hypot(x, y, z);
      minRadius = Math.min(minRadius, radius);
      maxRadius = Math.max(maxRadius, radius);
      radiusSum += radius;
      radiusSqSum += radius * radius;

      min[0] = Math.min(min[0], x);
      min[1] = Math.min(min[1], y);
      min[2] = Math.min(min[2], z);
      max[0] = Math.max(max[0], x);
      max[1] = Math.max(max[1], y);
      max[2] = Math.max(max[2], z);

      sum[0] += x;
      sum[1] += y;
      sum[2] += z;
      sqSum[0] += x * x;
      sqSum[1] += y * y;
      sqSum[2] += z * z;

      covarianceAcc[0] += x * x;
      covarianceAcc[1] += x * y;
      covarianceAcc[2] += x * z;
      covarianceAcc[3] += y * y;
      covarianceAcc[4] += y * z;
      covarianceAcc[5] += z * z;

      const voxel = [x, y, z]
        .map((value) => Math.max(0, Math.min(23, Math.floor(((value + 1) * 0.5) * 24))))
        .join(':');
      voxelSet.add(voxel);

      if (prev) {
        const dx = x - prev[0];
        const dy = y - prev[1];
        const dz = z - prev[2];
        const step = Math.hypot(dx, dy, dz);
        stepSum += step;
        stepSqSum += step * step;
      }
      prev = [x, y, z];

      if (samplePoints.length < 96 && i % Math.max(1, Math.floor(options.steps / 96)) === 0) {
        samplePoints.push([round6(x), round6(y), round6(z)]);
      }
      points += 1;
    }
  }

  const invPoints = 1 / Math.max(points, 1);
  const mean = sum.map((value) => value * invPoints);
  const variance = sqSum.map((value, axis) => Math.max(value * invPoints - mean[axis] * mean[axis], 0));
  const stddev = variance.map((value) => Math.sqrt(value));
  const extents = max.map((value, axis) => value - min[axis]);

  const cov = [
    covarianceAcc[0] * invPoints - mean[0] * mean[0],
    covarianceAcc[1] * invPoints - mean[0] * mean[1],
    covarianceAcc[2] * invPoints - mean[0] * mean[2],
    covarianceAcc[3] * invPoints - mean[1] * mean[1],
    covarianceAcc[4] * invPoints - mean[1] * mean[2],
    covarianceAcc[5] * invPoints - mean[2] * mean[2],
  ];

  const eigenvalues = jacobiEigenvalues([
    [cov[0], cov[1], cov[2]],
    [cov[1], cov[3], cov[4]],
    [cov[2], cov[4], cov[5]],
  ]).sort((lhs, rhs) => rhs - lhs);

  const stepCount = Math.max(points - options.seeds.length, 1);
  const avgStep = stepSum / stepCount;
  const stepVariance = Math.max(stepSqSum / stepCount - avgStep * avgStep, 0);
  const stepStddev = Math.sqrt(stepVariance);
  const radiusMean = radiusSum * invPoints;
  const radiusStddev = Math.sqrt(Math.max(radiusSqSum * invPoints - radiusMean * radiusMean, 0));

  return {
    samplePoints,
    metrics: {
      pointCount: points,
      bboxMin: min.map(round6),
      bboxMax: max.map(round6),
      extents: extents.map(round6),
      mean: mean.map(round6),
      stddev: stddev.map(round6),
      eigenvalues: eigenvalues.map(round6),
      occupiedVoxels: voxelSet.size,
      occupancyRatio: round6(voxelSet.size / (24 * 24 * 24)),
      avgStep: round6(avgStep),
      stepStddev: round6(stepStddev),
      radiusMean: round6(radiusMean),
      radiusStddev: round6(radiusStddev),
      radiusRange: [round6(minRadius), round6(maxRadius)],
    },
  };
}

function jacobiEigenvalues(matrix) {
  const m = matrix.map((row) => [...row]);
  const n = 3;
  for (let iter = 0; iter < 24; iter += 1) {
    let p = 0;
    let q = 1;
    let maxOffDiag = Math.abs(m[0][1]);
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const value = Math.abs(m[i][j]);
        if (value > maxOffDiag) {
          maxOffDiag = value;
          p = i;
          q = j;
        }
      }
    }
    if (maxOffDiag < 1e-10) break;
    const theta = 0.5 * Math.atan2(2 * m[p][q], m[q][q] - m[p][p]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    const app = c * c * m[p][p] - 2 * s * c * m[p][q] + s * s * m[q][q];
    const aqq = s * s * m[p][p] + 2 * s * c * m[p][q] + c * c * m[q][q];
    m[p][p] = app;
    m[q][q] = aqq;
    m[p][q] = 0;
    m[q][p] = 0;

    for (let r = 0; r < n; r += 1) {
      if (r === p || r === q) continue;
      const mrp = c * m[r][p] - s * m[r][q];
      const mrq = s * m[r][p] + c * m[r][q];
      m[r][p] = mrp;
      m[p][r] = mrp;
      m[r][q] = mrq;
      m[q][r] = mrq;
    }
  }
  return [m[0][0], m[1][1], m[2][2]];
}

function scoreCandidate(result) {
  const { extents, stddev, eigenvalues, occupiedVoxels, occupancyRatio, avgStep, stepStddev, radiusStddev } = result.metrics;
  const extentMin = Math.min(...extents);
  const extentMax = Math.max(...extents);
  const axisBalance = extentMin / Math.max(extentMax, 1e-4);
  const linearity = (eigenvalues[0] - eigenvalues[1]) / Math.max(eigenvalues[0], 1e-4);
  const filamentDepth = eigenvalues[1] / Math.max(eigenvalues[0], 1e-4);
  const depthRatio = eigenvalues[2] / Math.max(eigenvalues[0], 1e-4);
  const occupancyScore = bellScore(occupiedVoxels, 220, 880, 1450);
  const depthScore = bellScore(depthRatio, 0.035, 0.11, 0.22);
  const linearityScore = clamp(remap(linearity, 0.12, 0.70), 0, 1);
  const filamentScore = bellScore(filamentDepth, 0.06, 0.22, 0.42);
  const balanceScore = clamp(remap(axisBalance, 0.04, 0.34), 0, 1);
  const stepScore = clamp(remap(avgStep, 0.35, 1.55), 0, 1) * (1.0 - clamp(remap(stepStddev, 0.70, 1.55), 0, 1));
  const spreadScore = bellScore(radiusStddev, 0.08, 0.20, 0.34);
  const stddevFloor = Math.min(...stddev);
  const stddevPenalty = stddevFloor < 0.018 ? 0.35 : stddevFloor < 0.035 ? 0.16 : 0;
  const occupancyPenalty = occupancyRatio > 0.14 ? 0.24 : occupancyRatio > 0.09 ? 0.10 : 0;
  const cloudPenalty = depthRatio > 0.24 ? 0.22 : 0;

  return round6(
    0.24 * occupancyScore +
      0.22 * linearityScore +
      0.18 * filamentScore +
      0.16 * depthScore +
      0.10 * stepScore +
      0.10 * spreadScore +
      0.08 * balanceScore -
      stddevPenalty -
      occupancyPenalty -
      cloudPenalty);
}

function classify(result) {
  const { extents, eigenvalues, occupiedVoxels } = result.metrics;
  const longestAxis = ['x', 'y', 'z'][extents.indexOf(Math.max(...extents))];
  const linearity = (eigenvalues[0] - eigenvalues[1]) / Math.max(eigenvalues[0], 1e-4);
  const depthRatio = eigenvalues[2] / Math.max(eigenvalues[0], 1e-4);
  if (linearity > 0.58 && depthRatio > 0.03) return `filament-${longestAxis}`;
  if (depthRatio < 0.03) return `near-planar-${longestAxis}`;
  if (occupiedVoxels > 1200) return `dense-cloud-${longestAxis}`;
  if (occupiedVoxels > 550) return `woven-volume-${longestAxis}`;
  return `filament-cluster-${longestAxis}`;
}

function dedupeTop(results, keepTop) {
  const preferred = results.filter((candidate) => !candidate.visualClass.startsWith('near-planar'));
  const fallback = results.filter((candidate) => candidate.visualClass.startsWith('near-planar'));
  return takeDedupe(preferred, keepTop).concat(takeDedupe(fallback, keepTop, takeDedupe(preferred, keepTop))).slice(0, keepTop);
}

function takeDedupe(results, keepTop, seed = []) {
  const chosen = [...seed];
  const appended = [];
  for (const candidate of results) {
    const isNearExisting = chosen.some((existing) => {
      const da = Math.abs(existing.a - candidate.a);
      const db = Math.abs(existing.b - candidate.b);
      const dc = Math.abs(existing.c - candidate.c);
      return da < 0.14 && db < 0.14 && dc < 0.18;
    });
    if (!isNearExisting) {
      chosen.push(candidate);
      appended.push(candidate);
    }
    if (chosen.length >= keepTop) break;
  }
  return appended;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function bellScore(value, start, peak, end) {
  if (value <= start || value >= end) return 0;
  if (value === peak) return 1;
  if (value < peak) return (value - start) / Math.max(peak - start, 1e-6);
  return 1 - (value - peak) / Math.max(end - peak, 1e-6);
}

function remap(value, min, max) {
  return (value - min) / Math.max(max - min, 1e-6);
}

function round6(value) {
  return Number(value.toFixed(6));
}

function main() {
  const options = parseArgs(process.argv);
  const candidates = makeCandidates(options);
  const evaluated = [];

  for (const candidate of candidates) {
    const trajectory = sampleTrajectory(candidate, options);
    const score = scoreCandidate(trajectory);
    if (score <= 0) continue;
    evaluated.push({
      ...candidate,
      score,
      visualClass: classify(trajectory),
      ...trajectory,
    });
  }

  evaluated.sort((lhs, rhs) => rhs.score - lhs.score);
  const top = dedupeTop(evaluated, options.keepTop).map((entry, index) => ({
    rank: index + 1,
    id: `preset-${String(index + 1).padStart(2, '0')}`,
    label: `${entry.visualClass} #${index + 1}`,
    a: round6(entry.a),
    b: round6(entry.b),
    c: round6(entry.c),
    source: entry.source,
    seedName: entry.name,
    score: entry.score,
    visualClass: entry.visualClass,
    metrics: entry.metrics,
    samplePoints: entry.samplePoints,
  }));

  const payload = {
    generator: 'scripts/explore-simone-orbit3d.mjs',
    formula: {
      x: "sin(x*x - y*y - z*z + a)",
      y: "cos(2*x*y + b)",
      z: "sin(2*x*z + c)",
    },
    options,
    searchedCandidateCount: candidates.length,
    acceptedCandidateCount: evaluated.length,
    generatedAt: new Date().toISOString(),
    presets: top,
  };

  const outputPath = path.resolve(process.cwd(), options.output);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`);

  const summary = top.map((preset) => ({
    rank: preset.rank,
    label: preset.label,
    a: preset.a,
    b: preset.b,
    c: preset.c,
    score: preset.score,
    voxels: preset.metrics.occupiedVoxels,
    depth: preset.metrics.eigenvalues[2],
  }));
  console.table(summary);
  console.log(`Wrote ${top.length} presets to ${path.relative(process.cwd(), outputPath)}`);
}

main();