#!/usr/bin/env node

/**
 * shader-server.js
 *
 * Serves .metal shader files from the `shaders/` directory over HTTP (port 8888).
 * DynamicBoxRenderer fetches shaders from this server, compiles them at runtime,
 * and swaps the fragment function in the render pipeline.
 *
 * Endpoints:
 *   GET  /shaders/:name.metal   – serve a shader file
 *   GET  /shaders               – list available shader files (JSON array)
 *   POST /report-error          – log a compile error/success to shader-compiling-error.log
 *   POST /report-perf           – log a sampled performance report to shader-performance.log
 *
 * Usage:
 *   cd vr-dive/Demos/DynamicBox && node shader-server.js
 */

const http = require("http");
const fs   = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const PORT       = 8888;
const SHADERS_DIR = path.join(__dirname, "shaders");
const ERROR_LOG   = path.join(__dirname, "shader-compiling-error.log");
const SERVER_LOG  = path.join(__dirname, "server.log");
const PERF_LOG    = path.join(__dirname, "shader-performance.log");

function gitRepositoryRoot() {
  try {
    return execFileSync("git", ["-C", __dirname, "rev-parse", "--show-toplevel"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

const GIT_ROOT = gitRepositoryRoot();

// ─── Log helper (console + file) ──────────────────────────────────────────────
function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  try { fs.appendFileSync(SERVER_LOG, line + "\n", "utf-8"); } catch {}
}

// ─── Ensure directories exist ─────────────────────────────────────────────────
if (!fs.existsSync(SHADERS_DIR)) {
  fs.mkdirSync(SHADERS_DIR, { recursive: true });
}

// ─── MIME types ───────────────────────────────────────────────────────────────
function extToMIME(ext) {
  const map = { ".metal": "text/plain", ".json": "application/json", ".txt": "text/plain" };
  return map[ext] || "application/octet-stream";
}

function gitTimestamp(args) {
  if (!GIT_ROOT) return Number.POSITIVE_INFINITY;
  try {
    const output = execFileSync("git", ["-C", GIT_ROOT, ...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    if (!output) return Number.POSITIVE_INFINITY;
    const timestamp = Number(output.split("\n")[0]);
    return Number.isFinite(timestamp) ? timestamp * 1000 : Number.POSITIVE_INFINITY;
  } catch {
    return Number.POSITIVE_INFINITY;
  }
}

function compareShaderFiles(left, right) {
  const leftHasGitCreation = Number.isFinite(left.gitAddedAt);
  const rightHasGitCreation = Number.isFinite(right.gitAddedAt);
  if (leftHasGitCreation !== rightHasGitCreation) {
    return leftHasGitCreation ? -1 : 1;
  }
  if (leftHasGitCreation && left.gitAddedAt !== right.gitAddedAt) {
    return left.gitAddedAt - right.gitAddedAt;
  }
  if (left.createdAt !== right.createdAt) return left.createdAt - right.createdAt;

  const leftHasGitModification = Number.isFinite(left.gitModifiedAt);
  const rightHasGitModification = Number.isFinite(right.gitModifiedAt);
  if (leftHasGitModification !== rightHasGitModification) {
    return leftHasGitModification ? -1 : 1;
  }
  if (leftHasGitModification && left.gitModifiedAt !== right.gitModifiedAt) {
    return left.gitModifiedAt - right.gitModifiedAt;
  }
  return left.file.localeCompare(right.file);
}

// Prefer the order files first appeared in Git, then filesystem creation time,
// then the most recent Git modification. This survives copies between machines
// while still giving new, uncommitted shaders a sensible place in the picker.
function listShaderFiles() {
  return fs.readdirSync(SHADERS_DIR)
    .filter(file => file.endsWith(".metal"))
    .map(file => {
      const filePath = path.join(SHADERS_DIR, file);
      const gitPath = GIT_ROOT ? path.relative(GIT_ROOT, filePath) : filePath;
      const createdAt = fs.statSync(filePath).birthtimeMs;
      return {
        file,
        createdAt,
        gitAddedAt: gitTimestamp(["log", "--diff-filter=A", "--format=%ct", "--reverse", "--", gitPath]),
        gitModifiedAt: gitTimestamp(["log", "-1", "--format=%ct", "--", gitPath]),
      };
    })
    .sort(compareShaderFiles)
    .map(entry => entry.file);
}

// ─── Simple router ────────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  // CORS – allow the visionOS app to access from any origin
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;
  log(`[HTTP] ${req.method} ${pathname}`);

  // ── GET /shaders ──────────────────────────────────────────────────────────
  if (req.method === "GET" && pathname === "/shaders") {
    try {
      const files = listShaderFiles().map(file => file.replace(/\.metal$/, ""));
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(files));
    } catch (err) {
      res.writeHead(500);
      res.end("Failed to list shaders");
    }
    return;
  }

  // ── GET /shaders/:name.metal ──────────────────────────────────────────────
  const shaderMatch = pathname.match(/^\/shaders\/([^/]+)\.metal$/);
  if (req.method === "GET" && shaderMatch) {
    const name = shaderMatch[1];
    // Basic path traversal protection
    if (name.includes("..") || name.includes("/")) {
      res.writeHead(400);
      res.end("Invalid shader name");
      return;
    }
    const filePath = path.join(SHADERS_DIR, `${name}.metal`);
    try {
      const content = fs.readFileSync(filePath, "utf-8");
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      res.end(content);
    } catch {
      res.writeHead(404);
      res.end(`Shader "${name}" not found`);
    }
    return;
  }

  // ── POST /report-error ────────────────────────────────────────────────────
  if (req.method === "POST" && pathname === "/report-error") {
    let body = "";
    req.on("data", chunk => { body += chunk; });
    req.on("end", () => {
      const timestamp = new Date().toISOString();
      const isError = body.includes("error") || body.includes("Error") || body.includes("fail");
      const tag = isError ? "ERR" : "OK";
      const entry = `\n=== ${timestamp} [${tag}] ===\n${body}\n`;
      fs.appendFileSync(ERROR_LOG, entry, "utf-8");
      log(`[${tag}] ${body.split("\n")[0]}`);
      res.writeHead(200);
      res.end("OK");
    });
    return;
  }

  // ── POST /report-perf ─────────────────────────────────────────────────────
  // Sampled performance reports from DynamicBoxRenderer (up to ~10 per shader
  // load, only sent for noticeably slow frames). Appended to a dedicated log
  // so they don't get mixed in with compile error/success entries.
  if (req.method === "POST" && pathname === "/report-perf") {
    let body = "";
    req.on("data", chunk => { body += chunk; });
    req.on("end", () => {
      const timestamp = new Date().toISOString();
      const entry = `[${timestamp}] ${body}\n`;
      fs.appendFileSync(PERF_LOG, entry, "utf-8");
      log(`[PERF] ${body.split("\n")[0]}`);
      res.writeHead(200);
      res.end("OK");
    });
    return;
  }

  // ── 404 ───────────────────────────────────────────────────────────────────
  res.writeHead(404);
  res.end("Not found");
});

server.listen(PORT, () => {
  log(`Serving shaders from: ${SHADERS_DIR}`);
  log(`Listening on http://localhost:${PORT}`);
  log(`All logs → ${SERVER_LOG}`);
  log(`Errors     → ${ERROR_LOG}`);
  log(`Perf       → ${PERF_LOG}`);
  log(`Available shaders:`);
  try {
    const files = listShaderFiles();
    if (files.length === 0) {
      log("  (none – add .metal files to the shaders/ directory)");
    } else {
      files.forEach(f => log(`  - ${f}`));
    }
  } catch {
    log("  (could not read shaders directory)");
  }
});
