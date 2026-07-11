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
 *   POST /report-error          – log a compilation error to shader-compiling-error.log
 *
 * Usage:
 *   cd vr-dive/Demos/DynamicBox && node shader-server.js
 */

const http = require("http");
const fs   = require("fs");
const path = require("path");

const PORT       = 8888;
const SHADERS_DIR = path.join(__dirname, "shaders");
const ERROR_LOG   = path.join(__dirname, "shader-compiling-error.log");
const SERVER_LOG  = path.join(__dirname, "server.log");

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

  // ── GET /shaders ──────────────────────────────────────────────────────────
  if (req.method === "GET" && pathname === "/shaders") {
    try {
      const files = fs.readdirSync(SHADERS_DIR)
        .filter(f => f.endsWith(".metal"))
        .map(f => f.replace(/\.metal$/, ""));
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

  // ── 404 ───────────────────────────────────────────────────────────────────
  res.writeHead(404);
  res.end("Not found");
});

server.listen(PORT, () => {
  log(`Serving shaders from: ${SHADERS_DIR}`);
  log(`Listening on http://localhost:${PORT}`);
  log(`All logs → ${SERVER_LOG}`);
  log(`Errors     → ${ERROR_LOG}`);
  log(`Available shaders:`);
  try {
    const files = fs.readdirSync(SHADERS_DIR).filter(f => f.endsWith(".metal"));
    if (files.length === 0) {
      log("  (none – add .metal files to the shaders/ directory)");
    } else {
      files.forEach(f => log(`  - ${f}`));
    }
  } catch {
    log("  (could not read shaders directory)");
  }
});
