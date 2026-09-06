#!/usr/bin/env node

/**
 * check-shaders.js
 *
 * Type-checks every runtime-loadable shader in shaders/ by wrapping it with
 * the same boilerplate DynamicBoxRenderer.wrapShaderSource() injects at
 * runtime, then compiling the result with the real Metal compiler
 * (`xcrun metal -c ... -o ...`).
 *
 * These .metal files are excluded from the Xcode target (see
 * membershipExceptions in project.pbxproj) so they are never compiled as part
 * of a normal app build. This script is how we still catch type/compile
 * errors for them locally and in the git repo, without shipping them in the
 * app bundle.
 *
 * IMPORTANT: the PRELUDE constant below must be kept in sync with
 * DynamicBoxRenderer.wrapShaderSource() in DynamicBoxRenderer.swift. If you
 * change one, change the other.
 *
 * Usage:
 *   cd vr-dive/Demos/DynamicBox && node check-shaders.js
 *   cd vr-dive/Demos/DynamicBox && node check-shaders.js waves fibers-wave   # check specific files only
 *
 * Exit code is non-zero if any shader fails to compile.
 */

const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");

const SHADERS_DIR = path.join(__dirname, "shaders");
const MODULE_CACHE_DIR = fs.mkdtempSync(path.join(os.tmpdir(), "dynbox-metal-module-cache-"));
process.on("exit", () => fs.rmSync(MODULE_CACHE_DIR, { recursive: true, force: true }));

// ─── Prelude (mirrors DynamicBoxRenderer.wrapShaderSource) ────────────────────
const PRELUDE = `#include <metal_stdlib>
using namespace metal;

// ─── Auto-generated structs (must match Swift side) ─────────────────────
struct DynamicBoxUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;
    float  _pad;
    float4 objectCenter;
    float4x4 patternTransform;
};

struct DynamicBoxVertexOut {
    float4 clipPos   [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ─── Shared helpers ─────────────────────────────────────────────────────
#define DB_PI      3.14159265f
#define DB_BOXDIMS float3(0.95f, 0.95f, 1.25f)

static float db_boxHit(float3 ro, float3 rd, float3 r, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n  = ro * dr;
    float3 k  = r  * abs(dr);
    float3 pin  = -k - n;
    float3 pout =  k - n;
    float tin  = max(pin.x,  max(pin.y,  pin.z));
    float tout = min(pout.x, min(pout.y, pout.z));
    if (tin > tout) return -1.0f;
    if (entering) {
        nn = -sign(rd) * step(pin.zxy,  pin.xyz)  * step(pin.yzx,  pin.xyz);
        return tin;
    } else {
        nn =  sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
        return tout;
    }
}
`;

function listShaderFiles() {
  return fs
    .readdirSync(SHADERS_DIR)
    .filter(f => f.endsWith(".metal"))
    .sort();
}

function checkShader(fileName) {
  const filePath = path.join(SHADERS_DIR, fileName);
  const userSource = fs.readFileSync(filePath, "utf-8");
  const wrapped = `${PRELUDE}\n// ─── User shader: ${fileName} ───────────────────────────\n${userSource}`;

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "dynbox-shader-check-"));
  const tmpMetal = path.join(tmpDir, fileName);
  const tmpAir = path.join(tmpDir, fileName.replace(/\.metal$/, ".air"));

  try {
    fs.writeFileSync(tmpMetal, wrapped, "utf-8");
    execFileSync("xcrun", ["metal", "-c", "-Wall", `-fmodules-cache-path=${MODULE_CACHE_DIR}`, tmpMetal, "-o", tmpAir], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    return { ok: true };
  } catch (err) {
    const stderr = err.stderr ? err.stderr.toString() : String(err.message);
    return { ok: false, error: stderr };
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function main() {
  const requested = process.argv.slice(2);
  const allFiles = listShaderFiles();
  const files =
    requested.length > 0
      ? requested.map(name => (name.endsWith(".metal") ? name : `${name}.metal`))
      : allFiles;

  if (files.length === 0) {
    console.log("No shader files found in", SHADERS_DIR);
    process.exit(0);
  }

  let failures = 0;
  for (const file of files) {
    if (!allFiles.includes(file)) {
      console.log(`\u2717 ${file} \u2014 file not found in shaders/`);
      failures++;
      continue;
    }
    process.stdout.write(`checking ${file} ... `);
    const result = checkShader(file);
    if (result.ok) {
      console.log("OK");
    } else {
      console.log("FAILED");
      console.log(
        result.error
          .split("\n")
          .map(l => "    " + l)
          .join("\n")
      );
      failures++;
    }
  }

  console.log(`\n${files.length - failures}/${files.length} shaders passed type-check.`);
  if (failures > 0) {
    process.exitCode = 1;
  }
}

// Reuse the exact runtime wrapper for the optional on-Mac image preview.
module.exports = { PRELUDE };
if (require.main === module) main();
