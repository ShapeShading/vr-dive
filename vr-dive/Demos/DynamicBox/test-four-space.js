#!/usr/bin/env node
// Geometry invariants for the 4D demos. Equations mirror their Metal versions;
// combine this with check-shaders.js and real-GPU preview-shaders.js.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const fourSpaceNames = ["clifford-lantern", "hopf-fibration", "tesseract-jewel", "cell24-prism", "s3-trefoil"];
const runtimeNames = [...fourSpaceNames, "nacre-rosette", "orbital-lace",
  "voxel-tide", "lamellar-bloom", "quaternion-reef", "prismatic-plume",
  "coral-folds", "fiber-pleats", "liquid-contours", "pleated-marble",
  "radial-gills", "sediment-ribbons", "topographic-velvet"];
const sources = Object.fromEntries(runtimeNames.map(name => [name,
  fs.readFileSync(path.join(__dirname, "shaders", name + ".metal"), "utf8")]));
const project = fs.readFileSync(path.join(__dirname, "../../..", "vr-dive.xcodeproj/project.pbxproj"), "utf8");
function capture(text, pattern, message) {
  const match = text.match(pattern);
  assert(match, message);
  return match[1];
}
const exceptions = capture(project, /membershipExceptions\s*=\s*\(([\s\S]*?)\);/,
  "Xcode project must contain a membershipExceptions block");
for (const name of runtimeNames) {
  assert(exceptions.includes("Demos/DynamicBox/shaders/" + name + ".metal"), name + " must be runtime-only");
  assert(/patternTransform\s*\*\s*float4\s*\(\s*ro\s*,\s*1(?:\.0f)?\s*\)/.test(sources[name]),
    name + " must support navigation");
  assert(!/\bro\s*\+=\s*rd\s*\*/.test(sources[name]), name + " must not clip geometry at the Box entry");
}
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const length = a => Math.hypot(...a);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const normalize = a => a.map(v => v / length(a));
const near = (a, b, message, epsilon = 1e-8) => assert(Math.abs(a-b) < epsilon, message + ": " + a + " vs " + b);
function rotate(q, angles) {
  q = q.slice();
  [[0,3], [1,2], [2,3]].forEach(([i,j], k) => {
    const c = Math.cos(angles[k]), s = Math.sin(angles[k]), a = q[i], b = q[j];
    q[i] = c*a-s*b; q[j] = s*a+c*b;
  });
  return q;
}
const projectStereo = (q, scale) => q.slice(0,3).map(v => scale*v/(1-q[3]));
const wRow = angles => [0,1,2,3].map(i => rotate([0,1,2,3].map(j => i === j ? 1 : 0), angles)[3]);
const projectedRadius = (w, scale) => scale*Math.sqrt((1+w)/(1-w));

// Read the actual Metal lookup tables, rather than testing regenerated copies.
const vertexBlock = capture(sources["cell24-prism"], /CELL24_VERTICES\[24\] = \{([\s\S]*?)\};/,
  "cell24-prism must declare CELL24_VERTICES[24]");
const edgeBlock = capture(sources["cell24-prism"], /CELL24_EDGES\[96\] = \{([\s\S]*?)\};/,
  "cell24-prism must declare CELL24_EDGES[96]");
const vertices = [...vertexBlock.matchAll(/float4\(([^)]+)\)/g)].map(m => m[1].split(",").map(parseFloat));
const edges = [...edgeBlock.matchAll(/ushort2\((\d+),(\d+)\)/g)].map(m => [Number(m[1]), Number(m[2])]);
assert.equal(vertices.length, 24);
assert.equal(edges.length, 96);
assert.equal(new Set(edges.map(e => e.join(","))).size, 96);
const degree = Array(24).fill(0);
for (const [a,b] of edges) {
  assert(a >= 0 && a < b && b < 24);
  near(dot(sub(vertices[a],vertices[b]),sub(vertices[a],vertices[b])), 1, "24-cell edge", 1e-7);
  degree[a]++; degree[b]++;
}
assert(degree.every(d => d === 8));
for (const v of vertices) near(dot(v,v), 1, "24-cell unit vertex", 1e-7);
let tesseractEdges = 0;
for (let i = 0; i < 16; i++) for (let axis = 0; axis < 4; axis++) if (!(i & (1 << axis))) tesseractEdges++;
assert.equal(tesseractEdges, 32);

let maxClifford = 0, maxKnot = 0, minHopfDenominator = 1;
for (let frame = 0; frame <= 600; frame++) {
  const t = frame;
  const ca = [0.26*Math.sin(0.16*t+0.5), 0.12*t+0.4, 0.08*Math.sin(0.11*t)];
  const cr = wRow(ca), a = Math.hypot(cr[0],cr[1]), b = Math.hypot(cr[2],cr[3]);
  // The shell is contained in |theta-pi/4|<=asin(.020). Its filigree only
  // removes material, so this bounds every band including the rounded edges.
  const delta = Math.asin(0.020), mid = Math.PI/4;
  const theta = Math.max(mid-delta, Math.min(mid+delta, Math.atan2(b,a)));
  const maxW = a*Math.cos(theta) + b*Math.sin(theta);
  const cliffordRadius = projectedRadius(maxW, 0.40);
  maxClifford = Math.max(maxClifford, cliffordRadius);
  assert(cliffordRadius < 2.1, "Clifford bound at " + t);
  const ka = [0.22*Math.sin(t*0.13), t*0.11+0.38, 0.10*Math.sin(t*0.17+0.7)];
  const kr = wRow(ka);
  const knotW = 0.8*Math.hypot(kr[0],kr[1]) + 0.6*Math.hypot(kr[2],kr[3]);
  const knotRadius = projectedRadius(knotW,0.43)+0.052;
  maxKnot = Math.max(maxKnot,knotRadius);
  assert(knotRadius < 1.65, "Trefoil bound at " + t);
  const ha = [0.18*Math.sin(t*0.13), 0.11*t+0.32, 0.12*Math.sin(t*0.09+0.4)];
  for (let i = 0; i < 12; i++) {
    const phi = 2*Math.PI*i/12, eta = 0.55+0.13*Math.sin(3*phi+0.3);
    const ce = Math.cos(eta), se = Math.sin(eta), cp = Math.cos(phi), sp = Math.sin(phi);
    const a4 = rotate([ce,0,se*cp,se*sp],ha), b4 = rotate([0,ce,-se*sp,se*cp],ha);
    near(dot(a4,a4),1,"fiber basis length"); near(dot(a4,b4),0,"fiber basis orthogonal");
    const denominator = 1-a4[3]**2-b4[3]**2;
    minHopfDenominator = Math.min(minHopfDenominator,denominator);
    assert(denominator > 0.1, "Hopf projection near pole");
    const center = a4.slice(0,3).map((v,j) => 0.44*(a4[3]*v+b4[3]*b4[j])/denominator);
    const normal = normalize(cross(a4,b4));
    const radius = 0.44/Math.sqrt(denominator);
    for (let k = 0; k < 8; k++) {
      const angle = 2*Math.PI*k/8;
      const q = a4.map((v,j) => v*Math.cos(angle)+b4[j]*Math.sin(angle));
      const p = projectStereo(q,0.44), offset = sub(p,center);
      near(dot(offset,normal),0,"projected fiber plane");
      near(length(offset),radius,"projected fiber circle");
      const x = p.map(v => v/0.44), r2 = dot(x,x);
      const back = [...x.map(v => 2*v/(1+r2)),(r2-1)/(1+r2)];
      near(length(sub(q,back)),0,"inverse stereographic round-trip");
    }
  }
}
// Projection from distance 2.1 of any unit R4 vertex has this global bound.
assert(2.2/Math.sqrt(2.1**2-1)+0.048 < 1.35);
assert(2.0/Math.sqrt(2.1**2-1)+0.029 < 1.3);
console.log("Runtime exclusions / 24-cell topology / 32 tesseract edges: OK");
console.log("601 animation samples: Clifford max radius", maxClifford.toFixed(3),
            ", trefoil max radius", maxKnot.toFixed(3), ", Hopf min denominator", minHopfDenominator.toFixed(3));
console.log("57,696 projected Hopf points / circle equations / inverse projection: OK");
