use std::collections::HashSet;
use std::env;
use std::f64::consts::PI;
use std::fs;

const TWO_PI: f64 = 2.0 * PI;

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }

    fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
struct Params {
    a: f64,
    b: f64,
    c: f64,
}

#[derive(Clone, Debug)]
struct Candidate {
    name: String,
    source: String,
    params: Params,
}

#[derive(Clone, Debug)]
struct Metrics {
    point_count: usize,
    occupied_voxels: usize,
    occupancy_ratio: f64,
    extents: [f64; 3],
    stddev: [f64; 3],
    eigenvalues: [f64; 3],
    avg_step: f64,
    step_stddev: f64,
    radius_mean: f64,
    radius_stddev: f64,
    axis_balance: f64,
    linearity: f64,
    depth_ratio: f64,
}

#[derive(Clone, Debug)]
struct Evaluation {
    candidate: Candidate,
    score: f64,
    metrics: Metrics,
}

#[derive(Clone, Debug)]
struct Config {
    output: String,
    steps: usize,
    warmup: usize,
    keep_top: usize,
    grid: usize,
    jitter: f64,
    random_count: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            output: "vr-dive/Demos/SimoneOrbit3D/preset-search-results-rust.json".to_string(),
            steps: 14_000,
            warmup: 500,
            keep_top: 16,
            grid: 22,
            jitter: 0.16,
            random_count: 800,
        }
    }
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = self.state >> 11;
        (x as f64) / ((1u64 << 53) as f64)
    }
}

fn wrap_angle(v: f64) -> f64 {
    let mut r = v % TWO_PI;
    if r < 0.0 {
        r += TWO_PI;
    }
    r
}

fn simone_map(p: Vec3, params: Params) -> Vec3 {
    Vec3 {
        x: (p.x * p.x - p.y * p.y - p.z * p.z + params.a).sin(),
        y: (2.0 * p.x * p.y + params.b).cos(),
        z: (2.0 * p.x * p.z + params.c).sin(),
    }
}

fn derive_c(a: f64, b: f64, mode: &str) -> f64 {
    match mode {
        "half-delta" => 0.5 * (a - b),
        "half-sum" => 0.5 * (a + b),
        "neg-half-sum" => -0.5 * (a + b),
        "a" => a,
        "b" => b,
        _ => 0.5 * (a - b),
    }
}

fn jacobi_eigenvalues(mut m: [[f64; 3]; 3]) -> [f64; 3] {
    for _ in 0..28 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off_diag = m[0][1].abs();

        for i in 0..3 {
            for j in (i + 1)..3 {
                let v = m[i][j].abs();
                if v > max_off_diag {
                    max_off_diag = v;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off_diag < 1e-11 {
            break;
        }

        let theta = 0.5 * (2.0 * m[p][q]).atan2(m[q][q] - m[p][p]);
        let c = theta.cos();
        let s = theta.sin();

        let app = c * c * m[p][p] - 2.0 * s * c * m[p][q] + s * s * m[q][q];
        let aqq = s * s * m[p][p] + 2.0 * s * c * m[p][q] + c * c * m[q][q];
        m[p][p] = app;
        m[q][q] = aqq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;

        for r in 0..3 {
            if r == p || r == q {
                continue;
            }
            let mrp = c * m[r][p] - s * m[r][q];
            let mrq = s * m[r][p] + c * m[r][q];
            m[r][p] = mrp;
            m[p][r] = mrp;
            m[r][q] = mrq;
            m[q][r] = mrq;
        }
    }

    [m[0][0], m[1][1], m[2][2]]
}

fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

fn remap(v: f64, lo: f64, hi: f64) -> f64 {
    (v - lo) / (hi - lo)
}

fn bell_score(v: f64, low: f64, mid: f64, high: f64) -> f64 {
    if v <= low || v >= high {
        0.0
    } else if v < mid {
        clamp01(remap(v, low, mid))
    } else {
        clamp01(remap(high - v, high - mid, high - mid))
    }
}

fn score_metrics(m: &Metrics) -> f64 {
    let occupancy_score = bell_score(m.occupied_voxels as f64, 220.0, 900.0, 1700.0);
    let depth_score = bell_score(m.depth_ratio, 0.02, 0.09, 0.25);
    let linearity_score = clamp01(remap(m.linearity, 0.08, 0.76));
    let balance_score = clamp01(remap(m.axis_balance, 0.04, 0.40));
    let step_score =
        clamp01(remap(m.avg_step, 0.25, 1.25)) * (1.0 - clamp01(remap(m.step_stddev, 0.65, 1.6)));
    let radius_score = 1.0 - clamp01(remap(m.radius_stddev, 0.7, 1.7));

    occupancy_score * 0.28
        + depth_score * 0.18
        + linearity_score * 0.20
        + balance_score * 0.12
        + step_score * 0.13
        + radius_score * 0.09
}

fn evaluate_candidate(candidate: Candidate, cfg: &Config, seeds: &[Vec3]) -> Evaluation {
    let mut min = Vec3 {
        x: f64::INFINITY,
        y: f64::INFINITY,
        z: f64::INFINITY,
    };
    let mut max = Vec3 {
        x: f64::NEG_INFINITY,
        y: f64::NEG_INFINITY,
        z: f64::NEG_INFINITY,
    };
    let mut sum = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let mut sq_sum = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    let mut cov_acc = [0.0f64; 6];
    let mut step_sum = 0.0;
    let mut step_sq_sum = 0.0;
    let mut radius_sum = 0.0;
    let mut radius_sq_sum = 0.0;

    let mut point_count = 0usize;
    let mut voxel_set: HashSet<u32> = HashSet::new();

    for seed in seeds {
        let mut state = *seed;
        for _ in 0..cfg.warmup {
            state = simone_map(state, candidate.params);
        }

        let mut prev = state;
        for _ in 0..cfg.steps {
            state = simone_map(state, candidate.params);

            min.x = min.x.min(state.x);
            min.y = min.y.min(state.y);
            min.z = min.z.min(state.z);
            max.x = max.x.max(state.x);
            max.y = max.y.max(state.y);
            max.z = max.z.max(state.z);

            sum = sum.add(state);
            sq_sum.x += state.x * state.x;
            sq_sum.y += state.y * state.y;
            sq_sum.z += state.z * state.z;

            cov_acc[0] += state.x * state.x;
            cov_acc[1] += state.x * state.y;
            cov_acc[2] += state.x * state.z;
            cov_acc[3] += state.y * state.y;
            cov_acc[4] += state.y * state.z;
            cov_acc[5] += state.z * state.z;

            let step = state.sub(prev).length();
            step_sum += step;
            step_sq_sum += step * step;
            prev = state;

            let radius = state.length();
            radius_sum += radius;
            radius_sq_sum += radius * radius;

            let vx = (((state.x + 1.0) * 0.5) * 24.0).floor().clamp(0.0, 23.0) as u32;
            let vy = (((state.y + 1.0) * 0.5) * 24.0).floor().clamp(0.0, 23.0) as u32;
            let vz = (((state.z + 1.0) * 0.5) * 24.0).floor().clamp(0.0, 23.0) as u32;
            voxel_set.insert((vx << 10) | (vy << 5) | vz);

            point_count += 1;
        }
    }

    let inv_points = 1.0 / (point_count.max(1) as f64);
    let mean = sum.scale(inv_points);
    let var_x = (sq_sum.x * inv_points - mean.x * mean.x).max(0.0);
    let var_y = (sq_sum.y * inv_points - mean.y * mean.y).max(0.0);
    let var_z = (sq_sum.z * inv_points - mean.z * mean.z).max(0.0);

    let cov = [
        cov_acc[0] * inv_points - mean.x * mean.x,
        cov_acc[1] * inv_points - mean.x * mean.y,
        cov_acc[2] * inv_points - mean.x * mean.z,
        cov_acc[3] * inv_points - mean.y * mean.y,
        cov_acc[4] * inv_points - mean.y * mean.z,
        cov_acc[5] * inv_points - mean.z * mean.z,
    ];

    let mut eigen = jacobi_eigenvalues([
        [cov[0], cov[1], cov[2]],
        [cov[1], cov[3], cov[4]],
        [cov[2], cov[4], cov[5]],
    ]);
    eigen.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let extents = [max.x - min.x, max.y - min.y, max.z - min.z];
    let extent_min = extents[0].min(extents[1]).min(extents[2]);
    let extent_max = extents[0].max(extents[1]).max(extents[2]).max(1e-6);

    let step_count = (point_count - seeds.len()).max(1) as f64;
    let avg_step = step_sum / step_count;
    let step_var = (step_sq_sum / step_count - avg_step * avg_step).max(0.0);
    let step_stddev = step_var.sqrt();

    let radius_mean = radius_sum * inv_points;
    let radius_var = (radius_sq_sum * inv_points - radius_mean * radius_mean).max(0.0);
    let radius_stddev = radius_var.sqrt();

    let metrics = Metrics {
        point_count,
        occupied_voxels: voxel_set.len(),
        occupancy_ratio: voxel_set.len() as f64 / (24.0 * 24.0 * 24.0),
        extents,
        stddev: [var_x.sqrt(), var_y.sqrt(), var_z.sqrt()],
        eigenvalues: eigen,
        avg_step,
        step_stddev,
        radius_mean,
        radius_stddev,
        axis_balance: extent_min / extent_max,
        linearity: (eigen[0] - eigen[1]) / eigen[0].max(1e-6),
        depth_ratio: eigen[2] / eigen[0].max(1e-6),
    };

    let score = score_metrics(&metrics);

    Evaluation {
        candidate,
        score,
        metrics,
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args = env::args().collect::<Vec<_>>();

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" if i + 1 < args.len() => {
                cfg.output = args[i + 1].clone();
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                cfg.steps = args[i + 1].parse().unwrap_or(cfg.steps);
                i += 2;
            }
            "--warmup" if i + 1 < args.len() => {
                cfg.warmup = args[i + 1].parse().unwrap_or(cfg.warmup);
                i += 2;
            }
            "--keep-top" if i + 1 < args.len() => {
                cfg.keep_top = args[i + 1].parse().unwrap_or(cfg.keep_top);
                i += 2;
            }
            "--grid" if i + 1 < args.len() => {
                cfg.grid = args[i + 1].parse().unwrap_or(cfg.grid);
                i += 2;
            }
            "--jitter" if i + 1 < args.len() => {
                cfg.jitter = args[i + 1].parse().unwrap_or(cfg.jitter);
                i += 2;
            }
            "--random" if i + 1 < args.len() => {
                cfg.random_count = args[i + 1].parse().unwrap_or(cfg.random_count);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    cfg
}

fn build_candidates(cfg: &Config) -> Vec<Candidate> {
    let bourke_pairs = [
        (3.69, 4.51),
        (5.51, 4.84),
        (3.64, 1.71),
        (5.46, 4.55),
        (0.47, 2.25),
        (0.29, 0.95),
        (2.59, 2.49),
        (0.54, 1.23),
        (0.40, 5.11),
        (2.31, 1.64),
        (0.29, 4.00),
        (5.90, 5.64),
        (3.61, 4.24),
        (2.70, 2.32),
        (2.55, 0.93),
    ];

    let c_modes = ["half-delta", "half-sum", "neg-half-sum", "a", "b"];

    let mut out = Vec::<Candidate>::new();
    let mut rng = Lcg::new(0x9A22_11C3_BA89_77A1);

    for (pair_idx, (a, b)) in bourke_pairs.iter().enumerate() {
        for mode in c_modes {
            let c = derive_c(*a, *b, mode);
            out.push(Candidate {
                name: format!("bourke-{}-{}", pair_idx + 1, mode),
                source: "bourke-seed".to_string(),
                params: Params { a: *a, b: *b, c },
            });

            for j in 0..3usize {
                let ja = (rng.next_f64() - 0.5) * cfg.jitter * TWO_PI;
                let jb = (rng.next_f64() - 0.5) * cfg.jitter * TWO_PI;
                let jc = (rng.next_f64() - 0.5) * cfg.jitter * TWO_PI;
                out.push(Candidate {
                    name: format!("bourke-{}-{}-j{}", pair_idx + 1, mode, j + 1),
                    source: "bourke-jitter".to_string(),
                    params: Params {
                        a: wrap_angle(*a + ja),
                        b: wrap_angle(*b + jb),
                        c: wrap_angle(c + jc),
                    },
                });
            }
        }
    }

    let grid_size = cfg.grid.max(2);
    for ix in 0..grid_size {
        let fx = ix as f64 / (grid_size - 1) as f64;
        for iy in 0..grid_size {
            let fy = iy as f64 / (grid_size - 1) as f64;
            let a = fx * TWO_PI;
            let b = fy * TWO_PI;
            let c = wrap_angle(0.5 * (a - b));
            out.push(Candidate {
                name: format!("grid-{}-{}", ix + 1, iy + 1),
                source: "grid-half-delta".to_string(),
                params: Params { a, b, c },
            });
        }
    }

    for ridx in 0..cfg.random_count {
        let a = rng.next_f64() * TWO_PI;
        let b = rng.next_f64() * TWO_PI;
        let mode = c_modes[(rng.next_f64() * (c_modes.len() as f64)) as usize % c_modes.len()];
        let c =
            wrap_angle(derive_c(a, b, mode) + (rng.next_f64() - 0.5) * cfg.jitter * TWO_PI * 1.5);
        out.push(Candidate {
            name: format!("random-{}", ridx + 1),
            source: format!("random-{}", mode),
            params: Params { a, b, c },
        });
    }

    out
}

fn write_results(path: &str, top: &[Evaluation], cfg: &Config) {
    let mut body = String::new();
    body.push_str("{\n");
    body.push_str(&format!(
        "  \"meta\": {{\"steps\": {}, \"warmup\": {}, \"grid\": {}, \"random\": {}}},\n",
        cfg.steps, cfg.warmup, cfg.grid, cfg.random_count
    ));
    body.push_str("  \"top\": [\n");

    for (i, e) in top.iter().enumerate() {
        body.push_str("    {\n");
        body.push_str(&format!("      \"rank\": {},\n", i + 1));
        body.push_str(&format!("      \"name\": \"{}\",\n", e.candidate.name));
        body.push_str(&format!("      \"source\": \"{}\",\n", e.candidate.source));
        body.push_str(&format!("      \"score\": {:.6},\n", e.score));
        body.push_str(&format!(
            "      \"params\": {{\"a\": {:.6}, \"b\": {:.6}, \"c\": {:.6}}},\n",
            e.candidate.params.a, e.candidate.params.b, e.candidate.params.c
        ));
        body.push_str(&format!(
            "      \"metrics\": {{\"occupiedVoxels\": {}, \"occupancyRatio\": {:.6}, \"linearity\": {:.6}, \"depthRatio\": {:.6}, \"axisBalance\": {:.6}, \"avgStep\": {:.6}, \"stepStddev\": {:.6}, \"radiusMean\": {:.6}, \"radiusStddev\": {:.6}}}\n",
            e.metrics.occupied_voxels,
            e.metrics.occupancy_ratio,
            e.metrics.linearity,
            e.metrics.depth_ratio,
            e.metrics.axis_balance,
            e.metrics.avg_step,
            e.metrics.step_stddev,
            e.metrics.radius_mean,
            e.metrics.radius_stddev
        ));
        body.push_str(if i + 1 == top.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }

    body.push_str("  ]\n");
    body.push_str("}\n");

    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, body).expect("failed to write results");
}

fn main() {
    let cfg = parse_args();

    let seeds = [
        Vec3 {
            x: 0.12,
            y: -0.09,
            z: 0.04,
        },
        Vec3 {
            x: -0.18,
            y: 0.06,
            z: 0.11,
        },
        Vec3 {
            x: 0.07,
            y: 0.15,
            z: -0.13,
        },
    ];

    let candidates = build_candidates(&cfg);
    println!(
        "[simone-rust] candidates={} steps={} warmup={}",
        candidates.len(),
        cfg.steps,
        cfg.warmup
    );

    let mut evals = Vec::<Evaluation>::with_capacity(candidates.len());
    for candidate in candidates {
        evals.push(evaluate_candidate(candidate, &cfg, &seeds));
    }

    evals.sort_by(|lhs, rhs| {
        rhs.score
            .partial_cmp(&lhs.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = cfg.keep_top.min(evals.len());
    let top = &evals[..keep];

    for (i, e) in top.iter().take(8).enumerate() {
        println!(
            "#{:02} score={:.4} a={:.4} b={:.4} c={:.4} vox={} line={:.3} depth={:.3} {}",
            i + 1,
            e.score,
            e.candidate.params.a,
            e.candidate.params.b,
            e.candidate.params.c,
            e.metrics.occupied_voxels,
            e.metrics.linearity,
            e.metrics.depth_ratio,
            e.candidate.name
        );
    }

    write_results(&cfg.output, top, &cfg);
    println!("[simone-rust] wrote {}", cfg.output);
}
