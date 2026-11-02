# Simulation snapshot — 2026-11-02

## Current behavior

The reactive mission cruises forward through FOV-limited startup unknown space, without painting free space into the map. Known occupied cells remain blocking. At approximately 3 m frontal clearance it brakes, asks the original ETHZ NBV tree for a usable direction, turns to that direction, validates it against the current map, and resumes continuous cruise. It does not wait for the selected source edge endpoint or execute the sampled view yaw.

Wavemap is the only depth-fusion owner. NBV imports a 0.1 m OctoMap query projection; it does not build a second depth map. Planning and control use native estimated body state, not ground truth.

## Finite live verification

Run `20261102T031510Z-29de60ff` was started by the operator through RViz Explore. At the saved independent checkpoint, four turn-and-cruise cycles completed, Guard remained ARMED in generation 1, and the task remained enabled. Direction commits used map revisions 258, 349, 499, and 570. This is a finite local-navigation checkpoint, not long-term or global exploration acceptance.

During 13 observed map-gap samples of already-started turns, the source continued pure-turn index 1 with zero XYZ reference. Fresh body state and authority checks remain active. Starting a turn, committing a new cruise direction, and translating still require current-map validation. Zero XYZ reference does not prove zero physical drift in water.

The source Tracker reported its original reached/dwell criteria. An independent raw-state window for turn 3 briefly measured 0.0447 m/s; it is not evidence that every 25 Hz raw sample satisfied the source 0.04 m/s threshold throughout the whole second. The source consumes state at its own 20 Hz tick.

## Frozen approved settings

| Item | Value |
| --- | --- |
| NBV query resolution | 0.1 m |
| Wavemap fusion / publish | 10 / 5 Hz |
| Encounter distance | 3 m |
| JEPA samples | 64 |
| Solve deadline / maximum plan age / latency compensation | 0.36 / 0.60 / 0.36 s |
| Requested horizontal speed limit | 0.48 m/s |

These are recorded approved settings, not permission to tune other parameters. The original gain calculation, sampled yaw, 15/200 conditions, collision checks, and Guard/terminal expiry checks are retained. Source information gain is not a claim that its sampled viewing direction was actually executed.

## Known limits

- Long-term exploration and global route quality remain unverified; a selected direction can trigger another obstacle check after a short distance.
- Full map projection can create update gaps. Only an already-started stationary turn is exempt from map freshness while rotating; translation is not.
- The preceding run recorded an RViz OpenGL index-buffer exception and subsequently a Stonefish exit with code 0. The available evidence does not establish causality or GPU memory exhaustion.
- Duplicate RViz map/path display instances were removed from the saved configuration. That change applies when the configuration is next opened; it was not retroactively loaded into this checkpoint.
- VIO freshness improved in the observed window after separating ROS ingress from visual processing. This is not proof that all VIO failure modes are resolved.

## Provenance

Selected source records and the finite checkpoint audit are retained in [provenance/](provenance/). Raw runs, build logs, screenshots, and generated binaries stay local under ignored `runs/` and `build/` directories. Historical handover links into those directories refer to local evidence, which is not included in the lightweight Git snapshot. Provenance hashes describe their recorded checkpoints and must not be rewritten to impersonate later source revisions.
