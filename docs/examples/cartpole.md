# cartpole

Full-stack CartPole-v1 example: 4 parallel environments, DQN from ember-rl via `TrainingSession`, live stats, optional 2D rendering.

## Usage

```sh
# Train headless at maximum speed
cargo run --example cartpole --release

# Train with live 2D rendering
cargo run --example cartpole --features render --release -- --render

# Train with rendering at half speed
cargo run --example cartpole --features render --release -- --render --speed 0.5

# Evaluate a saved checkpoint (headless)
cargo run --example cartpole --release -- --eval runs/bevy_cartpole/v1

# Evaluate with live rendering
cargo run --example cartpole --features render --release -- --eval runs/bevy_cartpole/v1 --render
```

## What it demonstrates

- `BevyGymPlugin` with 4 parallel environments stepped each `FixedUpdate` tick
- `TrainingSession` from ember-rl as a `NonSendMut` resource -- automatic checkpointing, JSONL logging, and `best.mpk` saving
- `GymStatsPlugin` for rolling mean/max reward and steps/sec across all envs
- `GymRender` + `GymRenderPlugin` for live 2D visualisation (cart, pole, danger colouring)
- Headless mode: `MinimalPlugins` + `ScheduleRunnerPlugin` + virtual time at maximum speed
- `--eval` flag: loads `best.mpk` into a `DqnPolicy`, runs 20 greedy episodes, prints summary

## Rendering details

Each environment is drawn as a cart (blue rectangle) + pole (green rectangle) stacked vertically on screen. Colours shift as state approaches failure:

- Pole: green -> red as angle approaches the 12 degree limit
- Cart: blue -> orange as position approaches the +/-2.4 boundary

Red boundary markers are drawn at x = +/-2.4 (240 px at 100 px/unit scale).
