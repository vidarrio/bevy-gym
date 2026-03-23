# bevy-gym

[![crates.io](https://img.shields.io/crates/v/bevy-gym.svg)](https://crates.io/crates/bevy-gym)
[![docs.rs](https://docs.rs/bevy-gym/badge.svg)](https://docs.rs/bevy-gym)
[![CI](https://github.com/vidarrio/bevy-gym/actions/workflows/ci.yml/badge.svg)](https://github.com/vidarrio/bevy-gym/actions/workflows/ci.yml)

Bevy ECS plugin for parallelised RL environment simulation.

`bevy-gym` bridges [`rl-traits`](https://crates.io/crates/rl-traits) environments into Bevy's ECS,
giving you N parallel environment instances stepped in parallel via `par_iter_mut()` each
`FixedUpdate` tick — with no extra code.

## Ecosystem

| Crate | Role |
|---|---|
| [`rl-traits`](https://crates.io/crates/rl-traits) | Shared traits and types |
| [`ember-rl`](https://crates.io/crates/ember-rl) | Algorithm implementations (DQN, PPO, SAC) using Burn |
| **bevy-gym** | Bevy ECS plugin for parallelised environment simulation (this crate) |

## Design goals

**Free parallelism.** N environment instances live as separate Bevy entities. `par_iter_mut()`
steps all of them in parallel each `FixedUpdate` tick. The expensive simulation work runs across
all available CPU cores without any extra synchronisation.

**Decoupled tick rate.** RL simulation runs in `FixedUpdate` at a configurable Hz. Rendering
(if enabled) runs in `Update` at frame rate. The two are completely independent.

**Headless training mode.** Disable rendering entirely and run at maximum CPU throughput via
`ScheduleRunnerPlugin`. The `render` feature adds windowing and rendering on top of the headless
baseline.

**Message-driven policy integration.** `ActionRequestEvent` tells your policy system when to
provide the next action. `ExperienceEvent` delivers the full `(s, a, r, s', status)` transition
back to ember-rl (or any other consumer) without coupling to Bevy internals.

**Correct terminated/truncated distinction.** `EpisodeEndEvent` carries the actual
`EpisodeStatus` from rl-traits — bootstrapping algorithms receive exactly the signal they need.

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
bevy-gym = "0.3"
bevy = { version = "0.18", default-features = false, features = ["default_app", "multi_threaded"] }
```

### Headless training with 4 parallel environments

```rust
use bevy::prelude::*;
use bevy::app::ScheduleRunnerPlugin;
use bevy_gym::{BevyGymPlugin, ActionRequestEvent, GymSet};
use bevy_gym::components::{CurrentObservation, PendingAction};

App::new()
    .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::ZERO)))
    .add_plugins(BevyGymPlugin::new(|_| MyEnv::new(), 4).headless())
    .add_systems(FixedUpdate, policy_system.after(GymSet::ManualReset))
    .run();

fn policy_system(
    mut requests: MessageReader<ActionRequestEvent>,
    mut query: Query<(&CurrentObservation<MyEnv>, &mut PendingAction<MyEnv>)>,
) {
    for req in requests.read() {
        if let Ok((obs, mut pending)) = query.get_mut(req.entity) {
            pending.action = Some(my_policy(&obs.observation));
        }
    }
}
```

### Integrating with ember-rl's `TrainingSession`

Use `TrainingSession` from ember-rl as a NonSend resource to get automatic
checkpointing, JSONL logging, and best-model saving with no manual save calls:

```rust
use ember_rl::training::{TrainingSession, TrainingRun};
use ember_rl::traits::ActMode;

type Session = TrainingSession<MyEnv, DqnAgent<MyEnv, VecEncoder, UsizeActionMapper, B>>;

fn policy_system(
    mut requests: MessageReader<ActionRequestEvent>,
    mut query: Query<(&CurrentObservation<MyEnv>, &mut PendingAction<MyEnv>)>,
    mut session: NonSendMut<Session>,
) {
    for req in requests.read() {
        if let Ok((obs, mut pending)) = query.get_mut(req.entity) {
            pending.action = Some(session.act(&obs.observation, ActMode::Explore));
        }
    }
}

fn learn_system(
    mut events: MessageReader<ExperienceEvent<Vec<f32>, usize>>,
    mut session: NonSendMut<Session>,
) {
    for event in events.read() {
        session.observe(event.experience.clone());
    }
}

fn on_episode_end(
    mut events: MessageReader<EpisodeEndEvent>,
    mut session: NonSendMut<Session>,
) {
    for event in events.read() {
        // env_extras from EpisodeEndEvent are merged with agent extras automatically
        session.on_episode(event.total_reward, event.episode_steps, event.status.clone(), event.extras.clone());
    }
}
```

### Live training statistics

Add `GymStatsPlugin` to get a `GymStats` resource tracking rolling episode stats across all envs:

```rust
App::new()
    .add_plugins(BevyGymPlugin::new(env_factory, 4).headless())
    .add_plugins(GymStatsPlugin::new())           // add after BevyGymPlugin
    .add_systems(FixedUpdate, log_stats.after(GymSet::ManualReset))
    .run();

fn log_stats(stats: Res<GymStats>) {
    let g = stats.global();
    if g.episode_count() % 50 == 0 && g.episode_count() > 0 {
        println!("ep {:>5}  reward mean/max: {:.1}/{:.1}  steps/sec: {:.0}",
                 g.episode_count(), g.mean_reward(), g.max_reward(), g.steps_per_sec());
    }
}
```

`GymStats` also exposes per-environment stats via `.env(id)`.

### Manual environment resets

Add `ResetRequested` to any environment entity to trigger a reset on the next tick:

```rust
commands.entity(env_entity).insert(ResetRequested { seed: Some(42) });
```

## Plugin builder API

```rust
BevyGymPlugin::new(factory, num_envs)  // factory: Fn(usize) -> E
    .with_tick_rate(120.0)             // fixed tick rate in Hz (default: 60)
    .headless()                        // uncapped, no window

GymStatsPlugin::new()
    .with_window(200)                  // rolling window size (default: 100)
```

## Message types

| Message | When fired | Contents |
|---|---|---|
| `ActionRequestEvent` | After every step and reset | `env_id`, `entity` |
| `ExperienceEvent` | After every step | `env_id`, full `Experience` tuple |
| `EpisodeEndEvent` | When an episode ends | `env_id`, `status`, `total_reward`, `episode_steps`, `extras` |

## System ordering

All RL systems run in `FixedUpdate` in the `GymSet` order:

```
GymSet::Step → GymSet::AutoReset → GymSet::ManualReset
```

User policy and logging systems should run **after** `GymSet::ManualReset` to
ensure they see a fully consistent state for the current tick.

## Feature flags

| Feature | Description |
|---|---|
| *(default)* | Headless ECS only — no window, no rendering |
| `render` | Adds `bevy_render`, `bevy_winit`, `bevy_core_pipeline`, `bevy_asset`, `bevy_sprite`, `bevy_sprite_render` |

## Optional rendering

The `render` feature exposes the `GymRender` trait and `GymRenderPlugin`. Implement
`GymRender` for your environment to get live 2D visualisation with no changes to the
rest of your setup:

```rust
#[cfg(feature = "render")]
impl GymRender for MyEnv {
    type Visuals = MyVisuals;  // Component holding spawned mesh Entity handles

    fn setup_visuals(entity: Entity, env_id: usize, ctx: &mut SpawnCtx) {
        // spawn Mesh2d + MeshMaterial2d children, insert MyVisuals onto entity
    }

    fn sync_visuals(obs: &Self::Observation, visuals: &MyVisuals, transforms: &mut Query<&mut Transform>) {
        // reposition mesh entities from the latest observation
    }
}
```

Then add `GymRenderPlugin` alongside a camera and `DefaultPlugins`:

```rust
app.add_plugins(DefaultPlugins)
   .add_plugins(GymRenderPlugin::<MyEnv>::new())
   .add_systems(Startup, |mut commands: Commands| { commands.spawn(Camera2d); });
```

## CartPole example

The included CartPole example demonstrates the full stack: 4 parallel environments,
a DQN agent from `ember-rl` coordinated via `TrainingSession`, `GymStatsPlugin` for
live stats, automatic checkpointing, and optional live 2D rendering.

```
# Train headless at maximum speed
cargo run --example cartpole --release

# Train with live rendering
cargo run --example cartpole --features render --release -- --render

# Train with live rendering at half speed
cargo run --example cartpole --features render --release -- --render --speed 0.5

# Evaluate a saved checkpoint (headless)
cargo run --example cartpole --release -- --eval runs/bevy_cartpole/v1

# Evaluate with live rendering
cargo run --example cartpole --features render --release -- --eval runs/bevy_cartpole/v1 --render
```

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
