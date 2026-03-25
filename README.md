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

## Feature flags

| Feature | Description |
|---|---|
| *(default)* | Headless ECS only -- no window, no rendering |
| `render` | Adds `bevy_render`, `bevy_winit`, `bevy_core_pipeline`, `bevy_asset`, `bevy_sprite`, `bevy_sprite_render` |

## Examples

| Example | Notes |
|---|---|
| [`cartpole`](docs/examples/cartpole.md) | 4 parallel envs, DQN via TrainingSession, optional live rendering |

## Plugin docs

Detailed reference for each plugin:

- [BevyGymPlugin](docs/plugins/bevy_gym_plugin.md) -- core plugin, factory, headless/render modes, system ordering
- [GymRender / GymRenderPlugin](docs/plugins/gym_render.md) -- optional 2D visualisation
- [GymStatsPlugin / GymStats](docs/plugins/gym_stats.md) -- rolling episode statistics

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
