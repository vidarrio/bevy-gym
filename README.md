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
bevy-gym = "*"
bevy = { version = "0.18", default-features = false, features = ["default_app", "multi_threaded"] }
```

### Headless training with 16 parallel environments

```rust
use bevy::prelude::*;
use bevy_gym::{BevyGymPlugin, ActionRequestEvent};
use bevy_gym::components::PendingAction;

App::new()
    .add_plugins(MinimalPlugins)
    .add_plugins(
        BevyGymPlugin::new(|i| CartPoleEnv::new(i as u64), 16)
            .headless()
    )
    .add_systems(FixedUpdate, random_policy_system)
    .run();

fn random_policy_system(
    mut requests: MessageReader<ActionRequestEvent>,
    mut query: Query<(&EnvId, &mut PendingAction<CartPoleEnv>)>,
) {
    for req in requests.read() {
        if let Some((_, mut pending)) = query.iter_mut().find(|(id, _)| id.0 == req.env_id) {
            pending.action = Some(rand::random::<usize>() % 2);
        }
    }
}
```

### Collecting experience for training

```rust
fn collect_experience(
    mut events: MessageReader<ExperienceEvent<Vec<f32>, usize>>,
    mut buffer: ResMut<MyReplayBuffer>,
) {
    for event in events.read() {
        buffer.push(event.experience.clone());
    }
}
```

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
```

## Message types

| Message | When fired | Contents |
|---|---|---|
| `ActionRequestEvent` | After every step and reset | `env_id`, `entity` |
| `ExperienceEvent` | After every step | `env_id`, full `Experience` tuple |
| `EpisodeEndEvent` | When an episode ends | `env_id`, `status`, `total_reward`, `episode_steps` |

## System ordering

All RL systems run in `FixedUpdate` in the `GymSet` order:

```
GymSet::Step → GymSet::AutoReset → GymSet::ManualReset
```

Policy systems can be added before `GymSet::Step` (to write actions) or after
`GymSet::ManualReset` (for post-step processing).

## Feature flags

| Feature | Description |
|---|---|
| *(default)* | Headless ECS only — no window, no rendering |
| `render` | Adds `bevy_render`, `bevy_winit`, `bevy_core_pipeline`, `bevy_asset`, `bevy_sprite` |

## Development

This crate was developed with the assistance of AI coding tools (Claude by Anthropic).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT License](LICENSE-MIT) at your option.
