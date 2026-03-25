# BevyGymPlugin

The core plugin. Spawns N environment entities and steps them in parallel each `FixedUpdate` tick.

## Setup

```rust
App::new()
    .add_plugins(BevyGymPlugin::new(|i| MyEnv::new(), 4).headless())
    .add_systems(FixedUpdate, policy_system.after(GymSet::ManualReset))
    .run();
```

The factory closure receives the environment index (`0..num_envs`). Use it to seed environments differently or give them distinct configs.

## Builder API

```rust
BevyGymPlugin::new(factory, num_envs)  // factory: Fn(usize) -> E
    .with_tick_rate(120.0)             // fixed tick rate in Hz (default: 60)
    .headless()                        // uncapped tick rate, no window
```

`.headless()` sets tick rate to uncapped and skips window/rendering setup. Use this for maximum training throughput.

## System ordering (`GymSet`)

All RL systems run in `FixedUpdate` in this order:

```
GymSet::Step -> GymSet::AutoReset -> GymSet::ManualReset
```

Your policy and learning systems should run **after `GymSet::ManualReset`** to see a fully consistent state each tick.

## Message types

| Message | When fired | Key fields |
|---|---|---|
| `ActionRequestEvent` | After every step and reset | `env_id`, `entity` |
| `ExperienceEvent` | After every step | `env_id`, full `Experience` tuple |
| `EpisodeEndEvent` | When an episode ends | `env_id`, `status`, `total_reward`, `episode_steps`, `extras` |

## ECS components

Each environment entity carries:

| Component | Description |
|---|---|
| `EnvId(usize)` | Index `0..num_envs` |
| `EnvironmentComponent<E>` | The environment itself |
| `CurrentObservation<E>` | Latest observation + info |
| `PendingAction<E>` | Set by the policy system; read by the step system |
| `EnvStats` | Internal per-env step/episode counters |

## Manual resets

Insert `ResetRequested` on any environment entity to trigger a reset on the next tick:

```rust
commands.entity(env_entity).insert(ResetRequested { seed: Some(42) });
```

## Design notes

- Parallel step uses `par_iter_mut()`. Simulation runs across all CPU cores; event dispatch is serialised after.
- `E: Environment + Send + Sync + 'static` is required -- Bevy `Component` needs `Sync`.
- The factory closure is wrapped in `Arc<dyn Fn(usize) -> E>` so it can be cloned into the Startup system.
- Two RNG concerns are separated: environments seed themselves via the factory index; exploration is the policy system's responsibility.
