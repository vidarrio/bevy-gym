# GymStatsPlugin / GymStats

Rolling-window episode statistics across all parallel environments.

## Setup

Add after `BevyGymPlugin` (it reads `EpisodeEndEvent` which the gym plugin registers):

```rust
App::new()
    .add_plugins(BevyGymPlugin::new(env_factory, 4).headless())
    .add_plugins(GymStatsPlugin::new())           // after BevyGymPlugin
    .add_systems(FixedUpdate, log_stats.after(GymSet::ManualReset))
    .run();
```

## Builder API

```rust
GymStatsPlugin::new()
    .with_window(200)   // rolling window size (default: 100)
```

## Reading stats

```rust
fn log_stats(stats: Res<GymStats>) {
    let g = stats.global();
    println!("ep {:>5}  reward mean/max: {:.1}/{:.1}  steps/sec: {:.0}",
             g.episode_count(), g.mean_reward(), g.max_reward(), g.steps_per_sec());

    // Per-environment stats (returns None until the env has finished at least one episode)
    if let Some(env0) = stats.env(0) {
        println!("env 0 mean reward: {:.1}", env0.mean_reward());
    }
}
```

## `StatsView` methods

Returned by `GymStats::global()` and `GymStats::env(id)`:

| Method | Description |
|---|---|
| `mean_reward()` | Mean episode reward over the rolling window (`NAN` if no episodes yet) |
| `max_reward()` | Max episode reward in the rolling window |
| `mean_length()` | Mean episode length over the rolling window |
| `episode_count()` | Total episodes recorded (not limited to window) |
| `total_steps()` | Total environment steps recorded (not limited to window) |
| `steps_per_sec()` | Average steps/sec since the plugin started |

## `GymStats` top-level methods

| Method | Description |
|---|---|
| `global()` | Aggregated stats across all environments |
| `env(id)` | Stats for a single environment (by `EnvId` index) |
| `total_episodes()` | Total episodes across all envs |
| `total_steps()` | Total steps across all envs |
