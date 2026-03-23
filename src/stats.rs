use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use bevy::prelude::*;

use crate::events::EpisodeEndEvent;

// ── Rolling window accumulator ────────────────────────────────────────────────

struct Bucket {
    rewards: VecDeque<f64>,
    lengths: VecDeque<usize>,
    window: usize,
    total_steps: usize,
    episode_count: usize,
}

impl Bucket {
    fn new(window: usize) -> Self {
        Self {
            rewards: VecDeque::with_capacity(window),
            lengths: VecDeque::with_capacity(window),
            window,
            total_steps: 0,
            episode_count: 0,
        }
    }

    fn push(&mut self, reward: f64, steps: usize) {
        if self.rewards.len() == self.window {
            self.rewards.pop_front();
            self.lengths.pop_front();
        }
        self.rewards.push_back(reward);
        self.lengths.push_back(steps);
        self.total_steps += steps;
        self.episode_count += 1;
    }

    fn mean_reward(&self) -> f64 {
        if self.rewards.is_empty() { return f64::NAN; }
        self.rewards.iter().sum::<f64>() / self.rewards.len() as f64
    }

    fn max_reward(&self) -> f64 {
        self.rewards.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    fn mean_length(&self) -> f64 {
        if self.lengths.is_empty() { return f64::NAN; }
        self.lengths.iter().sum::<usize>() as f64 / self.lengths.len() as f64
    }
}

// ── GymStats resource ─────────────────────────────────────────────────────────

/// Live training statistics, updated after every episode.
///
/// Added to the `World` by [`GymStatsPlugin`]. Read this resource from any
/// system to display or log training progress.
///
/// ```rust,ignore
/// fn log_system(stats: Res<GymStats>) {
///     let g = stats.global();
///     println!("mean reward (last 100): {:.1}  steps/sec: {:.0}",
///              g.mean_reward(), g.steps_per_sec());
/// }
/// ```
#[derive(Resource)]
pub struct GymStats {
    global: Bucket,
    per_env: HashMap<usize, Bucket>,
    window: usize,
    start: Instant,
}

impl GymStats {
    fn new(window: usize) -> Self {
        Self {
            global: Bucket::new(window),
            per_env: HashMap::new(),
            window,
            start: Instant::now(),
        }
    }

    /// Aggregated stats across all environments.
    pub fn global(&self) -> StatsView<'_> {
        StatsView {
            bucket: &self.global,
            elapsed_secs: self.start.elapsed().as_secs_f64(),
        }
    }

    /// Per-environment stats. Returns `None` if `env_id` has no episodes yet.
    pub fn env(&self, env_id: usize) -> Option<StatsView<'_>> {
        self.per_env.get(&env_id).map(|b| StatsView {
            bucket: b,
            elapsed_secs: self.start.elapsed().as_secs_f64(),
        })
    }

    /// Total episodes recorded across all environments.
    pub fn total_episodes(&self) -> usize {
        self.global.episode_count
    }

    /// Total steps recorded across all environments.
    pub fn total_steps(&self) -> usize {
        self.global.total_steps
    }

    fn record(&mut self, env_id: usize, reward: f64, steps: usize) {
        self.global.push(reward, steps);
        self.per_env
            .entry(env_id)
            .or_insert_with(|| Bucket::new(self.window))
            .push(reward, steps);
    }
}

// ── StatsView ─────────────────────────────────────────────────────────────────

/// A snapshot view into a rolling stats bucket.
///
/// Returned by [`GymStats::global()`] and [`GymStats::env()`].
pub struct StatsView<'a> {
    bucket: &'a Bucket,
    elapsed_secs: f64,
}

impl StatsView<'_> {
    /// Mean episode reward over the rolling window. `NAN` if no episodes yet.
    pub fn mean_reward(&self) -> f64 {
        self.bucket.mean_reward()
    }

    /// Maximum episode reward seen in the rolling window.
    pub fn max_reward(&self) -> f64 {
        self.bucket.max_reward()
    }

    /// Mean episode length over the rolling window. `NAN` if no episodes yet.
    pub fn mean_length(&self) -> f64 {
        self.bucket.mean_length()
    }

    /// Total episodes recorded (not limited to window).
    pub fn episode_count(&self) -> usize {
        self.bucket.episode_count
    }

    /// Total environment steps recorded (not limited to window).
    pub fn total_steps(&self) -> usize {
        self.bucket.total_steps
    }

    /// Average environment steps per wall-clock second since the plugin started.
    pub fn steps_per_sec(&self) -> f64 {
        if self.elapsed_secs < 1e-6 { return 0.0; }
        self.bucket.total_steps as f64 / self.elapsed_secs
    }
}

// ── System ────────────────────────────────────────────────────────────────────

fn update_stats_system(
    mut events: MessageReader<EpisodeEndEvent>,
    mut stats: ResMut<GymStats>,
) {
    for event in events.read() {
        stats.record(event.env_id, event.total_reward, event.episode_steps);
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────────

/// Plugin that tracks per-environment and global episode statistics.
///
/// Adds a [`GymStats`] resource updated after every episode. Must be added
/// **after** [`BevyGymPlugin`](crate::BevyGymPlugin) since it reads
/// `EpisodeEndEvent` which the gym plugin registers.
///
/// # Usage
///
/// ```rust,ignore
/// App::new()
///     .add_plugins(BevyGymPlugin::new(env_factory, 4).headless())
///     .add_plugins(GymStatsPlugin::new())          // add after BevyGymPlugin
///     .add_systems(FixedUpdate, log_stats)
///     .run();
///
/// fn log_stats(stats: Res<GymStats>) {
///     let g = stats.global();
///     if g.episode_count() % 10 == 0 && g.episode_count() > 0 {
///         println!("ep {:>5}  reward (mean/max): {:.1}/{:.1}  steps/sec: {:.0}",
///                  g.episode_count(), g.mean_reward(), g.max_reward(), g.steps_per_sec());
///     }
/// }
/// ```
pub struct GymStatsPlugin {
    /// Number of episodes to include in rolling-window averages. Default: 100.
    window: usize,
}

impl GymStatsPlugin {
    pub fn new() -> Self {
        Self { window: 100 }
    }

    /// Set the rolling-window size for averaging stats. Default: 100.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }
}

impl Default for GymStatsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for GymStatsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GymStats::new(self.window));
        app.add_systems(FixedUpdate, update_stats_system.after(crate::plugin::GymSet::ManualReset));
    }
}
