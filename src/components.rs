use bevy::prelude::*;
use rl_traits::Environment;

/// The environment simulation state for one environment instance.
///
/// Wraps any `rl-traits` `Environment` implementation as a Bevy `Component`.
/// One entity per environment instance is spawned by `BevyGymPlugin`.
///
/// # Bevy ECS parallelism
///
/// Because `E` is `Send + Sync + 'static` (required by bevy-gym), and because
/// each entity's components are independent, `Query<&mut EnvironmentComponent<E>>`
/// can be iterated with `par_iter_mut()` at no extra cost.
#[derive(Component)]
pub struct EnvironmentComponent<E: Environment + Send + Sync + 'static> {
    pub env: E,
}

impl<E: Environment + Send + Sync + 'static> EnvironmentComponent<E> {
    pub fn new(env: E) -> Self {
        Self { env }
    }
}

/// The action queued for this environment's next `step()` call.
///
/// Written by whoever is providing the policy (ember-rl, a user system,
/// or the random exploration system). Consumed and cleared each tick by
/// `step_system`. If `None`, the step system skips this environment for
/// this tick -- the action has not arrived yet.
#[derive(Component)]
pub struct PendingAction<E: Environment + Send + Sync + 'static> {
    pub action: Option<E::Action>,
}

impl<E: Environment + Send + Sync + 'static> Default for PendingAction<E> {
    fn default() -> Self {
        Self { action: None }
    }
}

/// The most recent observation from this environment.
///
/// Updated every tick by `step_system` and after resets by `reset_system`.
/// Read by policy systems to produce the next action, and by rendering
/// systems to visualise the current state.
#[derive(Component)]
pub struct CurrentObservation<E: Environment + Send + Sync + 'static> {
    pub observation: E::Observation,
    pub info: E::Info,
}

/// Per-episode and overall statistics for this environment instance.
///
/// Useful for logging, debugging, and deciding when to start training.
#[derive(Component, Default)]
pub struct EnvStats {
    /// Total reward accumulated in the current episode.
    pub episode_reward: f64,

    /// Number of steps taken in the current episode.
    pub episode_steps: usize,

    /// Total number of completed episodes.
    pub total_episodes: usize,

    /// Total number of steps taken across all episodes.
    pub total_steps: usize,
}

impl EnvStats {
    pub(crate) fn record_step(&mut self, reward: f64) {
        self.episode_reward += reward;
        self.episode_steps += 1;
        self.total_steps += 1;
    }

    pub(crate) fn record_episode_end(&mut self) {
        self.total_episodes += 1;
        self.episode_reward = 0.0;
        self.episode_steps = 0;
    }
}

/// Stable index identifying this environment instance within the pool.
///
/// Ranges from `0..num_envs`. Used to correlate messages and queries
/// with specific environment entities.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvId(pub usize);
