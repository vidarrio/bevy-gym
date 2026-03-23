use std::collections::HashMap;

use bevy::prelude::*;
use rl_traits::{EpisodeStatus, Experience};

/// Fired after every successful `step()` call.
///
/// Carries the complete `(s, a, r, s', status)` transition so ember-rl
/// (or any other subscriber) can push it into a replay buffer, update
/// a trajectory store, or log it — without coupling to Bevy internals.
///
/// # Usage in ember-rl
///
/// ```rust,ignore
/// fn collect_experience<E: Environment>(
///     mut events: MessageReader<ExperienceEvent<E::Observation, E::Action>>,
///     mut buffer: ResMut<MyReplayBuffer<E>>,
/// ) {
///     for event in events.read() {
///         buffer.push(event.experience.clone());
///     }
/// }
/// ```
#[derive(Message)]
pub struct ExperienceEvent<O, A>
where
    O: Clone + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
{
    /// Which environment instance produced this experience.
    pub env_id: usize,

    /// The full transition tuple.
    pub experience: Experience<O, A>,
}

/// Fired when an episode ends, whether by termination or truncation.
///
/// Carries the final episode statistics. Useful for logging training
/// progress without having to subscribe to every `ExperienceEvent`.
#[derive(Message, Debug, Clone)]
pub struct EpisodeEndEvent {
    /// Which environment instance finished.
    pub env_id: usize,

    /// How the episode ended.
    pub status: EpisodeStatus,

    /// Total undiscounted reward accumulated during the episode.
    pub total_reward: f64,

    /// Number of steps the episode lasted.
    pub episode_steps: usize,

    /// Optional per-episode metrics from the environment (e.g. collisions, distance).
    ///
    /// Populated from [`rl_traits::Environment::episode_extras`] if the environment
    /// overrides that method. Empty by default.
    pub extras: HashMap<String, f64>,
}

/// Fired after each step (or reset) requesting the next action.
///
/// This is how bevy-gym asks for the next action from the policy.
/// A system listens for these messages, runs inference, and writes
/// the result into `PendingAction` on the entity.
///
/// Separating "request action" from "receive action" allows the policy
/// system to batch multiple requests, run them through a neural network
/// together, and write results back — a common optimisation in deep RL.
///
/// # Usage
///
/// ```rust,ignore
/// fn policy_system<E: Environment>(
///     mut requests: MessageReader<ActionRequestEvent>,
///     mut query: Query<(&EnvId, &CurrentObservation<E>, &mut PendingAction<E>)>,
///     policy: Res<MyPolicy>,
/// ) {
///     for req in requests.read() {
///         if let Some((_, obs, mut pending)) = query.iter_mut()
///             .find(|(id, _, _)| id.0 == req.env_id)
///         {
///             pending.action = Some(policy.act(&obs.observation));
///         }
///     }
/// }
/// ```
#[derive(Message, Debug, Clone, Copy)]
pub struct ActionRequestEvent {
    /// Which environment instance needs an action.
    pub env_id: usize,

    /// The entity that holds this environment's components, for direct
    /// lookup without scanning the query.
    pub entity: Entity,
}
