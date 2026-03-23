//! Bevy ECS plugin for parallelised RL environment simulation.
//!
//! `bevy-gym` bridges `rl-traits` environments into Bevy's ECS, giving you:
//!
//! - **Free parallelism**: N environment instances as separate Bevy entities,
//!   stepped in parallel via `par_iter_mut()` each `FixedUpdate` tick.
//!
//! - **Decoupled tick rate**: RL simulation runs in `FixedUpdate` at a
//!   configurable Hz. Rendering (if enabled) runs in `Update` at frame rate.
//!   The two are completely independent.
//!
//! - **Headless training mode**: disable rendering entirely and run at maximum
//!   CPU throughput via `ScheduleRunnerPlugin`.
//!
//! - **Event-driven policy integration**: `ActionRequestEvent` tells your
//!   policy system when to provide the next action. `ExperienceEvent` delivers
//!   the full `(s, a, r, s', status)` transition back to ember-rl (or any
//!   other consumer) without coupling to Bevy internals.
//!
//! # Quick start
//!
//! ```rust
//! use bevy::prelude::*;
//! use bevy_gym::{BevyGymPlugin, ActionRequestEvent, ExperienceEvent};
//! use bevy_gym::components::{CurrentObservation, PendingAction};
//!
//! // Headless training with 16 parallel environments
//! App::new()
//!     .add_plugins(MinimalPlugins)
//!     .add_plugins(
//!         BevyGymPlugin::new(|i| CartPoleEnv::new(i as u64), 16)
//!             .headless()
//!     )
//!     .add_systems(FixedUpdate, random_policy_system)
//!     .run();
//!
//! // Random policy: write a random action whenever one is requested
//! fn random_policy_system(
//!     mut requests: EventReader<ActionRequestEvent>,
//!     mut query: Query<(&EnvId, &mut PendingAction<CartPoleEnv>)>,
//! ) {
//!     for req in requests.read() {
//!         if let Some((_, mut pending)) = query.iter_mut()
//!             .find(|(id, _)| id.0 == req.env_id)
//!         {
//!             pending.action = Some(rand::random::<usize>() % 2);
//!         }
//!     }
//! }
//! ```

pub mod components;
pub mod events;
pub mod plugin;
#[cfg(feature = "render")]
pub mod render;
pub mod stats;
pub mod systems;

pub use components::{
    CurrentObservation, EnvId, EnvStats, EnvironmentComponent, PendingAction,
};
pub use events::{ActionRequestEvent, EpisodeEndEvent, ExperienceEvent};
pub use plugin::{spawn_environments, BevyGymPlugin, GymConfig, GymSet};
pub use stats::{GymStats, GymStatsPlugin, StatsView};
pub use systems::reset::ResetRequested;
