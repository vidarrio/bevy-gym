use bevy::prelude::*;
use rl_traits::Environment;

use crate::components::{
    CurrentObservation, EnvId, EnvStats, EnvironmentComponent, PendingAction,
};
use crate::events::{ActionRequestEvent, EpisodeEndEvent};

/// Watches for `EpisodeEndEvent`s and automatically resets the
/// corresponding environment, then fires `ActionRequestEvent` so the
/// policy knows to provide the first action of the new episode.
///
/// This runs in `FixedUpdate`, ordered *after* `step_system`. The reset
/// happens within the same tick that the episode ended, so there is never
/// a tick where an environment sits idle between episodes.
pub fn auto_reset_system<E: Environment + Send + Sync + 'static>(
    mut episode_end_events: MessageReader<EpisodeEndEvent>,
    mut query: Query<(
        Entity,
        &EnvId,
        &mut EnvironmentComponent<E>,
        &mut CurrentObservation<E>,
        &mut EnvStats,
        &mut PendingAction<E>,
    )>,
    mut action_req_writer: MessageWriter<ActionRequestEvent>,
) {
    for event in episode_end_events.read() {
        for (entity, id, mut env_comp, mut obs, mut stats, mut pending) in query.iter_mut() {
            if id.0 != event.env_id {
                continue;
            }

            let (new_obs, new_info) = env_comp.env.reset(None);
            obs.observation = new_obs;
            obs.info = new_info;
            pending.action = None;
            stats.record_episode_end();

            action_req_writer.write(ActionRequestEvent {
                env_id: id.0,
                entity,
            });

            break;
        }
    }
}

/// Marker component for environments that should be reset on the next tick.
///
/// Add this component to an environment entity to trigger a manual reset.
/// The reset system will remove the marker after resetting.
///
/// Useful for curriculum learning (reset to a specific state), evaluation
/// (reset with a fixed seed), or recovering from invalid states.
#[derive(Component)]
pub struct ResetRequested {
    /// Optional seed for deterministic reset. `None` for random.
    pub seed: Option<u64>,
}

/// Handles manually-requested resets via the `ResetRequested` marker component.
pub fn manual_reset_system<E: Environment + Send + Sync + 'static>(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &EnvId,
        &mut EnvironmentComponent<E>,
        &mut CurrentObservation<E>,
        &mut EnvStats,
        &mut PendingAction<E>,
        &ResetRequested,
    )>,
    mut action_req_writer: MessageWriter<ActionRequestEvent>,
) {
    for (entity, id, mut env_comp, mut obs, mut stats, mut pending, reset_req) in query.iter_mut() {
        let (new_obs, new_info) = env_comp.env.reset(reset_req.seed);
        obs.observation = new_obs;
        obs.info = new_info;
        pending.action = None;
        stats.record_episode_end();

        action_req_writer.write(ActionRequestEvent {
            env_id: id.0,
            entity,
        });

        commands.entity(entity).remove::<ResetRequested>();
    }
}
