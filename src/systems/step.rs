use std::sync::Mutex;

use bevy::prelude::*;
use rl_traits::{Environment, EpisodeStatus, Experience};

use crate::components::{CurrentObservation, EnvId, EnvStats, EnvironmentComponent, PendingAction};
use crate::events::{ActionRequestEvent, EpisodeEndEvent, ExperienceEvent};

/// The core RL tick system. Runs in `FixedUpdate`.
///
/// For each environment entity that has a `PendingAction`:
/// 1. Calls `env.step(action)`
/// 2. Updates `CurrentObservation` with the result
/// 3. Updates `EnvStats`
/// 4. Fires `ExperienceEvent` with the full transition
/// 5. Fires `EpisodeEndEvent` if the episode is over
/// 6. Fires `ActionRequestEvent` to request the next action
///
/// # Parallelism
///
/// Each environment entity's components are independent, so we use
/// `par_iter_mut()` for the compute-heavy step phase. However, Bevy's
/// `EventWriter` is not `Send`, so we can't fire events from within
/// the parallel closure directly.
///
/// Solution: collect step results into a `Mutex<Vec>` in parallel,
/// then drain and fire events serially. The serial phase is O(n) over
/// only the envs that actually stepped — the expensive part ran in parallel.
pub fn step_system<E: Environment + Send + Sync + 'static>(
    mut query: Query<(
        Entity,
        &EnvId,
        &mut EnvironmentComponent<E>,
        &mut PendingAction<E>,
        &mut CurrentObservation<E>,
        &mut EnvStats,
    )>,
    mut exp_writer: MessageWriter<ExperienceEvent<E::Observation, E::Action>>,
    mut episode_writer: MessageWriter<EpisodeEndEvent>,
    mut action_req_writer: MessageWriter<ActionRequestEvent>,
) {
    struct StepOutcome<O, A> {
        entity: Entity,
        env_id: usize,
        experience: Experience<O, A>,
        episode_done: bool,
        episode_status: EpisodeStatus,
        episode_reward: f64,
        episode_steps: usize,
    }

    let outcomes: Mutex<Vec<StepOutcome<E::Observation, E::Action>>> =
        Mutex::new(Vec::new());

    query.par_iter_mut().for_each(
        |(entity, id, mut env_comp, mut pending, mut obs, mut stats)| {
            let Some(action) = pending.action.take() else {
                return;
            };

            let prev_obs = obs.observation.clone();
            let result = env_comp.env.step(action.clone());
            let reward = result.reward;
            let status = result.status.clone();
            let episode_done = status.is_done();

            obs.observation = result.observation.clone();
            obs.info = result.info.clone();

            stats.record_step(reward);

            // Capture stats now — we can't re-query inside the serial phase
            // since par_iter_mut already has exclusive access to these components.
            let episode_reward = stats.episode_reward;
            let episode_steps = stats.episode_steps;

            let experience = Experience::new(
                prev_obs,
                action,
                reward,
                result.observation,
                result.status,
            );

            outcomes.lock().unwrap().push(StepOutcome {
                entity,
                env_id: id.0,
                experience,
                episode_done,
                episode_status: status,
                episode_reward,
                episode_steps,
            });
        },
    );

    for outcome in outcomes.into_inner().unwrap() {
        exp_writer.write(ExperienceEvent {
            env_id: outcome.env_id,
            experience: outcome.experience,
        });

        if outcome.episode_done {
            episode_writer.write(EpisodeEndEvent {
                env_id: outcome.env_id,
                status: outcome.episode_status,
                total_reward: outcome.episode_reward,
                episode_steps: outcome.episode_steps,
            });
        }

        action_req_writer.write(ActionRequestEvent {
            env_id: outcome.env_id,
            entity: outcome.entity,
        });
    }
}
