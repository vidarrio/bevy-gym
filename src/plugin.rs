use std::marker::PhantomData;
use std::sync::Arc;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy::time::Fixed;
use rl_traits::Environment;

use crate::components::{
    CurrentObservation, EnvId, EnvStats, EnvironmentComponent, PendingAction,
};
use crate::events::{ActionRequestEvent, EpisodeEndEvent, ExperienceEvent};
use crate::systems::{
    reset::{auto_reset_system, manual_reset_system},
    step::step_system,
};

/// Configuration for how the gym plugin runs.
#[derive(Resource, Clone)]
pub struct GymConfig {
    /// Number of parallel environment instances.
    pub num_envs: usize,

    /// Fixed tick rate in Hz. `None` means uncapped (headless training mode).
    pub tick_rate: Option<f64>,

    /// Whether to run headless (no window, no rendering).
    pub headless: bool,
}

impl Default for GymConfig {
    fn default() -> Self {
        Self {
            num_envs: 1,
            tick_rate: Some(60.0),
            headless: false,
        }
    }
}

/// System set for ordering RL systems within `FixedUpdate`.
///
/// Ordering: `Step` → `AutoReset` → `ManualReset`
///
/// This ensures resets happen in the same tick as episode completion,
/// and manual resets are processed after automatic ones.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GymSet {
    Step,
    AutoReset,
    ManualReset,
}

/// Bevy plugin that runs N parallel RL environments at a fixed tick rate.
///
/// # Usage
///
/// ```rust
/// use bevy::prelude::*;
/// use bevy_gym::BevyGymPlugin;
///
/// App::new()
///     .add_plugins(BevyGymPlugin::<MyEnv>::new(env_factory, 16).headless())
///     .add_systems(FixedUpdate, my_policy_system)
///     .run();
/// ```
///
/// # Type parameters
///
/// `E` is the `rl-traits` `Environment` implementation. The plugin is generic
/// over `E` so the full type system is available — observation and action types
/// are known at compile time throughout.
///
/// # Environment factory
///
/// Rather than requiring `E: Default`, the plugin takes a factory closure
/// `Fn(usize) -> E` where the argument is the environment index `0..num_envs`.
/// This lets you seed environments differently, give them different configs, etc.
pub struct BevyGymPlugin<E: Environment> {
    env_factory: Arc<dyn Fn(usize) -> E + Send + Sync>,
    num_envs: usize,
    tick_rate: Option<f64>,
    headless: bool,
    _phantom: PhantomData<E>,
}

impl<E: Environment + Send + Sync + 'static> BevyGymPlugin<E> {
    /// Create a new plugin with `num_envs` parallel instances.
    ///
    /// `factory` is called once per environment with the instance index.
    pub fn new(factory: impl Fn(usize) -> E + Send + Sync + 'static, num_envs: usize) -> Self {
        Self {
            env_factory: Arc::new(factory),
            num_envs,
            tick_rate: Some(60.0),
            headless: false,
            _phantom: PhantomData,
        }
    }

    /// Set the fixed tick rate in Hz.
    pub fn with_tick_rate(mut self, hz: f64) -> Self {
        self.tick_rate = Some(hz);
        self
    }

    /// Run headless at maximum speed (no window, no rendering, uncapped tick rate).
    ///
    /// Enables `ScheduleRunnerPlugin` with no sleep between ticks.
    /// Ideal for training where you want maximum environment throughput.
    pub fn headless(mut self) -> Self {
        self.headless = true;
        self.tick_rate = None;
        self
    }
}

impl<E: Environment + Send + Sync + 'static> Plugin for BevyGymPlugin<E> {
    fn build(&self, app: &mut App) {
        if self.headless {
            app.add_plugins(ScheduleRunnerPlugin::run_loop(
                std::time::Duration::ZERO,
            ));
        }

        if let Some(hz) = self.tick_rate {
            app.insert_resource(Time::<Fixed>::from_hz(hz));
        }

        app.insert_resource(GymConfig {
            num_envs: self.num_envs,
            tick_rate: self.tick_rate,
            headless: self.headless,
        });

        app.add_message::<ExperienceEvent<E::Observation, E::Action>>();
        app.add_message::<EpisodeEndEvent>();
        app.add_message::<ActionRequestEvent>();

        app.configure_sets(
            FixedUpdate,
            (GymSet::Step, GymSet::AutoReset, GymSet::ManualReset).chain(),
        );

        app.add_systems(FixedUpdate, step_system::<E>.in_set(GymSet::Step));
        app.add_systems(FixedUpdate, auto_reset_system::<E>.in_set(GymSet::AutoReset));
        app.add_systems(FixedUpdate, manual_reset_system::<E>.in_set(GymSet::ManualReset));

        let factory = Arc::clone(&self.env_factory);
        let num_envs = self.num_envs;
        app.add_systems(Startup, spawn_environments(move |i| factory(i), num_envs));
    }
}

/// Startup system that spawns all environment entities and requests their first actions.
///
/// Called automatically by `BevyGymPlugin`. Can also be used standalone if you
/// need to spawn environments separately from the plugin setup.
pub fn spawn_environments<E: Environment + Send + Sync + 'static>(
    factory: impl Fn(usize) -> E + Send + Sync + 'static,
    num_envs: usize,
) -> impl FnMut(Commands, MessageWriter<ActionRequestEvent>) {
    move |mut commands: Commands, mut action_req_writer: MessageWriter<ActionRequestEvent>| {
        for i in 0..num_envs {
            let mut env = factory(i);
            let (initial_obs, initial_info) = env.reset(Some(i as u64));

            let entity = commands
                .spawn((
                    EnvId(i),
                    EnvironmentComponent::new(env),
                    CurrentObservation::<E> {
                        observation: initial_obs,
                        info: initial_info,
                    },
                    PendingAction::<E>::default(),
                    EnvStats::default(),
                ))
                .id();

            action_req_writer.write(ActionRequestEvent {
                env_id: i,
                entity,
            });
        }
    }
}
