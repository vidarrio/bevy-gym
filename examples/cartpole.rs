//! CartPole-v1 trained with DQN via bevy-gym.
//!
//! Demonstrates the full ecosystem integration:
//!   rl-traits (env contract) → ember-rl (DQN + TrainingSession) → bevy-gym (parallel ECS loop)
//!
//! Four CartPole environments step in parallel each ECS tick. A single DQN
//! agent collects experience from all four, fills its replay buffer, and trains.
//! `TrainingSession` handles automatic checkpointing, JSONL episode logging,
//! and best-model saving — no manual save/checkpoint calls needed.
//!
//! # Train (default)
//!
//!   cargo run --example cartpole --release
//!
//! Saves checkpoints under `runs/bevy_cartpole/v1/<timestamp>/`.
//!
//! # Eval from a checkpoint
//!
//!   cargo run --example cartpole --release -- --eval bevy_cartpole_dqn
//!
//! Loads the checkpoint and runs the greedy policy, skipping all learning.

use std::env;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use burn::backend::{Autodiff, NdArray};
use ember_rl::{
    algorithms::dqn::{DqnAgent, DqnConfig, DqnPolicy},
    encoding::{UsizeActionMapper, VecEncoder},
    envs::cartpole::CartPoleEnv,
    training::{TrainingRun, TrainingSession},
    traits::ActMode,
};

use bevy_gym::{
    ActionRequestEvent, BevyGymPlugin, EpisodeEndEvent, ExperienceEvent, GymSet,
    GymStats, GymStatsPlugin,
    components::{CurrentObservation, PendingAction},
};

type B = Autodiff<NdArray>;
type InferB = NdArray;
type Session = TrainingSession<CartPoleEnv, DqnAgent<CartPoleEnv, VecEncoder, UsizeActionMapper, B>>;

const NUM_ENVS: usize = 4;
const MAX_STEPS: usize = 200_000;
const CHECKPOINT_FREQ: usize = 25_000;

// ── Config ────────────────────────────────────────────────────────────────────

fn cartpole_config() -> DqnConfig {
    DqnConfig {
        gamma: 0.99,
        learning_rate: 1e-3,
        batch_size: 64,
        buffer_capacity: 10_000,
        min_replay_size: 1_000,
        target_update_freq: 200,
        hidden_sizes: vec![64, 64],
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay_steps: 10_000,
    }
}

// ── Eval resource ─────────────────────────────────────────────────────────────

// DqnPolicy contains Burn tensors — not Sync. Store as NonSend.
struct DqnPolicyResource {
    policy: DqnPolicy<CartPoleEnv, VecEncoder, UsizeActionMapper, InferB>,
}

// ── Train systems ─────────────────────────────────────────────────────────────

fn policy_system(
    mut requests: MessageReader<ActionRequestEvent>,
    mut query: Query<(&CurrentObservation<CartPoleEnv>, &mut PendingAction<CartPoleEnv>)>,
    mut session: NonSendMut<Session>,
) {
    for req in requests.read() {
        if let Ok((obs, mut pending)) = query.get_mut(req.entity) {
            pending.action = Some(session.act(&obs.observation, ActMode::Explore));
        }
    }
}

fn learn_system(
    mut events: MessageReader<ExperienceEvent<Vec<f32>, usize>>,
    mut session: NonSendMut<Session>,
) {
    for event in events.read() {
        session.observe(event.experience.clone());
    }
}

fn log_and_stop_system(
    mut events: MessageReader<EpisodeEndEvent>,
    mut session: NonSendMut<Session>,
    stats: Res<GymStats>,
    mut exit: MessageWriter<AppExit>,
    mut last_logged: Local<usize>,
) {
    for event in events.read() {
        session.on_episode(
            event.total_reward,
            event.episode_steps,
            event.status.clone(),
            event.extras.clone(),
        );
    }

    let ep = stats.total_episodes();
    if ep > 0 && ep.is_multiple_of(50) && ep != *last_logged {
        *last_logged = ep;
        let g = stats.global();
        println!(
            "ep {:>5}  reward mean/max: {:>6.1}/{:>6.1}  len: {:>5.1}  steps/sec: {:.0}  ε {:.3}",
            ep, g.mean_reward(), g.max_reward(), g.mean_length(), g.steps_per_sec(),
            session.agent().epsilon(),
        );
    }

    if session.total_steps() >= MAX_STEPS {
        let g = stats.global();
        println!("\nTraining complete. Checkpoints saved to run directory.");
        println!(
            "Final stats — episodes: {}  mean reward: {:.1}  steps/sec: {:.0}",
            g.episode_count(), g.mean_reward(), g.steps_per_sec()
        );
        exit.write(AppExit::Success);
    }
}

// ── Eval systems ──────────────────────────────────────────────────────────────

fn eval_policy_system(
    mut requests: MessageReader<ActionRequestEvent>,
    mut query: Query<(&CurrentObservation<CartPoleEnv>, &mut PendingAction<CartPoleEnv>)>,
    policy: NonSend<DqnPolicyResource>,
) {
    use rl_traits::Policy;
    for req in requests.read() {
        if let Ok((obs, mut pending)) = query.get_mut(req.entity) {
            pending.action = Some(policy.policy.act(&obs.observation));
        }
    }
}

fn eval_log_and_stop_system(
    mut events: MessageReader<EpisodeEndEvent>,
    stats: Res<GymStats>,
    mut exit: MessageWriter<AppExit>,
) {
    for event in events.read() {
        println!(
            "ep {:>3}  env {}  steps {:>3}  reward {:>6.1}",
            stats.total_episodes(), event.env_id, event.episode_steps, event.total_reward,
        );
    }

    if stats.total_episodes() >= 20 {
        let g = stats.global();
        println!("\nmean reward: {:.1}  mean length: {:.1}", g.mean_reward(), g.mean_length());
        exit.write(AppExit::Success);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(pos) = args.iter().position(|a| a == "--eval") {
        let checkpoint = args
            .get(pos + 1)
            .map(|s| s.as_str())
            .unwrap_or("bevy_cartpole_dqn");
        run_eval(checkpoint);
    } else {
        run_train();
    }
}

fn run_train() {
    println!("Training DQN on CartPole-v1 ({NUM_ENVS} parallel envs, {MAX_STEPS} steps)...\n");

    let device = Default::default();
    let agent = DqnAgent::<CartPoleEnv, _, _, B>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        cartpole_config(),
        device,
        2,
    );

    let run = TrainingRun::create("bevy_cartpole", "v1").expect("failed to create training run");
    println!("Run directory: {}", run.dir().display());

    let session: Session = TrainingSession::new(agent)
        .with_run(run)
        .with_checkpoint_freq(CHECKPOINT_FREQ)
        .with_keep_checkpoints(3);

    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::ZERO,
        )))
        .add_plugins(BevyGymPlugin::new(|_| CartPoleEnv::new(), NUM_ENVS).headless())
        .add_plugins(GymStatsPlugin::new())
        .insert_non_send_resource(session)
        .add_systems(
            FixedUpdate,
            (
                policy_system.after(GymSet::ManualReset),
                learn_system.after(GymSet::ManualReset),
                log_and_stop_system.after(GymSet::ManualReset),
            ),
        )
        .run();
}

fn run_eval(checkpoint: &str) {
    println!("Loading checkpoint from {checkpoint}.mpk ...\n");

    let config = cartpole_config();
    let device: <InferB as burn::prelude::Backend>::Device = Default::default();

    let policy = DqnPolicy::<CartPoleEnv, _, _, InferB>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        &config,
        device,
    )
    .load(checkpoint)
    .expect("failed to load checkpoint — train first with: cargo run --example cartpole --release");

    println!("Running greedy evaluation (20 episodes across {NUM_ENVS} envs)...\n");

    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::ZERO,
        )))
        .add_plugins(BevyGymPlugin::new(|_| CartPoleEnv::new(), NUM_ENVS).headless())
        .add_plugins(GymStatsPlugin::new())
        .insert_non_send_resource(DqnPolicyResource { policy })
        .add_systems(
            FixedUpdate,
            (
                eval_policy_system.after(GymSet::ManualReset),
                eval_log_and_stop_system.after(GymSet::ManualReset),
            ),
        )
        .run();
}
