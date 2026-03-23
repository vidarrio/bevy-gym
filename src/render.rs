//! Optional 2D rendering support for bevy-gym environments.
//!
//! Implement [`GymRender`] for your environment type, then add
//! [`GymRenderPlugin`] to your Bevy app. You are responsible for adding
//! a camera and Bevy's rendering plugins (e.g. `DefaultPlugins`).
//!
//! # Example
//!
//! ```rust,ignore
//! use bevy::prelude::*;
//! use bevy::sprite_render::{ColorMaterial, MeshMaterial2d};
//! use bevy_gym::render::{GymRender, GymRenderPlugin, SpawnCtx};
//!
//! #[derive(Component)]
//! struct MyVisuals { rect: Entity }
//!
//! impl GymRender for MyEnv {
//!     type Visuals = MyVisuals;
//!
//!     fn setup_visuals(entity: Entity, env_id: usize, ctx: &mut SpawnCtx) {
//!         let rect = ctx.commands.spawn((
//!             Mesh2d(ctx.meshes.add(Rectangle::new(50.0, 20.0))),
//!             MeshMaterial2d(ctx.materials.add(ColorMaterial::from_color(Color::WHITE))),
//!             Transform::default(),
//!         )).id();
//!         ctx.commands.entity(entity).insert(MyVisuals { rect });
//!     }
//!
//!     fn sync_visuals(obs: &Self::Observation, visuals: &MyVisuals, transforms: &mut Query<&mut Transform>) {
//!         if let Ok(mut t) = transforms.get_mut(visuals.rect) {
//!             t.translation.x = obs[0] * 100.0;
//!         }
//!     }
//! }
//! ```

use std::marker::PhantomData;

use bevy::prelude::*;
use rl_traits::Environment;

use crate::components::{CurrentObservation, EnvId, EnvironmentComponent};

/// Context passed to [`GymRender::setup_visuals`].
///
/// Bundles mutable access to `Commands`, `Assets<Mesh>`, and
/// `Assets<ColorMaterial>` so implementations can spawn
/// `Mesh2d` + `MeshMaterial2d` entities for solid-colored 2D shapes.
pub struct SpawnCtx<'w, 's, 'a> {
    pub commands: &'a mut Commands<'w, 's>,
    pub meshes: &'a mut Assets<Mesh>,
    pub materials: &'a mut Assets<ColorMaterial>,
}

/// Environments that know how to render themselves in Bevy's 2D world.
///
/// Implement this trait for your environment type and add [`GymRenderPlugin`]
/// to your app. Each environment entity gets a [`Self::Visuals`] component
/// inserted by [`setup_visuals`], which holds [`Entity`] handles to spawned
/// mesh children. [`sync_visuals`] repositions those children every frame.
///
/// [`setup_visuals`]: GymRender::setup_visuals
/// [`sync_visuals`]: GymRender::sync_visuals
pub trait GymRender: Environment + Send + Sync + 'static {
    /// Component attached to the environment entity that holds handles to
    /// visual child entities (e.g. cart and pole meshes).
    type Visuals: Component;

    /// Spawn visual entities and insert [`Self::Visuals`] onto `entity`.
    ///
    /// Called once when the environment entity first appears. Use
    /// [`SpawnCtx`] to spawn `Mesh2d` + `MeshMaterial2d` children and
    /// store their [`Entity`] handles in `Visuals`.
    ///
    /// `env_id` is `0..num_envs`; use it to position multiple environments
    /// at different screen locations.
    fn setup_visuals(entity: Entity, env_id: usize, ctx: &mut SpawnCtx<'_, '_, '_>);

    /// Synchronise visual transforms with the latest observation.
    ///
    /// Called every `Update` frame. Use the entity handles in `visuals`
    /// and `transforms` to reposition / re-orient your mesh entities.
    fn sync_visuals(
        obs: &Self::Observation,
        visuals: &Self::Visuals,
        transforms: &mut Query<&mut Transform>,
    );
}

/// Plugin that drives [`GymRender`] for environment type `E`.
///
/// Adds two systems:
/// - **Setup**: calls [`GymRender::setup_visuals`] once per newly spawned
///   environment entity (runs in `First`, before `FixedUpdate`).
/// - **Sync**: calls [`GymRender::sync_visuals`] every `Update` frame.
///
/// You are responsible for adding a camera and Bevy's rendering plugins
/// (e.g. `DefaultPlugins`) before adding this plugin.
pub struct GymRenderPlugin<E: GymRender> {
    _phantom: PhantomData<E>,
}

impl<E: GymRender> Default for GymRenderPlugin<E> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<E: GymRender> GymRenderPlugin<E> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<E: GymRender> Plugin for GymRenderPlugin<E> {
    fn build(&self, app: &mut App) {
        // First runs before FixedUpdate, so meshes exist before the first step.
        app.add_systems(First, setup_visuals_system::<E>);
        // Sync every render frame with the latest observation from FixedUpdate.
        app.add_systems(Update, sync_visuals_system::<E>);
    }
}

fn setup_visuals_system<E: GymRender>(
    query: Query<
        (Entity, &EnvId),
        (With<EnvironmentComponent<E>>, Without<E::Visuals>),
    >,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let pending: Vec<(Entity, usize)> = query.iter().map(|(e, id)| (e, id.0)).collect();
    for (entity, env_id) in pending {
        let mut ctx = SpawnCtx {
            commands: &mut commands,
            meshes: &mut meshes,
            materials: &mut materials,
        };
        E::setup_visuals(entity, env_id, &mut ctx);
    }
}

fn sync_visuals_system<E: GymRender>(
    query: Query<(&CurrentObservation<E>, &E::Visuals)>,
    mut transforms: Query<&mut Transform>,
) {
    for (obs, visuals) in query.iter() {
        E::sync_visuals(&obs.observation, visuals, &mut transforms);
    }
}
