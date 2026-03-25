# GymRender / GymRenderPlugin

Optional 2D rendering support. Requires the `render` feature.

## Setup

```toml
bevy-gym = { version = "0.3", features = ["render"] }
```

Add `DefaultPlugins`, a camera, and `GymRenderPlugin` to your app:

```rust
app.add_plugins(DefaultPlugins)
   .add_plugins(GymRenderPlugin::<MyEnv>::new())
   .add_systems(Startup, |mut commands: Commands| { commands.spawn(Camera2d); });
```

## Implementing `GymRender`

```rust
#[derive(Component)]
struct MyVisuals {
    rect: Entity,
}

impl GymRender for MyEnv {
    type Visuals = MyVisuals;

    fn setup_visuals(entity: Entity, env_id: usize, ctx: &mut SpawnCtx) {
        // Called once per env entity. Spawn Mesh2d + MeshMaterial2d children,
        // store their Entity handles in Visuals, insert onto entity.
        let rect = ctx.commands.spawn((
            Mesh2d(ctx.meshes.add(Rectangle::new(50.0, 20.0))),
            MeshMaterial2d(ctx.materials.add(ColorMaterial::from_color(Color::WHITE))),
            Transform::default(),
        )).id();
        ctx.commands.entity(entity).insert(MyVisuals { rect });
    }

    fn sync_visuals(obs: &Self::Observation, visuals: &MyVisuals, transforms: &mut Query<&mut Transform>) {
        // Called every Update frame. Reposition mesh entities from the latest observation.
        if let Ok(mut t) = transforms.get_mut(visuals.rect) {
            t.translation.x = obs[0] * 100.0;
        }
    }
}
```

## How it works

`GymRenderPlugin` adds two systems:

- **Setup** (`First`): calls `setup_visuals` once per newly spawned environment entity (runs before `FixedUpdate`).
- **Sync** (`Update`): calls `sync_visuals` every render frame, reading `CurrentObservation` which is updated by `FixedUpdate`.

The two schedules are independent -- simulation runs at `FixedUpdate` Hz, rendering at frame rate.

## `SpawnCtx`

Bundles the resources needed to spawn 2D mesh entities:

| Field | Type |
|---|---|
| `commands` | `&mut Commands` |
| `meshes` | `&mut Assets<Mesh>` |
| `materials` | `&mut Assets<ColorMaterial>` |

## Multiple environments

`env_id` (0..num_envs) is passed to `setup_visuals`. Use it to position environments at different screen locations -- e.g. stack them vertically by offsetting `y` by `env_id * SPACING`.

## Simulation speed

With rendering enabled, pass `--speed <factor>` style control via virtual time:

```rust
app.add_systems(Startup, move |mut time: ResMut<Time<Virtual>>| {
    time.set_relative_speed(speed);
});
```

This scales `FixedUpdate` Hz relative to wall-clock time. Has no effect in headless mode.
