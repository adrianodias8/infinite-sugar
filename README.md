# Infinite Sugar

An emulated fruit fly, embodied and given infinite sugar.

[infinitesugar.cnqso.com](https://infinitesugar.cnqso.com/)

A whole-brain emulation using the FlyWire connectome: 139,255 neurons driving a simulated
fruit fly in a terrarium. The terrarium is its world: a sack of sugar to find, a day and night,
a beach ball, all reaching the brain through the real sensory populations, split by side. Its
descending neurons decide when it walks, takes off, turns and lands. Feeding, head and antennal
motion follow neural activity; the gait, wing movements, foot shuffles and flight use supplied
patterns gated by identified descending neurons. The corner map shows sampled FlyWire positions
and connections lighting up as neurons fire.

Inspired by *Infinite Pain* (2025) by Harris Rosenblum.

## Run locally

```sh
npm ci
./serve.sh 7377
```

Open [localhost:7377](http://localhost:7377). The first load takes a few seconds.
The browser code is TypeScript, compiled to unbundled JavaScript in `dist/`.
`npm run typecheck` checks the source; `npm test` builds and checks the simulation.
The local server builds once at startup; restart it after editing source files.

## Controls

Name the fly to begin. Drag to orbit and scroll to zoom. Tap the fly to touch it. The fly lives
on its own: it sees the terrarium through its own photoreceptors, smells the sugar sack, walks
in bouts when its walking neurons ask, feeds when it reaches the sugar, takes off when
something looms, turns away from threats and toward smells, and lands wherever it is calm (see
[docs/13-world.md](docs/13-world.md) and [docs/15-vision.md](docs/15-vision.md)). World, Sugar, pause,
recenter and sound controls sit below the fly. World switches the terrarium's senses off; Sugar
overrides with infinite sugar wherever the fly is; sound (off by default) is a wingbeat tone
and a tick per step, synthesised from the same state that draws them. The environment strip
forces the other senses: touch, heat, cool, odour, bitter and damp are switches, Light also
brightens the terrarium, and Loom sends a dark sphere at the fly. A status line says what the
fly is doing in a sentence (walking toward the sugar, feeding at the sack, flying from a
threat). The camera watches from where you put it and slides round the fly when the terrarium
hides it. About contains the premise, technical summary, credits and further reading; Inspect
shows neural activity, each sense as world level, switch level and population rate side by
side, and simulation statistics.

## Credits

- [FlyWire](https://flywire.ai) — connectome data, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- [FlyBody](https://github.com/TuragaLab/flybody), via [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody) — body model.
- [MuJoCo](https://mujoco.org), [three.js](https://threejs.org) and [cannon-es](https://github.com/pmndrs/cannon-es) — physics and rendering.
- Shiu et al., *A Drosophila computational brain model reveals sensorimotor processing* (2024), and [desktop-fly](https://github.com/DenisSergeevitch/desktop-fly) by Denis Shiryaev — LIF implementation references.
- [Prop models and artists](web/model/props/CREDITS.md).

Research, measurements and implementation notes are in [docs/](docs/).
