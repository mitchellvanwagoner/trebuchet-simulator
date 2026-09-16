"""Live, physics-driven 3D trebuchet animation embedded via Three.js.

This module does not solve any physics itself. It samples the already-solved
simulation state (`SimulationResult.solution`) into a timeline, ships that
timeline to the browser as JSON, and a small Three.js scene plays it back in
real time with play/pause/scrub controls - a live game-state playback rather
than a baked GIF.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

from trebuchet_sim.config import TrebuchetParams
from trebuchet_sim.web import theme
from trebuchet_sim.physics import (
    POST_RELEASE_ENERGY_SECONDS,
    SimulationResult,
    TrebuchetSimulator,
    sample_component_positions,
)
from trebuchet_sim.trajectory import integrate_ballistic_trajectory
from trebuchet_sim.visualization import dark_plot_palette, visible_energy_series

LAUNCH_SAMPLES = 150
BALLISTIC_SAMPLES = 150
# The machine settles for longer than the stone flies on a short throw, and the energy plot
# runs that far too (physics.POST_RELEASE_ENERGY_SECONDS), so the machine is sampled over
# whichever window is longer rather than over the flight alone.
SETTLE_SAMPLES = 200

# Three.js is vendored (pinned r128 - `examples/js/OrbitControls.js` was removed
# from newer releases) and inlined into the animation HTML so it renders without
# internet access on the viewing machine. Read once at import.
_STATIC_DIR = Path(__file__).parent / "static"
_THREE_JS = (_STATIC_DIR / "three-0.128.0.min.js").read_text(encoding="utf-8")
_ORBIT_JS = (_STATIC_DIR / "OrbitControls-0.128.0.js").read_text(encoding="utf-8")


def _build_timeline(params: TrebuchetParams, result: SimulationResult) -> dict:
    """Sample the solved simulation into a JSON-serializable state timeline."""
    sol = result.solution
    simulator = TrebuchetSimulator(params)
    release_occurred = bool(result.metrics.get("release_occurred", True))

    t_release = float(sol.t_release) if sol.release_occurred else float(sol.t_end)

    t_launch = np.linspace(0, t_release, LAUNCH_SAMPLES) if t_release > 0 else np.array([0.0])

    positions = sample_component_positions(params, sol, t_launch)
    launch_frames = [
        {
            "t": float(t),
            "arm_tip": [float(tip[0]), float(tip[1])],
            "projectile": [float(proj[0]), float(proj[1])],
            "counterweight": [float(cw[0]), float(cw[1])],
            # Where the counterweight hangs from: the pivot axle on the pulley machine,
            # the arm's short end on the traditional one. The scene draws the back arm
            # and the link from it.
            "cw_pin": [float(pin[0]), float(pin[1])],
        }
        for t, tip, proj, cw, pin in zip(
            t_launch, positions["arm_tip"], positions["projectile"],
            positions["counterweight"], positions["cw_pin"],
        )
    ]

    release_frames = []
    flight_time = 0.0
    final_distance = 0.0

    if release_occurred and "error" not in result.metrics:
        # Reuse the flight already integrated by the simulation; only re-integrate if it's
        # missing (e.g. a result deserialized without it).
        trajectory = result.trajectory
        if trajectory is None and sol.release_projectile_state is not None:
            (x0, y0), (vx0, vy0) = sol.release_projectile_state
            if y0 >= 0 and not (np.isnan(vx0) or np.isnan(vy0)):
                trajectory = integrate_ballistic_trajectory(
                    x0,
                    y0,
                    vx0,
                    vy0,
                    params.projectile_mass,
                    params.projectile_drag_coefficient,
                    params.projectile_area,
                )
        if trajectory is not None:
            flight_time = float(trajectory.flight_time)
            final_distance = float(trajectory.impact_x)
            t_ballistic = np.linspace(0, flight_time, BALLISTIC_SAMPLES) if flight_time > 0 else np.array([0.0])
            for t in t_ballistic:
                x, y = trajectory.position_at(float(t))
                release_frames.append({"t": float(t), "projectile": [float(x), float(y)]})

    # Post-release machine dynamics (arm/pulley/counterweight settling, independent of
    # the ballistic flight above - see physics.TrebuchetSimulator.simulate_aftermath),
    # sampled over the same flight-time window so it can be stitched frame-for-frame
    # with release_frames. Falls back to holding the release pose if the caller didn't
    # request aftermath tracking (result.aftermath is None).
    aftermath_frames = []
    settle_time = 0.0
    if result.aftermath is not None and flight_time > 0:
        settle_time = max(flight_time, POST_RELEASE_ENERGY_SECONDS)
        t_aftermath = np.linspace(0, settle_time, SETTLE_SAMPLES)
        for t in t_aftermath:
            theta, theta_dot, regime = result.aftermath.state_at(float(t))
            # psi carried through as well: a pinned counterweight keeps swinging about
            # its pin after release, and dropping it would render the link frozen
            # straight down. Inert on the pulley machine, whose weight has no swing.
            psi, psi_dot = result.aftermath.swing_at(float(t))
            machine_state = (theta, theta_dot, 0.0, 0.0, psi, psi_dot)
            arm_tip = simulator.arm_tip_position_velocity(machine_state)[0]
            cw_pos = (
                (params.pulley_radius, params.counter_weight_size / 2)  # box resting on the ground, bottom at y=0
                if regime == "slack"
                else simulator.weight_position_velocity(machine_state)[0]
            )
            pin = simulator.counterweight_pin_position(theta)
            aftermath_frames.append(
                {"t": float(t), "arm_tip": [float(arm_tip[0]), float(arm_tip[1])],
                 "counterweight": [float(cw_pos[0]), float(cw_pos[1])],
                 "cw_pin": [float(pin[0]), float(pin[1])]}
            )

    last_launch = launch_frames[-1] if launch_frames else None

    return {
        "energy": _energy_payload(result),
        "geometry": {
            "pivot_height": params.pivot_height,
            "arm_length": params.arm_length,
            "string_length": params.string_length,
            "pulley_radius": params.pulley_radius,
            "projectile_radius": params.projectile_radius,
            "counter_weight_size": params.counter_weight_size,
            # Which linkage to build the scene around (see config.MachineType).
            "has_pulley": params.has_pulley,
            "arm_back_length": params.arm_back_length,
        },
        "launch_frames": launch_frames,
        "release_frames": release_frames,
        "aftermath_frames": aftermath_frames,
        "hold_arm_tip": last_launch["arm_tip"] if last_launch else [0.0, 0.0],
        "hold_counterweight": last_launch["counterweight"] if last_launch else [0.0, 0.0],
        "hold_cw_pin": last_launch["cw_pin"] if last_launch else [0.0, 0.0],
        "t_release": t_release,
        "flight_time": flight_time,
        # The clock every part of the page shares, the energy chart included: the launch,
        # then the longer of the stone's flight and the machine's settling.
        "total_time": t_release + max(flight_time, settle_time),
        "final_distance": final_distance,
        "release_occurred": release_occurred,
    }


def _energy_payload(result: SimulationResult) -> "dict | None":
    """The energy history as series the page can draw, or None when none was tracked.

    Shipped with the frames rather than rendered to an image because the chart plays: it
    draws itself up to the animation's current time, off the same clock, so a line arriving
    at its peak and the arm reaching the top are the same instant on screen. Which series
    appear, and in which colour, is visualization's list (see visible_energy_series), so the
    CLI's saved figure and this chart never disagree.
    """
    history = result.energy_history
    if not history:
        return None
    palette = dark_plot_palette()
    return {
        "t": [float(sample["time"]) for sample in history],
        "series": [
            {
                "label": entry.short,
                "color": palette[entry.role],
                "dashed": entry.dashed,
                # Which of the chart's two strips it belongs to, and the reason the chart has
                # two: the counterweight's store of energy dwarfs the components it becomes,
                # so they cannot share a scale (see visualization.ENERGY_SERIES).
                "panel": entry.panels[0],
                "values": [float(sample[entry.key]) for sample in history],
            }
            for entry in visible_energy_series(history)
        ],
        "release_color": palette["release"],
        "axis_color": palette["muted"],
        "grid_color": palette["grid"],
    }


# The embedded page is a plain string, so the palette is injected by placeholder
# substitution (same mechanism as the timeline/Three.js payloads) - this keeps
# the scene, its controls, and the dashboard chrome on one set of colors.
def _hex_literal(css_hex: str) -> str:
    """'#0f1724' -> '0x0f1724', the form Three.js color constructors take."""
    return "0x" + css_hex.lstrip("#")


_THEME_SUBSTITUTIONS = {
    "__SCENE_BG__": theme.SCENE_BG,
    "__TEXT__": theme.TEXT,
    "__BORDER__": theme.BORDER_STRONG,
    "__SURFACE2__": theme.SURFACE_2,
    "__ACCENT__": theme.ACCENT,
    "__SCENE_BG_HEX__": _hex_literal(theme.SCENE_BG),
    "__GRID_HEX__": _hex_literal(theme.SCENE_GRID),
    "__GROUND_HEX__": _hex_literal(theme.SCENE_GROUND),
    "__TRAJ_HEX__": _hex_literal(theme.SCENE_TRAJECTORY),
    "__TRAIL_HEX__": _hex_literal(theme.SCENE_TRAIL),
}


_HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: __SCENE_BG__; }
  /* 100vh: fill whatever height the host gives the iframe (the Streamlit app
     stretches it with CSS), falling back to the height attribute otherwise. */
  #treb-root {
    position: relative; width: 100%; height: 100vh; display: flex; flex-direction: column;
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
  }
  /* The renderer writes this one explicit pixel dimensions, so it is the child that does
     not flex: the chart and the control bar take the height they need and the scene is
     given what is left (see viewportSize). */
  #treb-canvas { display: block; flex: 0 0 auto; }
  #treb-chart-wrap {
    flex: 0 0 auto; position: relative;
    background: rgba(8, 13, 22, 0.82); border-top: 1px solid __BORDER__;
  }
  #treb-chart { display: block; width: 100%; height: 100%; }
  #treb-controls {
    flex: 0 0 auto;
    display: flex; align-items: center; gap: 10px;
    padding: 7px 12px; background: rgba(8, 13, 22, 0.82); backdrop-filter: blur(6px);
    color: __TEXT__; font-size: 12.5px; border-top: 1px solid __BORDER__;
  }
  #treb-controls button {
    background: __SURFACE2__; border: 1px solid __BORDER__; color: __TEXT__; border-radius: 6px;
    padding: 5px 11px; cursor: pointer; font-size: 12.5px; font-weight: 600;
  }
  #treb-controls button:hover { border-color: __ACCENT__; color: __ACCENT__; }
  #treb-scrub { flex: 1; }
  #treb-phase { min-width: 110px; text-align: right; opacity: 0.85; }
  #treb-time { min-width: 90px; text-align: right; font-variant-numeric: tabular-nums; }
  #treb-speed { background: __SURFACE2__; color: __TEXT__; border: 1px solid __BORDER__; border-radius: 5px; padding: 3px 4px; }
</style>
</head>
<body>
<div id="treb-root">
  <canvas id="treb-canvas"></canvas>
  <div id="treb-chart-wrap"><canvas id="treb-chart"></canvas></div>
  <div id="treb-controls">
    <button id="treb-playpause">Pause</button>
    <button id="treb-view">2D view</button>
    <input id="treb-scrub" type="range" min="0" max="1000" value="0" />
    <span id="treb-time">0.00 / 0.00 s</span>
    <select id="treb-speed">
      <option value="0.125">0.125x</option>
      <option value="0.25">0.25x</option>
      <option value="0.5" selected>0.5x</option>
      <option value="1">1x</option>
      <option value="2">2x</option>
    </select>
    <span id="treb-phase">Launching</span>
  </div>
</div>

<script>__THREE_JS__</script>
<script>__ORBIT_JS__</script>
<script>
(function () {
  const DATA = __TIMELINE_JSON__;
  const geo = DATA.geometry;
  const launchFrames = DATA.launch_frames;
  const releaseFrames = DATA.release_frames;
  const aftermathFrames = DATA.aftermath_frames;
  const tRelease = DATA.t_release;
  const totalTime = Math.max(DATA.total_time, 0.001);

  const root = document.getElementById("treb-root");
  const canvas = document.getElementById("treb-canvas");
  const chartWrap = document.getElementById("treb-chart-wrap");
  const chartCanvas = document.getElementById("treb-chart");
  const controlsBar = document.getElementById("treb-controls");

  // The energy plot lives in this page rather than beside it precisely so it can share the
  // clock below; a run with no tracked history simply has no chart and gives the scene the
  // whole frame.
  const ENERGY = DATA.energy;
  const CHART_HEIGHT = 208;
  if (ENERGY) {
    chartWrap.style.height = CHART_HEIGHT + "px";
  } else {
    chartWrap.style.display = "none";
  }

  // ---------- Scene setup ----------
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(__SCENE_BG_HEX__);
  // Fog distances are set after the camera is framed, scaled to the scene.

  const extent = Math.max(
    DATA.final_distance || 0,
    geo.arm_length + geo.string_length,
    5
  ) * 1.15;

  // Use the iframe's own viewport size, not root.clientWidth/Height: the
  // component's DOM layout may not have settled yet on first script execution,
  // but window.inner{Width,Height} reflect the host-assigned iframe size immediately.
  // The scene takes the height the chart and the control bar leave it. Their heights are
  // known here - one is set just above, the other is a styled bar already in the DOM -
  // where the canvas's own is not yet.
  function viewportSize() {
    const w = window.innerWidth || root.clientWidth || 800;
    const h = (window.innerHeight || root.clientHeight || __HEIGHT__)
      - (ENERGY ? CHART_HEIGHT : 0) - (controlsBar.offsetHeight || 44);
    return [w, Math.max(h, 140)];
  }
  let [vw, vh] = viewportSize();

  // The camera sits behind the trebuchet (negative x, like an operator's view)
  // and frames THE MACHINE, not the whole flight.
  //
  // It used to frame both, anchoring the trebuchet in one corner and the impact
  // point in the other. That is the right shot for "where did it land", and the
  // wrong one for watching the machine work: a 30 m throw is two orders of
  // magnitude longer than a 0.4 m arm, so fitting the arc renders the whole
  // trebuchet about fifteen pixels wide and a 40 mm stone smaller than a pixel.
  // Everything worth looking at up close - the sling going slack, the stone
  // resting on the ground, the beam sweeping past it - was invisible at that
  // scale, which is exactly the question the animation is there to answer.
  //
  // So the subject is the launch: the machine's own reach plus wherever the
  // stone actually goes while it is still attached. The ballistic arc leaves the
  // frame, deliberately; the faint reference path is still drawn all the way to
  // the impact point, so where the stone exits is visible, and the 2D view keeps
  // its pan/zoom for following it out.
  const machineReach = geo.arm_length + geo.string_length;
  // Kept at full flight extent: the ground plane, the fog and the reference path
  // still have to cover where the stone lands, even though the shot does not.
  let trajMaxX = Math.max(DATA.final_distance || 0, 5);
  let trajMaxY = geo.pivot_height + machineReach;
  releaseFrames.forEach((f) => {
    trajMaxX = Math.max(trajMaxX, f.projectile[0]);
    trajMaxY = Math.max(trajMaxY, f.projectile[1]);
  });

  // The framed subject: everything the machine occupies during the launch. The
  // beam's back end counts too - on the traditional machine it carries the
  // counterweight and swings the other way.
  const backReach = Math.max(geo.arm_back_length || 0, geo.pulley_radius || 0);
  let subMinX = -Math.max(machineReach, backReach), subMaxX = machineReach;
  let subMinY = 0, subMaxY = geo.pivot_height + machineReach;
  const framePoints = [];
  launchFrames.forEach((f) => {
    ["projectile", "arm_tip", "counterweight", "cw_pin"].forEach((key) => {
      const p = f[key];
      if (!p) return;
      subMinX = Math.min(subMinX, p[0]); subMaxX = Math.max(subMaxX, p[0]);
      subMinY = Math.min(subMinY, p[1]); subMaxY = Math.max(subMaxY, p[1]);
    });
  });
  [[subMinX, subMinY], [subMaxX, subMinY], [subMinX, subMaxY], [subMaxX, subMaxY]].forEach(
    ([x, y]) => framePoints.push(new THREE.Vector3(x, y, 0))
  );

  // Shot composition: the subject box's two opposite corners pinned near
  // opposite corners of the frame, so the machine fills the shot. Solved
  // numerically at load by coordinate descent over the aim point, camera
  // elevation and distance, scoring the two anchor projections plus an
  // out-of-frame penalty on the box.
  const machineRef = new THREE.Vector3(subMinX, subMinY, 0);
  const impactRef = new THREE.Vector3(subMaxX, subMaxY, 0);
  const MACHINE_NDC = { x: -0.72, y: -0.72 };
  const IMPACT_NDC = { x: 0.72, y: 0.72 };

  const lookTarget = new THREE.Vector3();
  const camera = new THREE.PerspectiveCamera(38, vw / vh, 0.05, 2000);
  const subW = Math.max(subMaxX - subMinX, 1e-3);
  const subH = Math.max(subMaxY - subMinY, 1e-3);
  const fit = {
    lookX: (subMinX + subMaxX) / 2,
    lookY: (subMinY + subMaxY) / 2,
    dirY: 0.42,
    // Start well back so every corner is in front of the camera; starting too
    // close traps the descent in a degenerate local minimum. Scaled to the
    // subject now rather than to the throw - on a 30 m shot the old seed opened
    // 33 m away from a 1 m machine and the descent never pulled all the way in.
    dist: Math.max(subW, subH) * 2.2 + 2,
  };

  function placeCamera() {
    const dir = new THREE.Vector3(-1, fit.dirY, 0.62).normalize();
    lookTarget.set(fit.lookX, fit.lookY, 0);
    camera.position.copy(lookTarget).addScaledVector(dir, fit.dist);
    camera.lookAt(lookTarget);
    camera.updateMatrixWorld(true);
  }
  function ndcOf(p) {
    const toNdc = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    return p.clone().applyMatrix4(toNdc);
  }
  function composeScore() {
    placeCamera();
    const a = ndcOf(machineRef), b = ndcOf(impactRef);
    let s =
      (a.x - MACHINE_NDC.x) ** 2 + (a.y - MACHINE_NDC.y) ** 2 +
      (b.x - IMPACT_NDC.x) ** 2 + (b.y - IMPACT_NDC.y) ** 2;
    framePoints.forEach((p) => {
      const v = ndcOf(p);
      const ox = Math.max(0, Math.abs(v.x) - 0.95);
      const oy = Math.max(0, Math.abs(v.y) - 0.95);
      s += 25 * (ox * ox + oy * oy);
      if (Math.abs(v.z) > 1) s += 10;
    });
    if (camera.position.y < 0.3) s += 10 * (0.3 - camera.position.y) ** 2;
    return s;
  }

  // Pre-fit: grow the distance until the whole scene is inside the frustum,
  // giving the descent a sane, fully-visible starting configuration.
  function allInView() {
    placeCamera();
    return framePoints.every((p) => {
      const v = ndcOf(p);
      return Math.abs(v.x) <= 0.95 && Math.abs(v.y) <= 0.95 && Math.abs(v.z) <= 1;
    });
  }
  while (!allInView() && fit.dist < 2000) fit.dist *= 1.08;

  const steps = { lookX: subW * 0.25, lookY: Math.max(subH * 0.25, 0.25), dirY: 0.15, dist: fit.dist * 0.25 };
  // The lower distance clamp has to let the camera come in close enough to fill
  // the frame with a small machine, which machineReach * 1.5 did not.
  const clamps = { dirY: [0.02, 1.2], dist: [Math.max(subW, subH) * 0.6, 2000] };
  let bestScore = composeScore();
  for (let iter = 0; iter < 40; iter++) {
    for (const k in steps) {
      for (const sign of [1, -1]) {
        const prev = fit[k];
        fit[k] = prev + sign * steps[k];
        if (clamps[k]) fit[k] = Math.min(Math.max(fit[k], clamps[k][0]), clamps[k][1]);
        const s = composeScore();
        if (s < bestScore - 1e-9) bestScore = s; else fit[k] = prev;
      }
      steps[k] *= 0.85;
    }
  }
  placeCamera();
  camera.far = Math.max(500, fit.dist * 6);
  camera.updateProjectionMatrix();
  // Debug/verification handle (used by automated UI checks; no runtime role).
  window.__treb = { camera, fit, lookTarget, ndcOf, machineRef, impactRef, composeScore };
  // Fog scaled to the framed scene so the far end of the flight stays visible.
  scene.fog = new THREE.Fog(__SCENE_BG_HEX__, fit.dist + extent, (fit.dist + extent) * 4);

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(vw, vh);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.copy(lookTarget);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  // ---------- 2D side view (orthographic camera looking down -Z) ----------
  const camera2D = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.05, 500);
  const controls2D = new THREE.OrbitControls(camera2D, renderer.domElement);
  controls2D.enableRotate = false;
  controls2D.screenSpacePanning = true;
  controls2D.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
  controls2D.enabled = false;
  let is3D = true;

  // ---------- Persist the user's camera view across Streamlit reruns ----------
  // Each simulate/optimize click rebuilds this iframe from scratch (new timeline,
  // fresh auto-fit camera), which would otherwise snap the view back to the
  // default framing on every run. Since srcdoc iframes share the parent's origin,
  // localStorage survives the reload, so we save the user's pan/zoom/rotate and
  // restore it instead of the computed auto-fit once they've touched the camera.
  // Versioned, and the version is part of the contract: a saved view is only
  // meaningful against the auto-fit it was a deviation from. When the default
  // framing changes - as it did when the shot stopped fitting the whole flight
  // and started filling the frame with the machine - every stored view is a
  // camera parked for the old composition, and restoring it silently hides the
  // new one from exactly the people who have used the thing most. Bump this
  // whenever the framing changes; the old key is simply abandoned.
  const CAMERA_STORAGE_KEY = "trebuchet3d.cameraState.v2";
  let saveTimer = null;

  function saveCameraState() {
    try {
      localStorage.setItem(CAMERA_STORAGE_KEY, JSON.stringify({
        is3D,
        cam3: { pos: camera.position.toArray(), target: controls.target.toArray(), zoom: camera.zoom },
        cam2: {
          pos: camera2D.position.toArray(), target: controls2D.target.toArray(), zoom: camera2D.zoom,
          left: camera2D.left, right: camera2D.right, top: camera2D.top, bottom: camera2D.bottom,
        },
      }));
    } catch (e) { /* localStorage unavailable (e.g. private browsing) - just skip persistence */ }
  }

  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(saveCameraState, 300);
  }

  function loadCameraState() {
    try {
      const raw = localStorage.getItem(CAMERA_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  controls.addEventListener("change", scheduleSave);
  controls2D.addEventListener("change", scheduleSave);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const sun = new THREE.DirectionalLight(0xfff2d9, 0.9);
  sun.position.set(-extent * 0.3, extent * 0.6, extent * 0.4);
  scene.add(sun);

  // ---------- Ground ----------
  const groundSize = Math.max(extent * 2.4, 20);
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(groundSize, groundSize),
    new THREE.MeshStandardMaterial({ color: __GROUND_HEX__, roughness: 1, transparent: true, opacity: 0.42 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const grid = new THREE.GridHelper(groundSize, Math.round(groundSize / 2), __GRID_HEX__, __GRID_HEX__);
  grid.position.y = 0.01;
  scene.add(grid);

  // ---------- Helper: orient a cylinder mesh between two 3D points ----------
  function setSegment(mesh, start, end) {
    const dir = new THREE.Vector3().subVectors(end, start);
    const len = Math.max(dir.length(), 1e-4);
    mesh.position.copy(start).addScaledVector(dir, 0.5);
    mesh.scale.set(1, len, 1);
    const axis = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(axis, dir.clone().normalize());
    mesh.quaternion.copy(quat);
  }

  // ---------- Trebuchet frame (A-frame legs + pivot axle) ----------
  const legOffset = Math.max(geo.arm_length * 0.18, 0.25);
  const legRadius = Math.max(geo.pivot_height * 0.035, 0.02);
  const legMaterial = new THREE.MeshStandardMaterial({ color: 0x6b4a30, roughness: 0.9 });

  function makeLegPair(zOffset) {
    const group = new THREE.Group();
    [-1, 1].forEach((side) => {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(legRadius, legRadius * 1.3, 1, 10), legMaterial);
      setSegment(
        leg,
        new THREE.Vector3(side * legOffset * 0.9, 0, zOffset),
        new THREE.Vector3(0, geo.pivot_height, 0)
      );
      group.add(leg);
    });
    return group;
  }
  scene.add(makeLegPair(legOffset));
  scene.add(makeLegPair(-legOffset));

  const axle = new THREE.Mesh(
    new THREE.CylinderGeometry(legRadius * 0.8, legRadius * 0.8, legOffset * 2 + 0.3, 10),
    new THREE.MeshStandardMaterial({ color: 0x2a2a2a })
  );
  axle.rotation.x = Math.PI / 2;
  axle.position.set(0, geo.pivot_height, 0);
  scene.add(axle);

  // Pulley at the end of the pivot axle, directly above the counterweight.
  // A torus lies in the XY plane by default, so its rotation axis already
  // points along z - the same line as the axle - and needs no rotation.
  // The traditional machine has no pulley at all: its weight is pinned to the arm, so
  // the disc is left out and the counterweight rides in the arm's own plane instead of
  // out at the axle's end.
  const pulleyZ = -(legOffset + 0.15);
  const cwZ = geo.has_pulley ? pulleyZ : 0;
  if (geo.has_pulley) {
    const pulley = new THREE.Mesh(
      new THREE.TorusGeometry(Math.max(geo.pulley_radius, 0.05), Math.max(geo.pulley_radius * 0.12, 0.015), 10, 24),
      new THREE.MeshStandardMaterial({ color: 0x8a5a2a, metalness: 0.2, roughness: 0.6 })
    );
    pulley.position.set(0, geo.pivot_height, pulleyZ);
    scene.add(pulley);
  }

  // ---------- Dynamic parts ----------
  const armRadius = Math.max(geo.arm_length * 0.03, 0.02);
  const armMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(armRadius, armRadius * 0.7, 1, 10),
    new THREE.MeshStandardMaterial({ color: 0x4a3520, roughness: 0.8 })
  );
  scene.add(armMesh);

  const slingMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(armRadius * 0.25, armRadius * 0.25, 1, 6),
    new THREE.MeshStandardMaterial({ color: 0xcccccc })
  );
  scene.add(slingMesh);

  // Traditional machine only: the beam continues behind the pivot to the pin, and the
  // counterweight hangs from that pin on a link. Both swing with the arm, so they are
  // re-drawn every frame from the sampled pin position.
  let backArmMesh = null;
  let cwLinkMesh = null;
  if (!geo.has_pulley) {
    backArmMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(armRadius, armRadius, 1, 10),
      new THREE.MeshStandardMaterial({ color: 0x4a3520, roughness: 0.8 })
    );
    scene.add(backArmMesh);
    cwLinkMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(armRadius * 0.3, armRadius * 0.3, 1, 6),
      new THREE.MeshStandardMaterial({ color: 0x8a5a2a, roughness: 0.7 })
    );
    scene.add(cwLinkMesh);
  }

  // Matches the physical cube size used by the aftermath's ground-collision check
  // (TrebuchetParams.counter_weight_size), so the box's bottom face - not its center -
  // visibly touches the ground.
  const cwSize = Math.max(geo.counter_weight_size, 0.05);
  const counterweightMesh = new THREE.Mesh(
    new THREE.BoxGeometry(cwSize, cwSize, cwSize),
    new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.3, roughness: 0.6 })
  );
  scene.add(counterweightMesh);

  const projectileMesh = new THREE.Mesh(
    new THREE.SphereGeometry(geo.projectile_radius, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xd8432e, roughness: 0.5 })
  );
  scene.add(projectileMesh);

  // Full reference path (faint, drawn once) + growing traced trail (bright)
  function buildFullPath() {
    const pts = [];
    launchFrames.forEach((f) => pts.push(new THREE.Vector3(f.projectile[0], f.projectile[1], 0)));
    releaseFrames.forEach((f) => pts.push(new THREE.Vector3(f.projectile[0], f.projectile[1], 0)));
    return pts;
  }
  const fullPathPoints = buildFullPath();
  const refLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(fullPathPoints),
    new THREE.LineBasicMaterial({ color: __TRAJ_HEX__, transparent: true, opacity: 0.3 })
  );
  scene.add(refLine);

  // Frame the 2D side view on the same subject the 3D shot uses - the machine
  // and the launch - rather than on the full projectile path. Same reasoning:
  // fitting a 30 m arc leaves the trebuchet a few pixels across. This view keeps
  // its pan/zoom (controls2D), so following the stone out to the impact point is
  // a scroll away, and the faint reference path shows where it went.
  let bMinX = subMinX, bMaxX = subMaxX, bMinY = subMinY, bMaxY = subMaxY;

  function fit2D() {
    const pad = 1.08;
    const cx = (bMinX + bMaxX) / 2, cy = (bMinY + bMaxY) / 2;
    let halfW = ((bMaxX - bMinX) / 2) * pad;
    let halfH = ((bMaxY - bMinY) / 2) * pad;
    const aspect = vw / vh;
    if (halfW / halfH > aspect) halfH = halfW / aspect; else halfW = halfH * aspect;
    camera2D.left = -halfW; camera2D.right = halfW;
    camera2D.top = halfH; camera2D.bottom = -halfH;
    camera2D.position.set(cx, cy, 60);
    camera2D.lookAt(cx, cy, 0);
    camera2D.zoom = 1;
    camera2D.updateProjectionMatrix();
    controls2D.target.set(cx, cy, 0);
    controls2D.update();
  }
  fit2D();

  // Re-letterbox the 2D view to a new aspect ratio without touching the user's
  // (or restored) pan/zoom - unlike fit2D(), which recomputes the framing from
  // scratch and would stomp on whatever view the user left it at.
  function adjustAspect2D() {
    const cx = (camera2D.left + camera2D.right) / 2;
    const cy = (camera2D.top + camera2D.bottom) / 2;
    let halfW = (camera2D.right - camera2D.left) / 2;
    let halfH = (camera2D.top - camera2D.bottom) / 2;
    const aspect = vw / vh;
    if (halfW / halfH > aspect) halfH = halfW / aspect; else halfW = halfH * aspect;
    camera2D.left = cx - halfW; camera2D.right = cx + halfW;
    camera2D.top = cy + halfH; camera2D.bottom = cy - halfH;
    camera2D.updateProjectionMatrix();
  }

  // Ground reference line for the 2D view only: the ground plane is edge-on
  // there and rasterizes to nothing. Drawn well in front (ortho, so no
  // distortion) and toggled with the view.
  const groundLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(bMinX - groundSize, 0, 45),
      new THREE.Vector3(bMaxX + groundSize, 0, 45),
    ]),
    // depthTest off: the edge-on grid lines otherwise win the depth test on
    // this exact screen row and occlude it.
    new THREE.LineBasicMaterial({ color: 0x6fae6f, fog: false, depthTest: false })
  );
  groundLine.renderOrder = 1;
  groundLine.visible = false;
  scene.add(groundLine);

  const MAX_TRAIL = 4096;
  const trailGeom = new THREE.BufferGeometry();
  const trailPositions = new Float32Array(MAX_TRAIL * 3);
  trailGeom.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
  trailGeom.setDrawRange(0, 0);
  const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: __TRAIL_HEX__, linewidth: 2 }));
  scene.add(trailLine);

  // ---------- Timeline interpolation ----------
  function lerpFrames(frames, t, key) {
    if (frames.length === 0) return [0, 0];
    if (frames.length === 1 || t <= frames[0].t) return frames[0][key];
    if (t >= frames[frames.length - 1].t) return frames[frames.length - 1][key];
    let lo = 0, hi = frames.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].t <= t) lo = mid; else hi = mid;
    }
    const a = frames[lo], b = frames[hi];
    const span = b.t - a.t;
    const frac = span > 1e-9 ? (t - a.t) / span : 0;
    return [a[key][0] + (b[key][0] - a[key][0]) * frac, a[key][1] + (b[key][1] - a[key][1]) * frac];
  }

  const pivot = new THREE.Vector3(0, geo.pivot_height, 0);
  let trailCount = 0;

  function pushTrailPoint(x, y) {
    if (trailCount >= MAX_TRAIL) return;
    trailPositions[trailCount * 3] = x;
    trailPositions[trailCount * 3 + 1] = y;
    trailPositions[trailCount * 3 + 2] = 0;
    trailCount += 1;
    trailGeom.setDrawRange(0, trailCount);
    trailGeom.attributes.position.needsUpdate = true;
  }

  function resetTrail() {
    trailCount = 0;
    trailGeom.setDrawRange(0, 0);
  }

  let lastTrailT = -1;

  // Projectile position at an absolute timeline time, spanning both phases.
  function projectileAt(t) {
    return t <= tRelease
      ? lerpFrames(launchFrames, t, "projectile")
      : lerpFrames(releaseFrames, t - tRelease, "projectile");
  }

  // Rebuild the traced trail from t=0 up to `t`, across launch AND flight.
  function rebuildTrail(t) {
    resetTrail();
    const N = 80;
    for (let i = 0; i <= N; i++) {
      const p = projectileAt((t * i) / N);
      pushTrailPoint(p[0], p[1]);
    }
    lastTrailT = t;
  }

  function updateAtTime(t) {
    let armTip, cwPos, cwPin, projPos, phase;

    if (t <= tRelease) {
      phase = "Launching";
      armTip = lerpFrames(launchFrames, t, "arm_tip");
      cwPos = lerpFrames(launchFrames, t, "counterweight");
      cwPin = lerpFrames(launchFrames, t, "cw_pin");
      projPos = lerpFrames(launchFrames, t, "projectile");
    } else {
      // The machine keeps moving after release (arm/counterweight settling under
      // their own single-pendulum dynamics - see physics.simulate_aftermath), stitched
      // here with the independently-integrated ballistic flight via a shared clock.
      if (aftermathFrames.length) {
        armTip = lerpFrames(aftermathFrames, t - tRelease, "arm_tip");
        cwPos = lerpFrames(aftermathFrames, t - tRelease, "counterweight");
        cwPin = lerpFrames(aftermathFrames, t - tRelease, "cw_pin");
      } else {
        armTip = DATA.hold_arm_tip;
        cwPos = DATA.hold_counterweight;
        cwPin = DATA.hold_cw_pin;
      }
      // Against the flight rather than the whole clock: the timeline runs to the later of
      // the landing and the machine settling, so on a short throw the stone is down well
      // before playback ends and the label would otherwise still claim it was in the air.
      if (t - tRelease >= DATA.flight_time - 1e-9) {
        phase = "Landed";
        projPos = releaseFrames.length ? releaseFrames[releaseFrames.length - 1].projectile : DATA.hold_arm_tip;
      } else {
        phase = "In flight";
        projPos = lerpFrames(releaseFrames, t - tRelease, "projectile");
      }
    }

    const armTipVec = new THREE.Vector3(armTip[0], armTip[1], 0);
    const projVec = new THREE.Vector3(projPos[0], projPos[1], 0);
    // Pulley machine: the weight hangs in the pulley's plane at the end of the axle.
    // Traditional: it is pinned to the beam, so it rides in the beam's own plane.
    const cwVec = new THREE.Vector3(cwPos[0], cwPos[1], cwZ);

    setSegment(armMesh, pivot, armTipVec);
    if (backArmMesh) {
      const pinVec = new THREE.Vector3(cwPin[0], cwPin[1], 0);
      setSegment(backArmMesh, pivot, pinVec);
      setSegment(cwLinkMesh, pinVec, cwVec);
    }
    // The sling releases the projectile at t_release: only draw it while
    // still attached, otherwise it visibly stretches across the whole
    // ballistic trajectory.
    slingMesh.visible = phase === "Launching";
    if (slingMesh.visible) setSegment(slingMesh, armTipVec, projVec);
    counterweightMesh.position.copy(cwVec);
    projectileMesh.position.copy(projVec);

    if (t < lastTrailT - 1e-6) resetTrail();
    if (t > lastTrailT + 1e-6) {
      pushTrailPoint(projVec.x, projVec.y);
      lastTrailT = t;
    }

    document.getElementById("treb-phase").textContent = phase;
    document.getElementById("treb-time").textContent = t.toFixed(2) + " / " + totalTime.toFixed(2) + " s";
    document.getElementById("treb-scrub").value = Math.round((t / totalTime) * 1000);
    drawChart(t);
  }

  // ---------- Energy chart ----------
  // Drawn here, off the same clock as the scene, rather than shipped as a finished picture:
  // each line extends to wherever playback has reached, so a curve turning over and the arm
  // reaching the top are visibly the same instant, and scrubbing moves both together.
  const chartCtx = ENERGY ? chartCanvas.getContext("2d") : null;
  let chartW = 0, chartH = 0;
  // The history can outrun the animation - it always covers the settling machine, which on
  // a long throw ends before the stone lands - so the axis spans whichever is longer and
  // the playhead simply stops short.
  const chartSpan = ENERGY ? Math.max(totalTime, ENERGY.t[ENERGY.t.length - 1] || 0) : totalTime;

  // One strip per panel, each on its own scale, for the reason the figure has always had two
  // panels: the counterweight's store of energy is what the whole throw is spent out of -
  // over a kilojoule on the shipped pulley machine - where the components it turns into are
  // tens of joules, and on one pair of axes those are a flat line along the bottom. Each
  // range is fixed for the run and padded, since a range that grew with the playhead would
  // rescale the axes underneath the lines as they draw, which reads as the curves moving.
  const strips = !ENERGY ? [] : [1, 2].map((panel) => {
    const members = ENERGY.series.filter((s) => s.panel === panel);
    let lo = 0, hi = 0;
    members.forEach((s) => s.values.forEach((v) => {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }));
    const pad = (hi - lo) * 0.08 || 1;
    return { series: members, min: lo - pad, max: hi + pad };
  }).filter((strip) => strip.series.length > 0);

  function layoutChart() {
    if (!chartCtx) return;
    const dpr = window.devicePixelRatio || 1;
    chartW = chartWrap.clientWidth || window.innerWidth || 600;
    chartH = chartWrap.clientHeight || CHART_HEIGHT;
    chartCanvas.width = Math.round(chartW * dpr);
    chartCanvas.height = Math.round(chartH * dpr);
    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function energyValueAt(series, t) {
    const ts = ENERGY.t;
    if (!ts.length || t < ts[0] || t > ts[ts.length - 1]) return null;
    let lo = 0, hi = ts.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (ts[mid] <= t) lo = mid; else hi = mid;
    }
    const span = ts[hi] - ts[lo];
    const frac = span > 1e-9 ? (t - ts[lo]) / span : 0;
    return series.values[lo] + (series.values[hi] - series.values[lo]) * frac;
  }

  // Draws one strip's legend and returns where its plot area can start; the entries wrap
  // onto as many rows as the iframe's width needs.
  function drawChartLegend(ctx, series, y) {
    ctx.font = "10px -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.textBaseline = "middle";
    let x = 8;
    series.forEach((s) => {
      const itemW = 18 + ctx.measureText(s.label).width + 12;
      if (x + itemW > chartW - 4 && x > 8) { x = 8; y += 12; }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.setLineDash(s.dashed ? [4, 3] : []);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ENERGY.axis_color;
      ctx.fillText(s.label, x + 18, y);
      x += itemW;
    });
    return y + 9;
  }

  function drawChart(t) {
    if (!chartCtx || !chartW || !strips.length) return;
    chartCtx.clearRect(0, 0, chartW, chartH);
    const axisRoom = 13;   // the time labels, which only the bottom strip carries
    const each = (chartH - axisRoom) / strips.length;
    strips.forEach((strip, i) => {
      drawStrip(chartCtx, strip, i * each, (i + 1) * each, t, i === strips.length - 1);
    });
  }

  function drawStrip(ctx, strip, boxTop, boxBottom, t, isLast) {
    const top = drawChartLegend(ctx, strip.series, boxTop + 10);
    const left = 46, right = chartW - 8, bottom = boxBottom - 3;
    if (bottom - top < 10 || right - left < 24) return;  // too short to plot into

    const xOf = (tt) => left + ((right - left) * tt) / chartSpan;
    const yOf = (v) => bottom - ((bottom - top) * (v - strip.min)) / (strip.max - strip.min);

    ctx.font = "9px -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.strokeStyle = ENERGY.grid_color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top); ctx.lineTo(left, bottom); ctx.lineTo(right, bottom);
    ctx.stroke();

    ctx.fillStyle = ENERGY.axis_color;
    ctx.textBaseline = "middle";
    ctx.fillText(strip.max.toFixed(0) + " J", 4, top + 4);
    ctx.fillText(strip.min.toFixed(0), 4, bottom - 4);
    if (isLast) {
      ctx.textBaseline = "top";
      ctx.fillText("0 s", left, bottom + 3);
      const endLabel = chartSpan.toFixed(1) + " s";
      ctx.fillText(endLabel, right - ctx.measureText(endLabel).width, bottom + 3);
    }

    // Zero, when the axes straddle it: a component going negative - the counterweight
    // below the pivot it is measured from - reads differently from one merely getting small.
    if (strip.min < 0 && strip.max > 0) {
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(left, yOf(0)); ctx.lineTo(right, yOf(0)); ctx.stroke();
      ctx.setLineDash([]);
    }

    if (tRelease > 0 && tRelease < chartSpan) {
      ctx.strokeStyle = ENERGY.release_color;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xOf(tRelease), top); ctx.lineTo(xOf(tRelease), bottom); ctx.stroke();
      ctx.setLineDash([]);
    }

    strip.series.forEach((s) => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.dashed ? 1.3 : 1.7;
      ctx.setLineDash(s.dashed ? [4, 3] : []);
      ctx.beginPath();
      let drawn = false;
      for (let i = 0; i < ENERGY.t.length; i++) {
        if (ENERGY.t[i] > t) break;
        const px = xOf(ENERGY.t[i]), py = yOf(s.values[i]);
        if (drawn) ctx.lineTo(px, py); else { ctx.moveTo(px, py); drawn = true; }
      }
      // Finish exactly at the playhead rather than at the last sample behind it, so the
      // head of the line tracks the machine instead of stepping sample to sample.
      const head = energyValueAt(s, t);
      if (head !== null) {
        const px = xOf(t), py = yOf(head);
        if (drawn) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    });

    const playX = xOf(Math.min(t, chartSpan));
    ctx.strokeStyle = ENERGY.axis_color;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(playX, top); ctx.lineTo(playX, bottom); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  layoutChart();

  // ---------- Playback loop ----------
  let playing = true;
  let currentTime = 0;
  let speed = 0.5;
  let lastFrameMs = null;

  const playPauseBtn = document.getElementById("treb-playpause");
  const viewBtn = document.getElementById("treb-view");
  const scrub = document.getElementById("treb-scrub");
  const speedSelect = document.getElementById("treb-speed");

  // Draw immediately even when requestAnimationFrame is throttled (hidden tab):
  // interactions below call this so the canvas never shows stale/blank state.
  function renderFrame() {
    renderer.render(scene, is3D ? camera : camera2D);
  }

  viewBtn.addEventListener("click", () => {
    is3D = !is3D;
    viewBtn.textContent = is3D ? "2D view" : "3D view";
    controls.enabled = is3D;
    controls2D.enabled = !is3D;
    groundLine.visible = !is3D;
    if (!is3D) fit2D();
    renderFrame();
    saveCameraState();
  });

  playPauseBtn.addEventListener("click", () => {
    // At the end of playback the same button restarts from t=0 instead of
    // offering a no-op "resume" - merges what used to be separate
    // play/pause and replay buttons.
    if (!playing && currentTime >= totalTime - 1e-6) {
      currentTime = 0;
      resetTrail();
      lastTrailT = -1;
      playing = true;
      playPauseBtn.textContent = "Pause";
      lastFrameMs = null;
      updateAtTime(0);
      renderFrame();
      return;
    }
    playing = !playing;
    playPauseBtn.textContent = playing ? "Pause" : "Play";
    lastFrameMs = null;
  });

  scrub.addEventListener("input", () => {
    playing = false;
    playPauseBtn.textContent = "Play";
    currentTime = (parseInt(scrub.value, 10) / 1000) * totalTime;
    rebuildTrail(currentTime);
    updateAtTime(currentTime);
    renderFrame();
  });

  speedSelect.addEventListener("change", () => {
    speed = parseFloat(speedSelect.value);
  });

  function tick(nowMs) {
    if (playing) {
      if (lastFrameMs !== null) {
        const dt = (nowMs - lastFrameMs) / 1000;
        currentTime = Math.min(currentTime + dt * speed, totalTime);
        if (currentTime >= totalTime) {
          playing = false;
          playPauseBtn.textContent = "Replay";
        }
      }
      lastFrameMs = nowMs;
      updateAtTime(currentTime);
    }
    if (is3D) controls.update(); else controls2D.update();
    renderer.render(scene, is3D ? camera : camera2D);
    requestAnimationFrame(tick);
  }

  // Restore the user's last camera view, if any, overriding the auto-fit computed
  // above. Falls back to the auto-fit framing on first load, cleared storage, or
  // a malformed/foreign value (e.g. a schema change) instead of crashing the IIFE
  // and leaving the animation permanently blank.
  try {
    const savedCamera = loadCameraState();
    if (savedCamera) {
      is3D = savedCamera.is3D;
      camera.position.fromArray(savedCamera.cam3.pos);
      controls.target.fromArray(savedCamera.cam3.target);
      camera.zoom = savedCamera.cam3.zoom || 1;
      camera.updateProjectionMatrix();
      controls.update();

      camera2D.position.fromArray(savedCamera.cam2.pos);
      controls2D.target.fromArray(savedCamera.cam2.target);
      camera2D.zoom = savedCamera.cam2.zoom || 1;
      camera2D.left = savedCamera.cam2.left;
      camera2D.right = savedCamera.cam2.right;
      camera2D.top = savedCamera.cam2.top;
      camera2D.bottom = savedCamera.cam2.bottom;
      camera2D.updateProjectionMatrix();
      controls2D.update();

      viewBtn.textContent = is3D ? "2D view" : "3D view";
      controls.enabled = is3D;
      controls2D.enabled = !is3D;
      groundLine.visible = !is3D;
    }
  } catch (e) { /* malformed saved state - keep the auto-fit framing computed above */ }

  updateAtTime(0);
  renderFrame();
  requestAnimationFrame(tick);

  window.addEventListener("resize", () => {
    [vw, vh] = viewportSize();
    camera.aspect = vw / vh;
    camera.updateProjectionMatrix();
    if (!is3D) adjustAspect2D();
    renderer.setSize(vw, vh);
    // The chart is a bitmap sized in device pixels, so a resize has to re-measure it and
    // redraw at the current time rather than let the browser stretch the old one.
    layoutChart();
    drawChart(currentTime);
  });
})();
</script>
</body>
</html>
"""


def build_trebuchet_3d_html(params: TrebuchetParams, result: SimulationResult, height: int = 560) -> "str | None":
    """Build the self-contained animation HTML for a solved result.

    Separate from rendering so callers (the Streamlit app) can build once per
    simulation and cache the string across reruns.
    """
    if "error" in result.metrics:
        return None
    timeline = _build_timeline(params, result)
    html = (
        _HTML_TEMPLATE
        .replace("__THREE_JS__", _THREE_JS)
        .replace("__ORBIT_JS__", _ORBIT_JS)
        .replace("__TIMELINE_JSON__", json.dumps(timeline))
        .replace("__HEIGHT__", str(height))
    )
    for placeholder, value in _THEME_SUBSTITUTIONS.items():
        html = html.replace(placeholder, value)
    return html


def render_trebuchet_3d_html(html: str, height: int = 560) -> None:
    """Embed a previously built animation HTML block in the current Streamlit page."""
    st.iframe(html, height=height)
