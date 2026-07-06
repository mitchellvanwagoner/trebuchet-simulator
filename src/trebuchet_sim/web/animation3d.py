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
from trebuchet_sim.physics import SimulationResult, TrebuchetSimulator, sample_component_positions
from trebuchet_sim.trajectory import integrate_ballistic_trajectory

LAUNCH_SAMPLES = 150
BALLISTIC_SAMPLES = 150

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

    if sol.t_events[0].size > 0:
        t_release = float(sol.t_events[0][0])
        y_release = sol.y_events[0][0]
    else:
        t_release = float(result.metrics.get("simulation_time", sol.t[-1]))
        y_release = sol.y[:, -1]

    t_launch = np.linspace(0, t_release, LAUNCH_SAMPLES) if t_release > 0 else np.array([0.0])

    positions = sample_component_positions(params, sol, t_launch)
    launch_frames = [
        {
            "t": float(t),
            "arm_tip": [float(tip[0]), float(tip[1])],
            "projectile": [float(proj[0]), float(proj[1])],
            "counterweight": [float(cw[0]), float(cw[1])],
        }
        for t, tip, proj, cw in zip(
            t_launch, positions["arm_tip"], positions["projectile"], positions["counterweight"]
        )
    ]

    release_frames = []
    flight_time = 0.0
    final_distance = 0.0

    if release_occurred and "error" not in result.metrics:
        # Reuse the flight already integrated by the simulation; only re-integrate if it's
        # missing (e.g. a result deserialized without it).
        trajectory = result.trajectory
        if trajectory is None:
            (x0, y0), (vx0, vy0) = simulator.projectile_position_velocity(y_release)
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
    if result.aftermath is not None and flight_time > 0:
        t_aftermath = np.linspace(0, flight_time, BALLISTIC_SAMPLES)
        for t in t_aftermath:
            theta, theta_dot, regime = result.aftermath.state_at(float(t))
            machine_state = (theta, theta_dot, 0.0, 0.0)
            arm_tip = simulator.arm_tip_position_velocity(machine_state)[0]
            cw_pos = (
                (params.pulley_radius, params.counter_weight_size / 2)  # box resting on the ground, bottom at y=0
                if regime == "slack"
                else simulator.weight_position_velocity(machine_state)[0]
            )
            aftermath_frames.append(
                {"t": float(t), "arm_tip": [float(arm_tip[0]), float(arm_tip[1])],
                 "counterweight": [float(cw_pos[0]), float(cw_pos[1])]}
            )

    last_launch = launch_frames[-1] if launch_frames else None

    return {
        "geometry": {
            "pivot_height": params.pivot_height,
            "arm_length": params.arm_length,
            "string_length": params.string_length,
            "pulley_radius": params.pulley_radius,
            "projectile_radius": params.projectile_radius,
            "counter_weight_size": params.counter_weight_size,
        },
        "launch_frames": launch_frames,
        "release_frames": release_frames,
        "aftermath_frames": aftermath_frames,
        "hold_arm_tip": last_launch["arm_tip"] if last_launch else [0.0, 0.0],
        "hold_counterweight": last_launch["counterweight"] if last_launch else [0.0, 0.0],
        "t_release": t_release,
        "flight_time": flight_time,
        "total_time": t_release + flight_time,
        "final_distance": final_distance,
        "release_occurred": release_occurred,
    }


_HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #0b1220; }
  /* 100vh: fill whatever height the host gives the iframe (the Streamlit app
     stretches it with CSS), falling back to the height attribute otherwise. */
  #treb-root { position: relative; width: 100%; height: 100vh; font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  #treb-canvas { width: 100%; height: 100%; display: block; }
  #treb-controls {
    position: absolute; left: 0; right: 0; bottom: 0;
    display: flex; align-items: center; gap: 10px;
    padding: 8px 14px; background: rgba(10, 14, 24, 0.72); backdrop-filter: blur(4px);
    color: #e6ecf5; font-size: 13px;
  }
  #treb-controls button {
    background: #2b6fe0; border: none; color: white; border-radius: 5px;
    padding: 6px 12px; cursor: pointer; font-size: 13px;
  }
  #treb-controls button:hover { background: #3d80f0; }
  #treb-scrub { flex: 1; }
  #treb-phase { min-width: 110px; text-align: right; opacity: 0.85; }
  #treb-time { min-width: 90px; text-align: right; font-variant-numeric: tabular-nums; }
  #treb-speed { background: #182338; color: #e6ecf5; border: 1px solid #33456b; border-radius: 4px; padding: 3px 4px; }
</style>
</head>
<body>
<div id="treb-root">
  <canvas id="treb-canvas"></canvas>
  <div id="treb-controls">
    <button id="treb-playpause">Pause</button>
    <button id="treb-view" title="Toggle between the 3D isometric view and a flat 2D side view">2D view</button>
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

  // ---------- Scene setup ----------
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1220);
  // Fog distances are set after the camera is framed, scaled to the scene.

  const extent = Math.max(
    DATA.final_distance || 0,
    geo.arm_length + geo.string_length,
    5
  ) * 1.15;

  // Use the iframe's own viewport size, not root.clientWidth/Height: the
  // component's DOM layout may not have settled yet on first script execution,
  // but window.inner{Width,Height} reflect the host-assigned iframe size immediately.
  function viewportSize() {
    return [window.innerWidth || root.clientWidth || 800, window.innerHeight || root.clientHeight || __HEIGHT__];
  }
  let [vw, vh] = viewportSize();

  // The camera sits behind the trebuchet (negative x, like an operator's view)
  // but aims downrange toward the target, pulled back just far enough that the
  // machine AND the entire flight path sit inside the view frustum.
  const machineReach = geo.arm_length + geo.string_length;
  let trajMaxX = Math.max(DATA.final_distance || 0, 5);
  let trajMaxY = geo.pivot_height + machineReach;
  const framePoints = [
    new THREE.Vector3(-machineReach, 0, 0),
    new THREE.Vector3(0, trajMaxY, 0),
    new THREE.Vector3(trajMaxX, 0, 0),
  ];
  launchFrames.concat(releaseFrames).forEach((f) => {
    trajMaxX = Math.max(trajMaxX, f.projectile[0]);
    trajMaxY = Math.max(trajMaxY, f.projectile[1]);
    framePoints.push(new THREE.Vector3(f.projectile[0], f.projectile[1], 0));
  });

  // Shot composition: the trebuchet in the bottom-left of the frame, the
  // impact point in the top-right, and the whole trajectory contained.
  // Solved numerically at load: coordinate descent over the aim point, camera
  // elevation, and distance, scoring the two anchor projections plus an
  // out-of-frame penalty for every trajectory sample.
  const machineRef = new THREE.Vector3(0, geo.pivot_height * 0.5, 0);
  const impactRef = new THREE.Vector3(trajMaxX, 0, 0);
  const MACHINE_NDC = { x: -0.78, y: -0.7 };
  const IMPACT_NDC = { x: 0.8, y: 0.6 };

  const lookTarget = new THREE.Vector3();
  const camera = new THREE.PerspectiveCamera(38, vw / vh, 0.05, 2000);
  const fit = {
    lookX: trajMaxX * 0.5,
    lookY: trajMaxY * 0.35,
    dirY: 0.42,
    // Start well behind the machine so every point is in front of the camera;
    // starting too close traps the descent in a degenerate local minimum.
    dist: Math.max(machineReach * 3, trajMaxX * 1.1, 6),
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

  const steps = { lookX: trajMaxX * 0.25, lookY: Math.max(trajMaxY * 0.25, 1), dirY: 0.15, dist: fit.dist * 0.25 };
  const clamps = { dirY: [0.02, 1.2], dist: [machineReach * 1.5, 2000] };
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
  scene.fog = new THREE.Fog(0x0b1220, fit.dist + extent, (fit.dist + extent) * 4);

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
  const CAMERA_STORAGE_KEY = "trebuchet3d.cameraState";
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
    new THREE.MeshStandardMaterial({ color: 0x3a6b3a, roughness: 1, transparent: true, opacity: 0.45 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const grid = new THREE.GridHelper(groundSize, Math.round(groundSize / 2), 0x224422, 0x1c3a1c);
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
  const pulleyZ = -(legOffset + 0.15);
  const pulley = new THREE.Mesh(
    new THREE.TorusGeometry(Math.max(geo.pulley_radius, 0.05), Math.max(geo.pulley_radius * 0.12, 0.015), 10, 24),
    new THREE.MeshStandardMaterial({ color: 0x8a5a2a, metalness: 0.2, roughness: 0.6 })
  );
  pulley.position.set(0, geo.pivot_height, pulleyZ);
  scene.add(pulley);

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
    new THREE.LineBasicMaterial({ color: 0xffb020, transparent: true, opacity: 0.28 })
  );
  scene.add(refLine);

  // Frame the 2D side view around everything that moves: the full projectile
  // path plus the trebuchet itself (arm + string reach around the pivot).
  const reach = geo.arm_length + geo.string_length;
  let bMinX = -reach, bMaxX = Math.max(5, reach), bMinY = 0, bMaxY = geo.pivot_height + reach;
  fullPathPoints.forEach((p) => {
    bMinX = Math.min(bMinX, p.x); bMaxX = Math.max(bMaxX, p.x);
    bMaxY = Math.max(bMaxY, p.y);
  });

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
  const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0xffe066, linewidth: 2 }));
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
    let armTip, cwPos, projPos, phase;

    if (t <= tRelease) {
      phase = "Launching";
      armTip = lerpFrames(launchFrames, t, "arm_tip");
      cwPos = lerpFrames(launchFrames, t, "counterweight");
      projPos = lerpFrames(launchFrames, t, "projectile");
    } else {
      // The machine keeps moving after release (arm/counterweight settling under
      // their own single-pendulum dynamics - see physics.simulate_aftermath), stitched
      // here with the independently-integrated ballistic flight via a shared clock.
      if (aftermathFrames.length) {
        armTip = lerpFrames(aftermathFrames, t - tRelease, "arm_tip");
        cwPos = lerpFrames(aftermathFrames, t - tRelease, "counterweight");
      } else {
        armTip = DATA.hold_arm_tip;
        cwPos = DATA.hold_counterweight;
      }
      if (t >= totalTime) {
        phase = "Landed";
        projPos = releaseFrames.length ? releaseFrames[releaseFrames.length - 1].projectile : DATA.hold_arm_tip;
      } else {
        phase = "In flight";
        projPos = lerpFrames(releaseFrames, t - tRelease, "projectile");
      }
    }

    const armTipVec = new THREE.Vector3(armTip[0], armTip[1], 0);
    const projVec = new THREE.Vector3(projPos[0], projPos[1], 0);
    // Counterweight hangs in the pulley's plane at the end of the axle.
    const cwVec = new THREE.Vector3(cwPos[0], cwPos[1], pulleyZ);

    setSegment(armMesh, pivot, armTipVec);
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
  }

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
    return (
        _HTML_TEMPLATE
        .replace("__THREE_JS__", _THREE_JS)
        .replace("__ORBIT_JS__", _ORBIT_JS)
        .replace("__TIMELINE_JSON__", json.dumps(timeline))
        .replace("__HEIGHT__", str(height))
    )


def render_trebuchet_3d_html(html: str, height: int = 560) -> None:
    """Embed a previously built animation HTML block in the current Streamlit page."""
    st.iframe(html, height=height)
