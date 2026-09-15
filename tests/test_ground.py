"""The projectile stays on top of the ground.

One invariant, asked in every place it could be broken: at the cocked pose before
anything moves, throughout a launch on both shipped machines and on a random sweep of
the search space, while the projectile is resting on the surface, and in the frames the
animations draw. "The ground" for it is the line its own *centre* rests on - one
projectile radius up, not y = 0 - because a resting sphere sits on the surface rather
than with its middle in the dirt.

Deliberately nothing else. This file used to pin which regimes a particular launch
walked, how far hand-picked designs threw, and how the energy ledger balanced across a
landing, all on geometries chosen against a loading convention. Those broke every time
the model moved for reasons that had nothing to do with the floor. The ground's own
contract is that the stone stays above it; the regimes are how, not what.

The ground is frictionless under the projectile by choice - see
physics._grounded_slack_dynamics - so nothing here charges it for sliding.

These are the reference engine's tests. That the fast engine agrees is
tests/test_fastsim.py's business.
"""

import numpy as np
import pytest

from trebuchet_sim.config import (
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import (
    GROUNDED_REGIMES,
    TrebuchetSimulator,
    simulate_trebuchet,
)


def default_params(machine: MachineType) -> TrebuchetParams:
    return TrebuchetParams(
        machine=machine, **DEFAULT_MACHINE_PARAMS[machine], **DEFAULT_MACHINE_FIXED[machine]
    )


def _traditional_with(**overrides) -> TrebuchetParams:
    """The traditional default, with fixed geometry overridden by keyword."""
    fixed = {**DEFAULT_MACHINE_FIXED[MachineType.TRADITIONAL], **overrides}
    return TrebuchetParams(
        machine=MachineType.TRADITIONAL,
        **DEFAULT_MACHINE_PARAMS[MachineType.TRADITIONAL],
        **fixed,
    )


def _min_projectile_height(result, samples: int = 3000) -> float:
    sol = result.solution
    return min(
        sol.projectile_state(float(t))[0][1] for t in np.linspace(0.0, sol.t_end, samples)
    )


@pytest.mark.parametrize("machine", list(MachineType))
def test_the_projectile_never_goes_below_the_ground(machine):
    """The whole point of the model, on the shipped machines and on a random sweep.

    Before it existed the projectile simply flew through the floor: the traditional
    machine's default geometry started it 23 mm under, and a swung arm carried it further.
    The bound is the event solver's own resolution rather than zero - a landing is
    localized by bisection, so the state it hands over sits a fraction of a micron either
    side of the line.
    """
    assert _min_projectile_height(simulate_trebuchet(default_params(machine))) >= -1e-6

    rng = np.random.default_rng(11)
    from trebuchet_sim.optimization import PARAM_BOUNDS, param_names

    checked = 0
    for _ in range(80):
        values = {name: rng.uniform(*PARAM_BOUNDS[name]) for name in param_names(machine)}
        # Stand the pivot clear of the arm, so the launch ends at the release angle rather
        # than with the beam in the dirt (see physics._first_arm_ground_angle).
        pivot = max(TrebuchetParams.pivot_height, values["arm_length"] + 0.25)
        params = TrebuchetParams(machine=machine, pivot_height=pivot, **values)
        if params.string_length > 0.95 * params.arm_length:
            continue
        result = simulate_trebuchet(params, rtol=1e-6, dense_output=False)
        if "error" in result.metrics:
            continue
        checked += 1
        assert _min_projectile_height(result, samples=1500) >= -1e-5, values

    assert checked > 30


def test_a_stone_riding_the_beam_is_set_down_on_the_ground_rather_than_through_it():
    """A beam low enough to reach the dirt can carry the stone riding it down into the dirt.

    Neither beam-contact regime used to watch the ground. On this draw - a traditional
    machine on a pivot lower than its arm, loaded on the ground - the stone came to rest on
    the beam, rode it 4.9 mm under the line, slid off into TAUT already below it, where the
    landing event is a downward crossing with no crossing left to find, and swung 137 mm
    underground. The random sweep above stands every pivot clear of its arm, so it cannot
    reach this shape.
    """
    params = TrebuchetParams(
        machine=MachineType.TRADITIONAL,
        counter_weight_mass=37.858293791252876, length_counterweight=0.5123868432570576,
        arm_length=1.4791411168880968, string_length=1.9857572592176083,
        release_angle=-0.172059576525863, pivot_height=0.5413307800065783,
        projectile_radius=0.044089843044687715, projectile_mass=0.35515697389473283,
        initial_arm_angle=0.2712100310291484,
    )
    result = simulate_trebuchet(params, t_max=5.0)
    assert "error" not in result.metrics
    ground_y = params.projectile_radius
    assert _min_projectile_height(result, samples=5000) >= ground_y - 1e-6


def test_a_stone_snatched_off_the_ground_comes_back_down_onto_it_rather_than_through_it():
    """A segment that opens on the ground line has no downward crossing left to find.

    Here a slack sling over a stone lying on the ground comes taut and picks it up at
    4 mm/s - and the taut segment that follows opens with the stone at -1.5e-16 m, the
    rounding of the line rather than above it. The landing event had never been positive,
    so when the swing brought the stone straight back down it never fired, and the stone
    went 63 mm underground. See physics.GROUND_REENTRY_SLOP.
    """
    params = TrebuchetParams(
        machine=MachineType.TRADITIONAL,
        counter_weight_mass=55.15045821206423, length_counterweight=0.8241858124568844,
        arm_length=0.2026470069447065, string_length=1.7456157974545559,
        release_angle=-0.4193342727601208, pivot_height=1.0928399106506046,
        projectile_radius=0.09446962021739262, projectile_mass=0.5051261135189091,
        initial_arm_angle=-0.5667194404709714,
    )
    result = simulate_trebuchet(params, t_max=1.0)
    assert "error" not in result.metrics
    assert _min_projectile_height(result, samples=5000) >= params.projectile_radius - 1e-6


@pytest.mark.parametrize("projectile_radius", [0.02, 0.04, 0.05, 0.0508, 0.12])
def test_the_launch_starts_on_the_ground_rather_than_under_it(projectile_radius):
    """The cocked pose, which is where the stone was buried before anything moved.

    ground_start_state solves it as geometry with no fallback in it: the sling is a circle
    of radius `string_length` about the cocked tip, the ground is the line the resting
    stone's centre lies on, and the stone starts where the two meet - the downrange root,
    the higher x, on the side the machine throws towards. Both lengths run to the stone's
    centre, so it sits on the surface rather than through it.

    The radii straddle a cliff that used to be there. The traditional machine was once
    cocked with its tip 50 mm off the ground - the angle was solved for rather than chosen
    then - so any stone of 50 mm radius or more stood taller than the tip its sling hung
    from. An ordinary stone, but a `drop < 0` bail-out read it as "cannot be loaded",
    dropped the machine back to the hanging pose, and started it 385 mm underground.
    Nothing flagged it, because a stone that begins below the surface never crosses it
    going down. 50 mm exactly is where the sign test also lost to rounding, the tip
    computing out at 0.04999999999999993. The cocked tip stands well clear of the ground
    now, but the signed drop is what keeps every one of these radii on the surface.
    """
    params = _traditional_with(projectile_radius=projectile_radius)
    simulator = TrebuchetSimulator(params)
    state = simulator.ground_start_state()

    assert state is not None
    tip_x, tip_y = simulator.arm_tip_position_velocity(simulator.initial_state())[0]
    px, py = state[2], state[3]
    assert py == projectile_radius  # on the surface, not through it
    assert px > tip_x  # the downrange root of the two the circle offers
    assert np.hypot(px - tip_x, py - tip_y) == pytest.approx(params.string_length)

    # And the launch that follows starts there and stays up. Approximately, not exactly:
    # the sling is already carrying enough load at this pose to lift the stone straight
    # off the ground, so _settle_grounded opens the launch in TAUT rather than in a
    # grounded regime, and there the projectile's position is reconstructed from the sling
    # angle instead of carried as its own state. That round trip costs the last bit or two.
    result = simulate_trebuchet(params)
    start_x, start_y = result.solution.projectile_state(0.0)[0]
    assert start_x == pytest.approx(px, abs=1e-12)
    assert start_y == pytest.approx(py, abs=1e-12)
    assert _min_projectile_height(result) >= -1e-5


def test_a_sling_too_short_to_reach_the_ground_hangs_above_it():
    """The one pose that is not solved on the ground, because there is nothing to solve.

    A pulley machine's tip stands 2.7 m up on a 33 cm sling, so the loading circle never
    meets the ground line at all and the stone really does hang. That is the other
    geometry rather than a fallback - and it still has to hang above the floor.
    """
    params = default_params(MachineType.PULLEY)
    simulator = TrebuchetSimulator(params)

    assert simulator.ground_start_state() is None
    assert simulator.simulate().solution.projectile_state(0.0)[0][1] > params.projectile_radius


@pytest.mark.parametrize("machine", list(MachineType))
def test_a_grounded_projectile_stays_exactly_on_the_ground(machine):
    """The ground is a constraint, not a restoring force: py and pvy are held, not integrated.

    Which is why the grounded regimes carry the projectile explicitly and pin two of its
    components - an integrated height would drift off the line and the sling geometry
    would drift with it. Skipped rather than asserted when a launch never touches down;
    which machines do is a property of the shipped geometry, not of the ground.
    """
    params = default_params(machine)
    launch = simulate_trebuchet(params).solution
    grounded = [seg for seg in launch.segments if seg.regime in GROUNDED_REGIMES]

    for seg in grounded:
        assert np.all(seg.sol.y[3, :] == params.projectile_radius)
        assert np.all(seg.sol.y[5, :] == 0.0)


@pytest.mark.parametrize("machine", list(MachineType))
def test_the_animations_draw_the_projectile_on_the_ground_rather_than_under_it(machine):
    """Both frontends build their frames from the solved launch, so they cannot disagree
    with the physics - but only as long as they keep sampling it rather than re-deriving
    a pose of their own. This is the check that they still do.
    """
    from trebuchet_sim.web.animation3d import _build_timeline

    params = default_params(machine)
    result = simulate_trebuchet(params, simulate_aftermath=True)
    frames = _build_timeline(params, result)["launch_frames"]

    # Against the projectile's own resting line - its centre one radius up - not
    # against y = 0. The looser bound passed for a stone drawn with its centre in
    # the dirt, i.e. buried to its equator, which is the whole error this file is
    # about and is a full radius of slack on a 40 mm ball.
    assert min(frame["projectile"][1] for frame in frames) >= params.projectile_radius - 1e-6
    assert frames[0]["projectile"] == list(result.solution.projectile_state(0.0)[0])


@pytest.mark.parametrize("machine", list(MachineType))
def test_a_machine_cocked_into_the_ground_is_refused_rather_than_simulated(machine):
    """The beam through the floor, which is how the stone used to get under it.

    A cocked pose can put the beam underground, and the beam's own ground event cannot
    catch it: `_first_arm_ground_angle` looks for the first crossing *below* the cocked
    angle, which for a pose that starts under is the angle where the beam comes back up.
    So the launch used to run happily with its tip buried and the stone hanging a sling
    length below that, until the beam surfaced - the projectile "resting on the ground"
    half a metre under it, and no ground contact recorded, because nothing had gone down
    through the surface.

    Reachable on both machines by cocking an arm longer than its pivot straight down; on
    the traditional machine it became reachable the moment that angle went back to being
    settable rather than derived by walking the arm down to the ground.
    """
    # A pivot half the arm's length, so straight down buries the tip by the other half -
    # taken from the arm rather than fixed, because the two machines' arms differ by 4x
    # and a single number would leave the pulley machine's unable to reach the ground.
    arm_length = DEFAULT_MACHINE_PARAMS[machine]["arm_length"]
    params = TrebuchetParams(
        machine=machine,
        **DEFAULT_MACHINE_PARAMS[machine],
        **{**DEFAULT_MACHINE_FIXED[machine], "pivot_height": arm_length / 2},
        initial_arm_angle=-np.pi / 2,
    )
    # The pose really is underground, so this is testing the guard and not the geometry.
    assert params.arm_length > params.pivot_height
    result = simulate_trebuchet(params)

    assert "error" in result.metrics
    assert result.distance == 0.0
    depth = result.metrics["cocked_beam_depth"]
    assert depth == pytest.approx(params.arm_length - params.pivot_height)
    # The advice has to clear the beam, and is carried in SI beside the message so a
    # caller showing ft/in can give it in ft/in (see web/app.py's error branch).
    assert result.metrics["required_pivot_height"] == pytest.approx(params.arm_length)


@pytest.mark.parametrize("machine", list(MachineType))
def test_a_machine_cocked_just_clear_of_the_ground_still_launches(machine):
    """The other side of the guard above: it must refuse only the poses that are actually
    buried, not every pose that comes near the ground. The derived traditional angle sits
    50 mm off it by construction, so a guard that was even slightly over-eager would
    reject that machine's own shipped defaults.
    """
    result = simulate_trebuchet(default_params(machine))

    assert "error" not in result.metrics
    assert result.distance > 0.0
