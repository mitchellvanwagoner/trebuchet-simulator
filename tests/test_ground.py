"""The projectile's own contact with the ground.

The ground turns a launch from two regimes into four: the sling is either carrying load
or not, and the projectile is either resting on the ground or off it, independently. A
sling can go slack over a projectile already lying in the dirt, and a taut one can drag a
projectile along it - so both pairs exist and both are reachable from the cocked pose.

These are the reference engine's tests. That the fast engine walks the same four regimes
to the same answer is tests/test_fastsim.py's business.
"""

import numpy as np
import pytest

from trebuchet_sim.config import (
    DEFAULT_MACHINE_FIXED,
    DEFAULT_MACHINE_PARAMS,
    G,
    MachineType,
    TrebuchetParams,
)
from trebuchet_sim.physics import (
    GROUNDED_REGIMES,
    SLACK_GROUND,
    TAUT,
    TAUT_GROUND,
    TrebuchetSimulator,
    simulate_trebuchet,
)


def default_params(machine: MachineType) -> TrebuchetParams:
    return TrebuchetParams(
        machine=machine, **DEFAULT_MACHINE_PARAMS[machine], **DEFAULT_MACHINE_FIXED[machine]
    )


# A pulley machine that walks every regime in one launch and still throws 80 m: it swings
# taut, the sling lets go, the projectile falls to the ground and lies there, the sling
# comes taut over it and drags it along, and then picks it up and throws it. Found by the
# same random sweep of PARAM_BOUNDS tests/test_fastsim.py uses (seed 42), with its pivot
# raised to clear the arm the way that sweep raises it.
FOUR_REGIME_PARAMS = {
    "counter_weight_mass": 52.725642457680614,
    "pulley_radius": 0.6070760857653384,
    "arm_length": 1.0901717662568329,
    "string_length": 0.9980417041772385,
    "release_angle": -4.2438199212106875,
    "pivot_height": 1.3401717662568329,
}

# Two traditional machines that start the way the ground makes them start, one for each
# answer physics._initial_launch_regime can give a projectile already lying on it. The
# first has its sling laid out slack and has to take up the slack before anything happens;
# the second starts with the sling already loaded and drags the stone along the ground
# before picking it up. Both throw properly rather than merely releasing, so the regimes
# are being exercised on the way to an answer worth having.
#
# Both pin `counter_weight_rope_length`. Left unset it falls back to twice the *pulley*
# radius - a pulley machine's parameter, reached through a dataclass default the
# traditional machine has no use for - so re-tuning the pulley defaults would silently
# re-tune the link these swing on.
SLACK_GROUND_START_PARAMS = {
    "counter_weight_mass": 46.24753100586381,
    "length_counterweight": 0.5934164076240657,
    "arm_length": 1.6718386794028646,
    "string_length": 1.5592132509207859,
    "release_angle": -4.455398534412017,
    "pivot_height": 1.9668,
    "counter_weight_rope_length": 0.5,
}

TAUT_GROUND_START_PARAMS = {
    "counter_weight_mass": 27.810102143919462,
    "length_counterweight": 0.47801181071158894,
    "arm_length": 1.9436716587545877,
    "string_length": 1.1589232699419212,
    "release_angle": -4.840310454474894,
    "pivot_height": 2.3038,
    "counter_weight_rope_length": 0.5,
}


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


def test_a_sling_that_reaches_the_ground_starts_the_projectile_lying_on_it():
    """A trebuchet is loaded by laying the stone out behind it, not by dangling it.

    The projectile goes as far back from the pivot as the sling reaches, on the side the
    cocked tip leans towards - which is the side away from the throw, so the launch sweeps
    it up and across rather than dragging it back through the frame.
    """
    params = default_params(MachineType.TRADITIONAL)
    simulator = TrebuchetSimulator(params)
    state = simulator.ground_start_state()

    assert state is not None
    (px, py), (pvx, pvy) = simulator.simulate().solution.projectile_state(0.0)
    assert (px, py) == (state[2], state[3])
    assert py == 0.0
    assert (pvx, pvy) == (0.0, 0.0)

    tip_x, tip_y = simulator.arm_tip_position_velocity(simulator.initial_state())[0]
    # At full stretch, and further from the pivot than the tip is - laid out behind it.
    assert np.hypot(px - tip_x, py - tip_y) == pytest.approx(params.string_length)
    assert abs(px) > abs(tip_x)
    assert np.sign(px) == np.sign(tip_x)


def test_a_sling_too_short_to_reach_the_ground_still_hangs():
    """The pulley machine's tip stands a metre up on a 24 cm sling: nothing to rest on."""
    params = default_params(MachineType.PULLEY)
    simulator = TrebuchetSimulator(params)

    assert simulator.ground_start_state() is None
    assert simulator.simulate().solution.projectile_state(0.0)[0][1] > 0.0


def test_one_launch_walks_all_four_regimes_and_still_throws():
    result = simulate_trebuchet(TrebuchetParams(**FOUR_REGIME_PARAMS))

    assert [seg.regime for seg in result.solution.segments] == [
        "taut", "slack", "slack_ground", "taut_ground", "taut",
    ]
    assert result.metrics["release_occurred"] is True
    assert result.distance > 50.0
    # It landed once and lay there for part of the launch; the snap is the sling coming
    # taut over it again, which is a separate event from the landing that preceded it.
    assert result.metrics["projectile_ground_contacts"] == 1
    assert 0.0 < result.metrics["projectile_ground_fraction"] < 1.0
    assert result.metrics["sling_snap_count"] == 1


@pytest.mark.parametrize(
    "values, expected_first",
    [(SLACK_GROUND_START_PARAMS, SLACK_GROUND), (TAUT_GROUND_START_PARAMS, TAUT_GROUND)],
    ids=["sling slack", "sling loaded"],
)
def test_a_machine_can_start_in_the_grounded_regimes_rather_than_reach_them(
    values, expected_first
):
    """Loading the machine puts the projectile on the ground before anything moves.

    Which grounded regime it starts in is then decided by the constraint forces at that
    cocked pose rather than assumed - a sling laid out slack carries nothing until the arm
    has taken up the slack, while one already at full stretch drags the stone along the
    ground - and the third answer, a sling that snatches it straight up, is the shipped
    traditional machine's own (see test_a_sling_that_reaches_the_ground_...).
    """
    result = simulate_trebuchet(TrebuchetParams(machine=MachineType.TRADITIONAL, **values))
    regimes = [seg.regime for seg in result.solution.segments]

    assert regimes[0] == expected_first
    assert TAUT in regimes  # it does get picked up
    # It was never in the air to fall out of it, so nothing was ever taken on the way in:
    # the launch begins on the ground rather than arriving there.
    assert result.metrics["projectile_ground_contacts"] == 0
    assert result.metrics["projectile_ground_energy"] == 0.0
    assert result.metrics["projectile_ground_fraction"] > 0.0
    assert result.metrics["release_occurred"] is True
    assert result.distance > 5.0


def test_the_ground_takes_exactly_what_the_launch_records_and_never_gives():
    """A contact is an impulse, so the total energy steps down by the recorded loss.

    Both impulses in play are dissipative - the ground absorbs the projectile's downward
    momentum, and the sling transmits its share of the jerk to the arm - so the step is
    always downward. Checking the recorded number against the energy either side of the
    instant is what makes the bookkeeping an accounting of the physics rather than a
    plausible-looking number beside it.
    """
    params = TrebuchetParams(**FOUR_REGIME_PARAMS)
    simulator = TrebuchetSimulator(params)
    result = simulator.simulate()
    launch = result.solution

    assert launch.ground_times
    eps = 1e-9
    for t, recorded in zip(launch.ground_times, launch.ground_energy_losses):
        before = simulator.launch_energy_at(launch, t - eps)["total"]
        after = simulator.launch_energy_at(launch, t + eps)["total"]
        assert recorded >= 0.0
        # Sampled a nanosecond either side of the instant, so the bound is what the
        # continuous dynamics move over that nanosecond rather than the impulse's own error.
        assert before - after == pytest.approx(recorded, rel=1e-6)

    assert result.metrics["projectile_ground_energy"] == pytest.approx(
        sum(launch.ground_energy_losses)
    )
    assert result.metrics["projectile_ground_contacts"] == len(launch.ground_times)


def test_a_grounded_projectile_stays_exactly_on_the_ground():
    """The ground is a constraint, not a restoring force: py and pvy are held at zero.

    Which is why the grounded regimes carry the projectile explicitly and pin two of its
    components rather than integrating them - an integrated height would drift off the
    line and the sling geometry would drift with it.
    """
    launch = simulate_trebuchet(TrebuchetParams(**FOUR_REGIME_PARAMS)).solution
    grounded = [seg for seg in launch.segments if seg.regime in GROUNDED_REGIMES]

    assert grounded
    for seg in grounded:
        assert np.all(seg.sol.y[3, :] == 0.0)
        assert np.all(seg.sol.y[5, :] == 0.0)


def test_a_dragged_projectile_is_pressed_down_and_pulled_along():
    """Both constraint forces are one-sided, and TAUT_GROUND is where both are live.

    The ground can push up but not pull down, and the sling can pull but not push, so the
    regime lasts exactly as long as both stay non-negative - and each of them going to
    zero is the exit to a different regime.
    """
    simulator = TrebuchetSimulator(TrebuchetParams(**FOUR_REGIME_PARAMS))
    launch = simulator.simulate().solution
    dragged = [seg for seg in launch.segments if seg.regime == TAUT_GROUND]

    assert dragged
    for seg in dragged:
        for i in range(len(seg.sol.t)):
            tension, normal = simulator.grounded_forces(seg.sol.y[:, i])
            assert tension >= -1e-9
            assert normal >= -1e-9


def test_the_projectile_spends_potential_energy_from_where_it_actually_started():
    """Efficiency divides by the potential energy the launch spent, so the pose matters.

    The traditional machine's default geometry used to credit the launch with a fall from
    23 mm underground, which is where the hanging pose put a projectile that in fact rests
    on the surface.
    """
    params = default_params(MachineType.TRADITIONAL)
    result = simulate_trebuchet(params)

    assert result.solution.projectile_state(0.0)[0][1] == 0.0
    assert result.metrics["projectile_pe_spent"] == pytest.approx(
        -result.metrics["release_height"] * params.projectile_mass * G
    )


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

    assert min(frame["projectile"][1] for frame in frames) >= -1e-6
    assert frames[0]["projectile"] == list(result.solution.projectile_state(0.0)[0])
