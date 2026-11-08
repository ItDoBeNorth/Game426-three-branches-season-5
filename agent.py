"""A Days at Three Branches agent built entirely from ``sandbox.village``.

Every villager lives by the day's phases (dawn, morning, midday, evening, night). Each tick,
``_assign`` picks an activity -- family evening, bedtime, the dawn bell and wake-up, a role-group
gathering, else the role's cell in ``_SCHEDULES`` -- and ``_decide`` runs it after reacting to
whoever is around. The lighter works the night shift: rings the dawn bell, sleeps through the
morning, lights the village from midday and keeps watch at night. Everyone else sleeps until the
bell, works their role's props (stalls and lights follow the time of day), meets their role group
in the morning and at midday, spends the evening at the inn, meets family at home and goes to bed
at staggered times. Greetings depend on the relationship and the time of day.
"""

import heapq
import math
import re

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props

# Duration (in ticks) to hold "use" on a prop. Plot and shrine are well under their catalog
# durations; pump, bell, toggles and occupancy are about 1.5x the Season 2 holds. "sleep" is
# reserved for the end-of-day return home, not idle flavor.
_TIMED_USE_TICKS = {"plot": 50, "shrine": 75, "bell": 15, "pump": 5}
_TOGGLE_USE_TICKS = 5
_OCCUPANCY_USE_TICKS = (6, 9) 
_TOGGLE_TYPES = ("lantern", "hearth", "stall")
_OCCUPANCY_TYPES = ("bench", "repair_bench")
# Catalog durations: a timed prop stays done this long after its last use, then reverts
_REVERT_TICKS = {"plot": 600, "shrine": 300, "pump": 10, "bell": 40}
_OCCUPANCY_MEMORY_TICKS = 30  # a bench seen occupied is worth checking again after this long
_ARRIVAL_LOOK_TICKS = 3  # ticks spent turning to look at a target before giving up on it
# Catalog footprints (width, height) in cells, counted from a prop's cell corner
_PROP_SIZES = {
    "stall": (2, 2),
    "lantern": (1, 1),
    "bench": (2, 1),
    "shrine": (3, 3),
    "board": (2, 2),
    "plot": (4, 2),
    "hearth": (1, 1),
    "repair_bench": (2, 1),
    "pump": (1, 1),
    "bell": (1, 1),
}
_REPOSITION_TURNS = (90.0, -90.0, 180.0)  # tried in order when "use" would pick a different prop
_ARRIVAL_SLACK = 0.75  # how close to a repositioned standing spot counts as arrived
_CLAIM_TTL_TICKS = 150  # how long a heard "I'll take care of X" keeps X off our list...
_CLAIM_AWAY_DEGREES = 90.0  # ...unless the claimer is seen walking more than this far off course from X
_HEADING_MATCH_DEGREES = 15.0  # someone walking within this angle of a prop is treated as heading there
_CLAIM_TEXT = "I'll take care of {}."
_CLAIM_PATTERN = re.compile(r"I'll take care of ([a-z0-9_]+)\.")
# Dropping a claimed prop (yielding, giving up, or off to something else) tells whoever heard the
# claim, so it doesn't keep the prop off their list -- once a stall stayed open all night that way
_RELEASE_TEXT = "I'll leave {} to you."
_RELEASE_PATTERN = re.compile(r"I'll leave ([a-z0-9_]+) to you\.")
_IDLE_EMOTES = tuple(emote for emote in action.EMOTES if emote != "sleep")
# Each prop's untouched state. Anything else means another villager already handled it (a tended
# plot, a lit lantern) or is on it right now (an occupied bench), so there is no work left here.
_PROP_START_STATE = {
    "stall": "closed",
    "lantern": "unlit",
    "bench": "empty",
    "shrine": "untended",
    "plot": "overgrown",
    "hearth": "unlit",
    "repair_bench": "idle",
    "pump": "idle",
    "bell": "silent",
}
# Day phases (rules.json). day.phase() names the current one; these bounds also cover seasons
# with day/night off, where day.phase() just says "day".
_PHASES = (("dawn", 1, 120), ("morning", 121, 480), ("midday", 481, 720), ("evening", 721, 960), ("night", 961, 1200))
# Toggle props have two states, (on, off). Which one is wanted depends on the time of day:
# stalls open for the working day, lanterns and the hearth lit from midday until the day ends.
_TOGGLE_STATES = {"stall": ("open", "closed"), "lantern": ("lit", "unlit"), "hearth": ("lit", "unlit")}
_LIGHTS_WANTED = {"dawn": "unlit", "morning": "unlit", "midday": "lit", "evening": "lit", "night": "lit"}
_PHASE_WANTED = {
    "stall": {"dawn": "open", "morning": "open", "midday": "open", "evening": "closed", "night": "closed"},
    "lantern": _LIGHTS_WANTED,
    "hearth": _LIGHTS_WANTED,
}
_LIGHTER_ONLY_PHASES = ("midday",)  # lights go on at midday by the lighter alone; others wait for evening
_STUCK_TICKS_LIMIT = 30  # abandon a destination after this many ticks of near-zero movement
_BLOCKED_TICKS = 3  # this many ticks walking without moving, with someone right beside us...
_BLOCKER_RANGE = 1.2  # ...this close (metres): the higher player number makes way
_BACK_OFF_TICKS = 3  # how long it steps back (or sidesteps) for
_BACK_OFF_MEMORY_TICKS = 30  # stepped back this recently and still blocked: sidestep instead
_FAMILY_TIME_TICKS = 40  # time spent together once family has met at the home doorway
_FAMILY_MEET_DISTANCE = 3.0  # metres; this close counts as together
# Family evening: household home_k meets at its doorway from (evening, 0.75 + 0.05 * k), so the five
# homes are staggered; anyone who hasn't met their family by the give-up time stops waiting.
_FAMILY_EVENING_START = ("evening", 0.75)
_FAMILY_EVENING_STAGGER = 0.05
_FAMILY_GIVE_UP = ("night", 0.35)
# After meeting the visitor, greetings for the rest of that phase pass the news on. Startled by an
# unseen visitor (within earshot but outside the view cone), the villager never saw who it was.
_VISITOR_NEWS = {"saw": " I saw the visitor recently.", "scared": " I got scared by an unknown person."}
_REPLY_DELAY_TICKS = 3  # beat before answering someone who greeted us
_REPLY_WINDOW_TICKS = 30  # a greeting not answered within this long is let go
_GREETING_NOTICE_RANGE = 6.0  # metres (hearing range); a wave or nod from farther isn't meant for us
_GREETING_FACING_DEGREES = 30.0  # they must be facing us this closely for a wave or nod to count
_DISTANT_WAVE_COOLDOWN_TICKS = 40  # after a silent wave at someone out of earshot, don't re-notice them
# A pair greets at most once per phase (a greeting and its reply count once). Family greet once in
# the morning (dawn counts as morning), not at all at midday, and only in passing after that.
_FAMILY_GREET_PHASES = {"dawn": "morning", "morning": "morning", "evening": "evening", "night": "night"}
_FAMILY_PASSING_PHASES = ("evening", "night")
# Greetings and replies follow the time of day: {greet}, {time} and {short} come from _PHASE_WORDS
_PHASE_WORDS = {
    "dawn": {"greet": "Good morning", "time": "morning", "short": "Morning"},
    "morning": {"greet": "Good morning", "time": "morning", "short": "Morning"},
    "midday": {"greet": "Good afternoon", "time": "afternoon", "short": "Afternoon"},
    "evening": {"greet": "Good evening", "time": "evening", "short": "Evening"},
    "night": {"greet": "Good night", "time": "night", "short": "Night"},
}
_GREETING_LINES = {
    "visitor": "{greet}, visitor.",
    "stranger": "{greet}, stranger.",
    "acquaintance": "{greet} again.",
    "friend": "{greet}, friend.",
    "family": "{greet}, family. How are you?",
}
_REPLY_LINES = {
    "visitor": "Have a good {time}.",
    "stranger": "{short}.",
    "acquaintance": "{greet} to you too.",
    "friend": "Nice {time}, isn't it?",
    "family": "I'm good, I will see you later.",
}
# How each role meets the visitor; anyone not listed stops, turns and greets as in Season 3
_VISITOR_REACTIONS = {
    "farmer": "nod",  # a nod and a word without stopping work
    "stall_tender": "point",  # points the visitor toward the stalls
    "building_wanderer": "follow",  # walks along with the visitor for a bit, like a guide
}
_VISITOR_EXTRA_LINES = {"stall_tender": " Come and see the stalls!", "building_wanderer": " Let me show you around!"}
_FOLLOW_TICKS = 40  # how long the building wanderer walks with the visitor
_FOLLOW_DISTANCE = 2.5  # metres it keeps from them
_WAVE_RELATIONS = ("visitor", "friend", "family")  # everyone else gets a nod
_STARTLE_COOLDOWN_TICKS = 20  # separate & shorter, so startling doesn't use up the greet cooldown
_GREET_DELAY_TICKS = 10  # beat spent facing someone after noticing them, before actually greeting
_VISITOR_APPROACH_RANGE = 8.0  # blocks; only detour to greet if visitor is this close when noticed
_VISITOR_GREET_DISTANCE = 2.0  # blocks; walk to this distance before actually greeting

_HOP_REACH_MIN = 5.0  # blocks; short legs toward a destination, sweeping between each
_HOP_REACH_MAX = 30.0
_OCCUPIED_CHECK_RANGE = 2.0  # blocks; someone this close to a prop already has it, pick another

_BREAK_MIN_TICKS = 15  # after completing a task, idle-hop for a random duration before next task
_BREAK_MAX_TICKS = 45

_LIGHT_TYPES = ("lantern", "hearth")

# Which prop types each role deliberately seeks out, once no one else is already on the nearest
# one. Only farmers touch plots; the building-wanderer role has no props of its own (see
# _advance_building) so it is intentionally left out of this mapping.
_ROLE_TARGET_TYPES = {
    "farmer": ("plot",),
    "shrine_keeper": ("shrine",),
    "wellkeeper": ("pump",),
    "stall_tender": ("stall",),
    "occupancy_tender": ("bench", "repair_bench"),
    "lighter": _LIGHT_TYPES,
}

# 10 seats this season: 2 farmers, 2 shrine keepers, 1 wellkeeper, 2 stall tenders, 1 occupancy
# tender, 1 lighter, 1 building-wanderer. Indexed by player number (wraps if the cast is smaller).
_ROLE_ORDER = (
    "farmer",
    "farmer",
    "shrine_keeper",
    "shrine_keeper",
    "wellkeeper",
    "stall_tender",
    "stall_tender",
    "occupancy_tender",
    "lighter",
    "building_wanderer",
)

# What each role does in each phase. _assign() checks family evening and bedtime before this table.
# "work" is the role's props (or building visits, for the wanderer); with nothing to do, roles in
# _IDLE_ROUNDS walk between their props instead of wandering off. "bell" walks to the bell and rings
# it, then sleeps. "sleep" goes home to bed. "patrol" is work for the night watch: relight anything
# out, otherwise walk the lantern rounds.
_PHASE_NAMES = tuple(name for name, _, _ in _PHASES)
_SCHEDULES = {
    #                     dawn    morning  midday  evening   night
    "farmer":            ("wake", "work",  "work", "social", "social"),
    "shrine_keeper":     ("wake", "work",  "work", "social", "social"),
    "wellkeeper":        ("wake", "work",  "work", "social", "social"),
    "stall_tender":      ("wake", "work",  "work", "social", "social"),
    "occupancy_tender":  ("wake", "work",  "work", "social", "social"),
    "lighter":           ("bell", "sleep", "work", "social", "patrol"),
    "building_wanderer": ("wake", "work",  "work", "social", "social"),
}
# "social" is the evening at the inn, until the family evening and bed. Each role first finishes its
# evening duty: a last round of work until (evening, fraction) below, and any stalls to close or
# lights to light before it goes at all.
_SOCIAL_FROM = {
    "farmer": 0.15,
    "shrine_keeper": 0.15,  # a dusk tending first
    "wellkeeper": 0.10,  # final pumping
    "stall_tender": 0.0,  # closing the stalls is the duty
    "occupancy_tender": 0.0,
    "lighter": 0.0,  # once every light is lit
    "building_wanderer": 0.0,
}
_SOCIAL_REACH = 3.0  # metres from the chosen inn spot that counts as arrived
_SOCIAL_LINGER_TICKS = (40, 80)  # time at one spot before moving to another (or joining a group)
_INN_OUTSIDE_RANGE = 4.0  # metres outside the inn doorway that still count as "at the inn"
_INN_INSIDE_SHARE = 0.7  # chance a spot is picked inside rather than out front
_CLUSTER_PAIR_RANGE = 3.0  # two idle villagers this close together make a group worth joining
_CLUSTER_NEAR_INN = 10.0  # ...as long as they're this close to the inn doorway (or inside)
_LATE_EVENING = ("evening", 0.6)  # from here the inn crowd starts to shrug (tired) now and then

# Role groups meet once in the morning and once at midday, each time at a different spot, after
# finishing whatever task they're on. Spots come from the layout so every member agrees.
_GROUPS = {
    "farmer": "fields",
    "stall_tender": "fields",
    "shrine_keeper": "keepers",
    "wellkeeper": "keepers",
    "occupancy_tender": "town",
    "building_wanderer": "town",
}  # the lighter sleeps through the day
# group -> phase -> (kind, what): "centroid" of a prop type, the first "prop" of a type, or just
# outside a building's "doorway"
_GATHER_SPOTS = {
    "fields": {"morning": ("centroid", "plot"), "midday": ("centroid", "stall")},
    "keepers": {"morning": ("prop", "shrine"), "midday": ("prop", "pump")},
    "town": {"morning": ("doorway", "shed"), "midday": ("prop", "board")},
}
_GATHER_START = {"fields": 0.25, "keepers": 0.35, "town": 0.45}  # fraction through the phase
_GATHER_LENGTH = 0.35  # of the phase; members join as they finish a task and stay until it ends
_GATHER_JOIN_MARGIN_TICKS = 20  # too late to be worth the walk with less than this left
_GATHER_REACH = 3.0  # metres from the spot that counts as there
# Farmers walk slowly across the fields and are rarely between plots while the window is open, so
# they join on the way to their next plot (that plot stays in the pool for later)
_GATHER_ON_THE_WAY = ("farmer",)
_DOORWAY_STEP_OUT = 2.5  # metres out from a doorway, so a gathering doesn't block it
# "wake": asleep in bed until the lighter rings the dawn bell. Villagers living within _BELL_RANGE
# of it walk over and gather there before greeting anyone; the rest get up at staggered times. No
# bell by the end of dawn and everyone gets up anyway.
_BELL_RANGE = 45.0  # metres from bed to the bell
_MUSTER_REACH = 4.0  # this close to the bell counts as gathered (the first ones there crowd it)
_MUSTER_TICKS = 20  # time spent together at the bell before work
_MUSTER_GIVE_UP_TICKS = 120  # still walking to the bell after this long: just start the day
_WAKE_LATEST = ("morning", 0.0)
_WAKE_STAGGER = 0.3  # out of range, villagers get up over this fraction of dawn after the bell, by player number
# The lighter gives up on ringing the bell after this and goes to bed. A little past dawn, since the
# bell can be ~85 m from home; villagers get up at the end of dawn regardless (step 4).
_BELL_DEADLINE = ("morning", 0.05)
# Roles that, with no work left, walk between their own props rather than wandering off:
# role -> (prop type, phases it applies, linger ticks (min, max), emote for passers-by while lingering)
_IDLE_ROUNDS = {
    "stall_tender": ("stall", ("dawn", "morning", "midday"), (40, 80), "point"),  # minding the open stalls
    "lighter": ("lantern", ("midday", "evening", "night"), (3, 8), "none"),  # the lantern rounds
}
_URGENT_TYPES = ("lantern", "hearth", "stall", "bell")  # trips to these are on the clock: greet in passing

# When each villager (by player number) goes to bed, as (phase, fraction through it): farmers
# first (early risers), the building wanderer last. The lighter keeps the night watch (None).
_BEDTIMES = {
    1: ("evening", 0.90),
    2: ("evening", 0.95),
    3: ("night", 0.05),
    6: ("night", 0.10),
    7: ("night", 0.15),
    5: ("night", 0.20),
    8: ("night", 0.30),
    4: ("night", 0.35),
    10: ("night", 0.45),
    9: None,
}
_DEFAULT_BEDTIME = ("night", 0.25)  # any seat past the tenth
_BED_REACH = 1.0  # metres from the spot this villager woke up in that counts as in bed
_BED_STUCK_RANGE = 4.0  # stuck this close to bed (e.g. family in the way): sleep where we stand


def _cell_centre(cell: dict[str, int]) -> dict[str, float]:
    """Return the point at the centre of one village cell."""

    return {"x": cell["x"] + 0.5, "y": cell["y"] + 0.5}


def _player_number(player_id: str) -> int:
    return int(player_id.split("_")[1])


def _role_of(player_id: str) -> str:
    """Any villager's role, computed the same way in every instance."""
    return _ROLE_ORDER[(_player_number(player_id) - 1) % len(_ROLE_ORDER)]


def _prop_centre(prop) -> dict[str, float]:
    """Middle of a prop's footprint (its ``cell`` is the corner)."""
    width, height = _PROP_SIZES.get(prop["type"], (1, 1))
    return {"x": prop["cell"]["x"] + width / 2, "y": prop["cell"]["y"] + height / 2}


def _phase_at_tick(tick: int) -> str:
    """The day phase a tick falls in (rules.json bounds)."""
    return next((phase for phase, start, end in _PHASES if start <= tick <= end), _PHASES[-1][0])


def _phase_length(tick: int) -> int:
    """How many ticks long the phase a tick falls in is."""
    return next((end - start + 1 for _, start, end in _PHASES if start <= tick <= end), _PHASES[-1][2] - _PHASES[-1][1] + 1)


def _phase(observation) -> str:
    """The current day phase; with day/night off it is worked out from the tick instead."""
    name = day.phase(observation)
    if name != "day":
        return name
    return _phase_at_tick(day.tick(observation))


def _tick_at(when) -> int:
    """The tick a (phase, fraction through it) time of day falls on."""
    phase, fraction = when
    for name, start, end in _PHASES:
        if name == phase:
            return start + round(fraction * (end - start + 1))
    raise ValueError(f"unknown phase {phase!r}")


def _reached(observation, when) -> bool:
    """Whether a (phase, fraction) time of day has come; None never comes."""
    return when is not None and day.tick(observation) >= _tick_at(when)


def _outside_doorway(observation, building_id: str):
    """A point a couple of metres out from a building's doorway, so whoever waits there leaves it clear."""
    door = layout.doorway(observation, building_id)
    centre = layout.building_center(observation, building_id)
    if door is None or centre is None:
        return door
    away = geometry.distance(centre, door) or 1.0
    return {
        "x": door["x"] + _DOORWAY_STEP_OUT * (door["x"] - centre["x"]) / away,
        "y": door["y"] + _DOORWAY_STEP_OUT * (door["y"] - centre["y"]) / away,
    }


def _gathering_spot(observation, group: str, phase: str):
    """Where a role group meets this phase (see _GATHER_SPOTS), as a walkable point, or None."""
    kind, what = _GATHER_SPOTS[group][phase]
    if kind == "doorway":
        building = next((item for item in layout.buildings(observation) if item["type"] == what), None)
        point = _outside_doorway(observation, building["id"]) if building is not None else None
    else:
        matching = sorted((prop for prop in props.all(observation) if prop["type"] == what), key=lambda prop: prop["id"])
        if not matching:
            return None
        centres = [_prop_centre(prop) for prop in (matching[:1] if kind == "prop" else matching)]
        point = {
            "x": sum(centre["x"] for centre in centres) / len(centres),
            "y": sum(centre["y"] for centre in centres) / len(centres),
        }
    cell = layout.cell_at(observation, point) if point is not None else None
    return _cell_centre(_nearest_walkable_cell(observation, cell)) if cell is not None else None


def _work_state(prop_type: str, phase: str):
    """The state a prop is in when it needs work: a toggle in the wrong state for this phase, or
    any other prop still in its start state (None for stateless props like the board)."""
    wanted = _PHASE_WANTED.get(prop_type)
    if wanted is None:
        return _PROP_START_STATE.get(prop_type)
    on, off = _TOGGLE_STATES[prop_type]
    return off if wanted[phase] == on else on


def _prop_needs_work(observation, prop) -> bool:
    """Whether a prop in reach is, right now, in the state that needs work for this phase.

    ``props.usable`` selects by reach and line of sight alone, regardless of facing, so it can
    hand back a prop that is not actually in the vision cone. ``props.state`` needs the cone, so
    that case reads as unknown here -- and an unconfirmed reading is treated as "leave it alone"
    rather than "assume it needs work", since using a lantern already right would toggle it wrong.
    """
    work = _work_state(prop["type"], _phase(observation))
    if work is None:
        return True  # e.g. "board", which has no state to speak of
    return props.state(observation, prop["id"]) == work


def _nearest_walkable_cell(observation, cell, max_radius=15):
    """Return the closest walkable cell to one that may itself be blocked (e.g. a prop's footprint)."""
    if layout.walkable(observation, cell):
        return cell
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = {"x": cell["x"] + dx, "y": cell["y"] + dy}
                if layout.walkable(observation, candidate):
                    return candidate
    return cell


_SQRT2 = math.sqrt(2.0)
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
_SEARCH_BUDGET = 5000  # heap pops before a search gives up and the goal counts as unreachable
_HEURISTIC_WEIGHT = 2.0  # above 1 the search heads for the goal: paths a bit longer, far less work


# The current day's map, and route facts about it that are the same for every villager: whether a
# body can step between two cells, and each cell's ground cost. Villagers that share a process share
# these; one in a process of its own just fills them in alone. Replaced when a new village turns up.
_DAY_MAP = {"village": None, "caches": ({}, {}, {})}


def _day_map(village):
    """This day's village and its route caches (edges, costs, neighbours), reusing the shared ones
    if it's the same village."""
    if _DAY_MAP["village"] is not village and _DAY_MAP["village"] != village:
        _DAY_MAP.update(village=village, caches=({}, {}, {}))
    return _DAY_MAP["village"], _DAY_MAP["caches"]


def _octile(node, goal):
    """Straight-line cost across an 8-way grid; never more than the real path cost."""
    dx, dy = abs(node[0] - goal[0]), abs(node[1] - goal[1])
    return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)


def _step_cost(observation, cost_cache, cell):
    """Time to cross one cell: slow ground (fields, reeds) costs more, so paths favour roads."""
    cost = cost_cache.get(cell)
    if cost is None:
        ground = layout.ground_at(observation, {"x": cell[0], "y": cell[1]})
        cost = 1.0 / max(layout.SPEED_LIMITS.get(ground, 1.0), 0.1)
        cost_cache[cell] = cost
    return cost


def _neighbours(observation, caches, node):
    """The cells one step from ``node`` that a body can reach, each with the time the step takes.

    A diagonal step is only allowed when both cells beside it are clear too, so a body never clips
    a wall corner. Worked out once per cell per day (``caches`` = edge, cost, neighbour caches):
    the layout never changes, so every later search just looks the answer up.
    """
    edge_cache, cost_cache, neighbour_cache = caches
    known = neighbour_cache.get(node)
    if known is not None:
        return known

    def clear(first, second):
        # can_step is symmetric (both cells walkable, the body clears the segment between their
        # centres), so one answer covers both directions -- half the calls, its biggest cost
        key = (first, second) if first < second else (second, first)
        step = edge_cache.get(key)
        if step is None:
            step = layout.can_step(observation, {"x": first[0], "y": first[1]}, {"x": second[0], "y": second[1]})
            edge_cache[key] = step
        return step

    found = []
    for dx, dy in _STEPS:
        neighbour = (node[0] + dx, node[1] + dy)
        if dx and dy:
            beside_x = (node[0] + dx, node[1])
            beside_y = (node[0], node[1] + dy)
            if not (
                clear(node, beside_x)
                and clear(node, beside_y)
                and clear(beside_x, neighbour)
                and clear(beside_y, neighbour)
            ):
                continue
        elif not clear(node, neighbour):
            continue
        found.append((neighbour, (_SQRT2 if dx and dy else 1.0) * _step_cost(observation, cost_cache, neighbour)))
    known = neighbour_cache[node] = tuple(found)
    return known


def _astar_path(observation, caches, start_cell, dest_cell):
    """A good-enough 8-way path of cells from start to dest, or None when none is found cheaply.

    Steps come from _neighbours (corner-safe, cost by ground). The search is weighted toward the goal
    and gives up after ``_SEARCH_BUDGET`` pops, so an unreachable goal costs a bounded amount.
    """
    start = (int(start_cell["x"]), int(start_cell["y"]))
    goal = (int(dest_cell["x"]), int(dest_cell["y"]))
    if start == goal:
        return []

    best = {start: 0.0}
    parent = {}
    closed = set()
    queue = [(0.0, 0.0, start)]
    pops = 0
    while queue:
        _, _, node = heapq.heappop(queue)
        if node in closed:
            continue
        pops += 1
        if pops > _SEARCH_BUDGET:
            return None
        if node == goal:
            path = []
            while node != start:
                path.append({"x": node[0], "y": node[1]})
                node = parent[node]
            path.reverse()
            return path
        closed.add(node)
        cost = best[node]
        for neighbour, step in _neighbours(observation, caches, node):
            if neighbour in closed:
                continue
            new_cost = cost + step
            if new_cost < best.get(neighbour, math.inf):
                best[neighbour] = new_cost
                parent[neighbour] = node
                heuristic = _octile(neighbour, goal)
                heapq.heappush(queue, (new_cost + _HEURISTIC_WEIGHT * heuristic, heuristic, neighbour))
    return None


class Agent:
    """One villager living by the day's schedule (see the module docstring), reacting to people it sees."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Initialize per-day state: home, role, and tracking structures."""

        # The village map never changes during a day, but every observation carries a fresh copy, and
        # sandbox.village rebuilds its map model whenever it sees a new copy. Keep one copy for the
        # day (shared with any other villager in this process) and hand it to every call; see act().
        self._village, self._route_caches = _day_map(observation["village"])
        observation = {**observation, "village": self._village}

        # Home & doorway (static for the day)
        self.home = me.home(observation)
        self.home_doorway = layout.doorway(observation, self.home) if self.home != "none" else None

        # Role: fixed by player number, not randomized. Determines which props (if any) this
        # villager deliberately seeks out all day; see _ROLE_TARGET_TYPES and _advance_building.
        index = int(me.player_id(observation).split("_")[1]) - 1
        self.role = _ROLE_ORDER[index % len(_ROLE_ORDER)]
        self.player_id = me.player_id(observation)
        self.number = index + 1
        self.reposition_tries = 0  # 90-degree moves around the current target so "use" picks it
        self.claims = {}  # {prop_id: (player_id, tick)} heard "I'll take care of" messages
        self.claim_told = set()  # villagers already told about the current target
        self._nearby_ids = set()  # who is in earshot this tick (set at the top of _decide)

        self.rng = me.rng(observation, seed)  # kept for hop picks, idle emotes, and building picks

        # Daily schedule: the activity _assign() picked (see _SCHEDULES), and where/when this villager sleeps
        self.activity = None  # "work", "family" or "sleep"
        self.asleep = False  # in bed; nothing wakes a sleeper until the schedule moves on
        self._hurrying = False  # heading home: greet in passing instead of stopping
        self.bell = next((prop for prop in props.all(observation) if prop["type"] == "bell"), None)
        self.wake = {"stage": "asleep", "until": None}  # dawn wake-up: asleep -> stagger | to_bell -> at_bell -> up
        self._greet_only_near = None  # at the bell muster: only greet those who've gathered there too
        self.group = _GROUPS.get(self.role)
        self.gathering = None  # {"spot", "end"} while at (or walking to) this phase's group gathering
        self.gathered = set()  # phases whose group gathering this villager has already joined
        self.social = None  # {"spot", "since", "linger", "sit"?} current spot at the inn
        self._inn = None  # inn spots, worked out once on first need (see _inn_layout)
        self.bed_spot = me.position(observation) if self.home != "none" else None  # the day starts in bed
        self.bedtime = _BEDTIMES.get(self.number, _DEFAULT_BEDTIME)
        home_index = int(self.home.split("_")[1]) if self.home.startswith("home_") else 0
        self.family_time = (
            _FAMILY_EVENING_START[0],
            _FAMILY_EVENING_START[1] + _FAMILY_EVENING_STAGGER * home_index,
        )
        self.announced_role = False  # broadcast role once at start of day
        self.used_props = set()  # occupancy/board prop ids already used today, never repeated
        self.timed_done = {}  # {prop_id: tick} when this villager finished a timed prop
        self.toggle_done = {}  # {prop_id: wanted state} toggles done (or given up) until the wanted state changes
        self.prop_memory = {}  # {prop_id: (state, tick)} as last seen
        self.arrival_looks = 0  # ticks spent turning to see the current target on arrival
        self.interactions = {}  # {player_id: {"last", "count", "initiated_by"}} greetings and replies
        self.reply_due = {}  # {player_id: tick they greeted us} still waiting for our answer
        self.homes = {entry["id"]: entry["home"] for entry in people.roster(observation)}
        self.seen_ever = set()  # everyone this villager has laid eyes on today
        self.distant_waves = {}  # {player_id: tick} of silent waves at people out of earshot
        self.family_evening = None  # {"stage": going|waiting|together, "since", "partner"?} during family time
        self.family_evening_done = False  # once a day
        self.visitor_news = None  # ("saw" | "scared", phase) to pass on in greetings that phase
        self.news_told = set()  # villagers already told the current news
        self.following = None  # {"id", "until"}: the building wanderer walking along with the visitor
        self._passing_emote = None  # wave added to this tick's action after greeting a friend on the move
        self.startled = {}  # {player_id: last_startled_tick}, separate from greeted on purpose
        self.pending_greet = None  # {"id", "ready_tick", "approach_player"?} while greeting

        # Role-task state: the prop (or, for building_wanderer, doorway) currently claimed as
        # this villager's next job, and the short waypoint it is hopping toward on the way there
        self.role_target = None  # {"prop_id", "type", "position"} or None
        self.hop_point = None
        self.active_task = None  # {"prop_id": str, "ticks_remaining": int} while mid-use
        self.use_counter = 0
        self.break_ticks_remaining = 0  # idle-hop after completing a task before picking next

        # Pathfinding cache, shared by every kind of travel
        self._path = None  # cached cardinal-step path to whichever destination is active
        self._path_destination = None  # (x, y) tuple of cached path's target
        self._route = None  # (target cell, path) of the current hop trip, reused hop to hop
        self.stuck_ticks = 0  # consecutive near-zero-movement ticks while walking a destination
        self._failed_goals = set()  # goal cells a search gave up on; never searched again today
        self._walk_failed = False  # set by _walk_toward when the current destination has no route
        self._blocked_ticks = 0  # walking ticks in a row without moving (see _make_way)
        self._backing_off = 0  # ticks left stepping back to let someone through
        self._back_off_heading = 0.0
        self._last_back_off = -_BACK_OFF_MEMORY_TICKS - 1  # tick we last stepped back for someone

        # Messages queued by act() for chat(), which the harness calls later the same tick
        self._outbox = []

    def _prop_use_ticks(self, prop_type: str) -> int:
        """How many ticks to hold "use" on a prop, based on its catalog transition type."""
        if prop_type in _TIMED_USE_TICKS:
            return _TIMED_USE_TICKS[prop_type]
        if prop_type in _TOGGLE_TYPES:
            return _TOGGLE_USE_TICKS
        if prop_type in _OCCUPANCY_TYPES:
            return self.rng.randint(*_OCCUPANCY_USE_TICKS)
        return 1  # e.g. "board": no state to hold, just a passing glance

    @property
    def is_performing(self) -> bool:
        """Mid-use on a prop right now."""
        return self.active_task is not None

    @property
    def is_chosen(self) -> bool:
        """Has picked a prop to work on but hasn't started using it yet."""
        return self.active_task is None and self.role_target is not None and self.role_target["prop_id"] is not None

    def _find(self, observation: ThreeBranchesObservation, start_cell, dest_cell):
        """Path between two cells using this day's caches, or None when the search gives up."""
        return _astar_path(observation, self._route_caches, start_cell, dest_cell)

    def _walk_toward(self, destination: dict[str, float], observation: ThreeBranchesObservation) -> float:
        """Heading toward destination, following a cached path around obstacles."""
        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)
        dest_cell = layout.cell_at(observation, destination)
        if here_cell is None or dest_cell is None:
            return geometry.heading_to(here, destination)
        dest_cell = _nearest_walkable_cell(observation, dest_cell)

        cache_key = (dest_cell["x"], dest_cell["y"])
        if self._path_destination != cache_key or self._path is None:
            path = None if cache_key in self._failed_goals else self._find(observation, here_cell, dest_cell)
            self._walk_failed = path is None
            if path is None:
                self._failed_goals.add(cache_key)
            self._path = path or []
            self._path_destination = cache_key

        # A diagonal step can carry us past the next cell, so drop everything up to the cell we stand in
        for index, cell in enumerate(self._path[:4]):
            if cell["x"] == here_cell["x"] and cell["y"] == here_cell["y"]:
                del self._path[: index + 1]
                break

        if not self._path:
            return self._make_way(observation, here, geometry.heading_to(here, destination))
        return self._make_way(observation, here, geometry.heading_to(here, _cell_centre(self._path[0])))

    def _make_way(self, observation: ThreeBranchesObservation, here, walk_heading: float) -> float:
        """Get unstuck from someone in the way (paths don't know about people). After a few blocked
        ticks with someone right beside us: two villagers squeezing through a doorway push against
        each other forever, so the higher player number steps back once to let the lower one through
        (the same "lower number goes first" rule as greetings). If that didn't clear it -- they're
        just standing there, minding a stall say -- or we're the lower number, step round them."""
        if self._backing_off > 0:
            self._backing_off -= 1
            return self._back_off_heading
        self._blocked_ticks = self._blocked_ticks + 1 if me.moved(observation) < 0.05 else 0
        if self._blocked_ticks < _BLOCKED_TICKS:
            return walk_heading
        blocker = next(
            (person for person in people.nearby(observation) if geometry.distance(here, person["position"]) <= _BLOCKER_RANGE),
            None,
        )
        if blocker is None:
            return walk_heading
        tick = day.tick(observation)
        self._blocked_ticks = 0
        self._backing_off = _BACK_OFF_TICKS - 1
        toward_them = geometry.heading_to(here, blocker["position"])
        if _player_number(blocker["id"]) < self.number and tick - self._last_back_off > _BACK_OFF_MEMORY_TICKS:
            self._last_back_off = tick
            self._back_off_heading = (toward_them + 180.0) % 360.0
        else:
            # Sidestep: whichever side of our way forward points further from them
            left, right = (walk_heading + 90.0) % 360.0, (walk_heading - 90.0) % 360.0
            gap = lambda side: abs((side - toward_them + 180.0) % 360.0 - 180.0)  # noqa: E731
            self._back_off_heading = left if gap(left) >= gap(right) else right
        return self._back_off_heading

    def _random_nearby_point(self, observation: ThreeBranchesObservation, here) -> dict[str, float]:
        """A random walkable point a short hop away, for idling when a role has no work left."""
        for _ in range(20):
            angle = math.radians(self.rng.uniform(0.0, 360.0))
            reach = self.rng.uniform(_HOP_REACH_MIN, _HOP_REACH_MAX)
            point = {"x": here["x"] + reach * math.cos(angle), "y": here["y"] + reach * math.sin(angle)}
            cell = layout.cell_at(observation, point)
            if cell is None or not layout.walkable(observation, cell):
                continue
            if (cell["x"], cell["y"]) in self._failed_goals:
                continue  # a search already gave up on this spot, so pick somewhere else
            return _cell_centre(cell)
        return here

    def _pick_hop_point(self, observation: ThreeBranchesObservation, here, target_pos) -> dict[str, float]:
        """A short waypoint along the real path toward target_pos, so travel comes in small hops."""
        here_cell = layout.cell_at(observation, here)
        dest_cell = layout.cell_at(observation, target_pos)
        if here_cell is None or dest_cell is None:
            return target_pos
        dest_cell = _nearest_walkable_cell(observation, dest_cell)
        # Same target as the last hop and we're still on that route: the rest of it is the path, no
        # new search (a long trip used to search the whole way again at every hop)
        key = (dest_cell["x"], dest_cell["y"])
        path = None
        if self._route is not None and self._route[0] == key:
            here_key = (here_cell["x"], here_cell["y"])
            for index, cell in enumerate(self._route[1]):
                if (cell["x"], cell["y"]) == here_key:
                    path = self._route[1][index + 1 :]
                    break
        if path is None:
            path = self._find(observation, here_cell, dest_cell) or []
        self._route = (key, path)
        if not path:
            return target_pos
        hop_len = self.rng.uniform(_HOP_REACH_MIN, _HOP_REACH_MAX)
        travelled = 0.0
        previous = here
        hop_index = len(path) - 1
        for index, cell in enumerate(path):
            centre = _cell_centre(cell)
            travelled += geometry.distance(previous, centre)
            previous = centre
            if travelled >= hop_len:
                hop_index = index
                break
        # Keep the route up to the hop point, so walking there needs no second search
        hop_cell = path[hop_index]
        self._path = path[: hop_index + 1]
        self._path_destination = (hop_cell["x"], hop_cell["y"])
        self._walk_failed = False
        return _cell_centre(hop_cell)

    def _prop_needs_doing(self, prop, tick: int, phase: str) -> bool:
        """Best guess whether a prop still needs doing, from what this villager has done and seen."""
        prop_id, prop_type = prop["id"], prop["type"]
        if prop_id in self.used_props:
            return False
        wanted = _PHASE_WANTED.get(prop_type)
        if wanted is not None and self.toggle_done.get(prop_id) == wanted[phase]:
            return False  # we already set it right (or gave up on it) for this time of day
        revert = _REVERT_TICKS.get(prop_type)
        if revert is not None and tick - self.timed_done.get(prop_id, -revert) < revert:
            return False  # we tended it and it hasn't reverted yet
        work = _work_state(prop_type, phase)
        remembered = self.prop_memory.get(prop_id)
        if work is None or remembered is None:
            return True  # stateless (board) or never seen: worth a visit
        state, seen_tick = remembered
        if state == work:
            return True
        if revert is not None:
            return tick - seen_tick >= revert  # seen tended; reverts at the latest this long after
        if prop_type in _OCCUPANCY_TYPES:
            return tick - seen_tick >= _OCCUPANCY_MEMORY_TICKS
        return False  # a toggle last seen in the state this phase wants stays done

    def _mark_done(self, prop_id: str, prop_type: str, tick: int, phase: str) -> None:
        """Take a prop off this villager's list: timed props until they revert, toggles until the
        time of day wants them the other way, everything else for the rest of the day."""
        if prop_type in _REVERT_TICKS:
            self.timed_done[prop_id] = tick
        elif prop_type in _PHASE_WANTED:
            self.toggle_done[prop_id] = _PHASE_WANTED[prop_type][phase]
        else:
            self.used_props.add(prop_id)

    def _available_props(self, observation: ThreeBranchesObservation, prop_types):
        """Every prop of these types, anywhere in the village, that seems to need doing and that
        nobody visible is already standing at."""
        tick = day.tick(observation)
        phase = _phase(observation)
        here = me.position(observation)
        watchers = [person["position"] for person in people.seen(observation)]
        watchers += [person["position"] for person in people.nearby(observation)]
        candidates = []
        for prop in props.all(observation):
            if prop["type"] not in prop_types:
                continue
            if not self._prop_needs_doing(prop, tick, phase):
                continue
            if self._someone_else_has(observation, prop, here, tick):
                continue
            prop_pos = _cell_centre(prop["cell"])
            if any(geometry.distance(prop_pos, watcher) <= _OCCUPIED_CHECK_RANGE for watcher in watchers):
                continue  # someone is already standing right on this one -- let them have it
            candidates.append(prop)
        return candidates

    def _pick_available_prop(self, observation: ThreeBranchesObservation, prop_types):
        """Next prop of these types that needs doing and nobody visible is already working, or None.
        A toggle round (lighting up, opening or closing stalls) takes the nearest next, so it sweeps
        the village instead of zig-zagging; other work is picked at random for variety."""
        candidates = self._available_props(observation, prop_types)
        if not candidates:
            return None
        if all(prop_type in _TOGGLE_TYPES for prop_type in prop_types):
            here = me.position(observation)
            return min(candidates, key=lambda prop: geometry.distance(here, _prop_centre(prop)))
        return self.rng.choice(candidates)

    def _opportunistic_toggle(self, observation: ThreeBranchesObservation):
        """A lantern/hearth right here in the wrong state for the time of day, to fix on the way:
        anyone puts lights out at dawn and in the morning and lights them in the evening and at
        night, but at midday only the lighter lights them."""
        usable = props.usable(observation)
        if usable is None or usable["type"] not in _LIGHT_TYPES:
            return None
        phase = _phase(observation)
        if phase in _LIGHTER_ONLY_PHASES and self.role != "lighter":
            return None
        if self.toggle_done.get(usable["id"]) == _PHASE_WANTED[usable["type"]][phase]:
            return None
        if not _prop_needs_work(observation, usable):
            return None
        return usable

    def _advance_hop(self, observation: ThreeBranchesObservation, heading: float, here, target_pos, sweep_at_target=False):
        """Walk toward target_pos in short hops, sweeping at each waypoint before the next leg."""
        if self.hop_point is None:
            self._path = None
            self.hop_point = self._pick_hop_point(observation, here, target_pos)  # leaves its route in self._path
            self.stuck_ticks = 0

        # On the final leg the caller decides arrival (prop reach is tighter than 2.0), so don't
        # sweep-stop short of the target -- that looped forever just outside use range.
        final_leg = geometry.distance(self.hop_point, target_pos) < 1.0
        if (sweep_at_target or not final_leg) and geometry.distance(here, self.hop_point) < 2.0:
            self.hop_point = None
            return action.stand(heading, "sweep")

        if me.moved(observation) < 0.05:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        if self.stuck_ticks >= _STUCK_TICKS_LIMIT:
            self.hop_point = None
            self.stuck_ticks = 0
            self._path = None
            return action.stand(heading, "shrug")

        walk_heading = self._walk_toward(self.hop_point, observation)
        if self._walk_failed:
            if self.role_target is not None and self.role_target["prop_id"] is not None:
                self._mark_done(
                    self.role_target["prop_id"], self.role_target["type"], day.tick(observation), _phase(observation)
                )
            self.role_target = None
            self.hop_point = None
            self._path = None
            self.stuck_ticks = 0
            self._walk_failed = False
            return action.stand(heading, "shrug")
        return action.walk(walk_heading, 1.0, "none")

    def _idle_hop(self, observation: ThreeBranchesObservation, heading: float, here):
        """Keep moving in short hops when a role has run out of things to do for the day."""
        if self.hop_point is None:
            self.hop_point = self._random_nearby_point(observation, here)
            self._path = None
            self.stuck_ticks = 0
            return action.stand(heading, self.rng.choice(_IDLE_EMOTES))
        return self._advance_hop(observation, heading, here, self.hop_point, sweep_at_target=True)

    def _rounds_due(self, observation: ThreeBranchesObservation) -> bool:
        """Whether this role walks rounds between its own props when it has no work right now."""
        rounds = _IDLE_ROUNDS.get(self.role)
        return rounds is not None and _phase(observation) in rounds[1]

    def _advance_rounds(self, observation: ThreeBranchesObservation, heading: float, here):
        """With no work left, walk to one of this role's own props and linger there, then another:
        stall tenders mind the open stalls (pointing at passers-by), the lighter walks the lantern
        rounds. The target has no prop_id, so any real work that turns up replaces it at once."""
        prop_type, _, linger, emote = _IDLE_ROUNDS[self.role]
        target = self.role_target
        if target is None:
            choices = [prop for prop in props.all(observation) if prop["type"] == prop_type]
            if not choices:
                return self._idle_hop(observation, heading, here)
            prop = self.rng.choice(choices)
            target = self.role_target = {
                "prop_id": None,
                "type": None,
                "round_prop": prop,
                "position": _cell_centre(_nearest_walkable_cell(observation, prop["cell"])),
                "linger": self.rng.randint(*linger),
                "since": None,
            }
            self.hop_point = None
        if geometry.distance(here, target["position"]) > 2.0:
            return self._advance_hop(observation, heading, here, target["position"])
        tick = day.tick(observation)
        if target["since"] is None:
            target["since"] = tick
        if tick - target["since"] >= target["linger"]:
            self.role_target = None
            self.hop_point = None
            return action.stand(heading, "none")
        facing = geometry.heading_to(here, _prop_centre(target["round_prop"]))
        passer_by = emote != "none" and any(
            geometry.distance(here, person["position"]) <= _VISITOR_APPROACH_RANGE for person in people.seen(observation)
        )
        expression = emote if passer_by and (tick - target["since"]) % 15 == 5 else "none"
        return action.stand(facing, expression)

    def _advance_building(self, observation: ThreeBranchesObservation, heading: float, here):
        """The building-wanderer role: head to a random building's doorway, sweep, pick another."""
        if self.role_target is None:
            candidates = list(layout.buildings(observation))
            self.rng.shuffle(candidates)
            doorway = None
            for building in candidates:
                doorway = layout.doorway(observation, building["id"])
                if doorway is not None:
                    break
            if doorway is None:
                return self._idle_hop(observation, heading, here)
            self.role_target = {"prop_id": None, "type": None, "position": doorway}
            self.hop_point = None

        target_pos = self.role_target["position"]
        if geometry.distance(here, target_pos) <= 2.0:
            self.role_target = None
            self.hop_point = None
            return action.stand(heading, "sweep")

        return self._advance_hop(observation, heading, here, target_pos)

    def _arrival_check(self, observation: ThreeBranchesObservation, heading: float, here):
        """Confirm the role target still needs doing before using it; None means go ahead."""
        target = self.role_target
        work = _work_state(target["type"], _phase(observation))
        state = props.state(observation, target["prop_id"])
        if work is None or state == work:
            self.arrival_looks = 0
            return None
        if state is None and self.arrival_looks < _ARRIVAL_LOOK_TICKS:
            self.arrival_looks += 1
            prop = props.find(observation, target["prop_id"])
            return action.stand(geometry.heading_to(here, _prop_centre(prop)), "none")
        if state is None:
            return self._give_up_target(observation, heading)  # never came into view
        self._clear_target()  # already done by someone else; memory now knows
        return action.stand(heading, "shrug")
 
    def _clear_target(self) -> None:
        target = self.role_target
        if target is not None and target.get("prop_id") is not None:
            for player_id in self.claim_told & self._nearby_ids:  # only those in earshot would hear it
                self._outbox.append({"to": player_id, "text": _RELEASE_TEXT.format(target["prop_id"])})
        self.role_target = None
        self.hop_point = None
        self.arrival_looks = 0
        self.reposition_tries = 0
        self.claim_told = set()
 
    def _give_up_target(self, observation: ThreeBranchesObservation, heading: float):
        """Drop the role target for now: timed props come back once they'd revert, toggles once the
        time of day changes what they should be, others not today."""
        target = self.role_target
        self._mark_done(target["prop_id"], target["type"], day.tick(observation), _phase(observation))
        self._clear_target()
        return action.stand(heading, "shrug")

    def _arrived(self, here) -> bool:
        reach = _ARRIVAL_SLACK if self.role_target.get("tight") else geometry.PROP_REACH
        return geometry.distance(here, self.role_target["position"]) <= reach

    def _reposition(self, observation: ThreeBranchesObservation, heading: float, here):
        """"use" would pick a different prop: step 90 degrees around the target until it's ours."""
        prop = props.find(observation, self.role_target["prop_id"])
        centre = _prop_centre(prop)
        width, height = _PROP_SIZES.get(prop["type"], (1, 1))
        radius = max(width, height) / 2 + 0.7
        bearing = self.role_target.setdefault("bearing", geometry.heading_to(centre, here))
        while self.reposition_tries < len(_REPOSITION_TURNS):
            angle = math.radians(bearing + _REPOSITION_TURNS[self.reposition_tries])
            self.reposition_tries += 1
            spot = {"x": centre["x"] + radius * math.cos(angle), "y": centre["y"] + radius * math.sin(angle)}
            cell = layout.cell_at(observation, spot)
            if cell is not None and layout.walkable(observation, cell):
                self.role_target["position"] = spot
                self.role_target["tight"] = True
                self.hop_point = None
                return action.stand(geometry.heading_to(here, centre), "none")
        return self._give_up_target(observation, heading)

    def _interested(self, player_id: str, prop_type: str) -> bool:
        """Whether this villager's role works this kind of prop."""
        return people.is_npc(player_id) and prop_type in _ROLE_TARGET_TYPES.get(_role_of(player_id), ())

    @staticmethod
    def _claim_abandoned(claimer, prop_id: str, centre) -> bool:
        """A claimer we can see using a different prop, or walking away from the one they claimed."""
        if claimer["expression"]["type"] == "use":
            return claimer["expression"]["target"] != prop_id
        if claimer["moved"] < 0.05:
            return False  # standing still (greeting someone, say): could still be on their way
        toward = geometry.heading_to(claimer["position"], centre)
        return abs((float(claimer["heading"]) - toward + 180.0) % 360.0 - 180.0) > _CLAIM_AWAY_DEGREES

    def _someone_else_has(self, observation: ThreeBranchesObservation, prop, here, tick: int) -> bool:
        """Is another villager using, claiming, or clearly walking to this prop ahead of us?

        Ties go to whoever is closer to the prop, then the lower player number, so two villagers
        that can see each other reach the same answer without talking.
        """
        prop_id = prop["id"]
        centre = _prop_centre(prop)
        mine = (geometry.distance(here, centre), self.number)
        seen = people.seen(observation)
        for person in seen:
            if person["expression"]["type"] == "use" and person["expression"]["target"] == prop_id:
                return True
        claim = self.claims.get(prop_id)
        sighted = next((person for person in seen if claim is not None and person["id"] == claim[0]), None)
        if sighted is not None and self._claim_abandoned(sighted, prop_id, centre):
            del self.claims[prop_id]  # they've moved on to something else: the claim lapses early
            claim = None
        if claim is not None and claim[0] != self.player_id and tick - claim[1] <= _CLAIM_TTL_TICKS:
            claimer = people.find(observation, claim[0])
            if claimer is None:
                return True  # out of earshot now; assume they're on it
            if (geometry.distance(claimer["position"], centre), _player_number(claim[0])) < mine:
                return True
        for person in seen:
            if not self._interested(person["id"], prop["type"]) or person["moved"] < 0.05:
                continue
            gap = abs((float(person["heading"]) - geometry.heading_to(person["position"], centre) + 180.0) % 360.0 - 180.0)
            if gap > _HEADING_MATCH_DEGREES:
                continue
            if (geometry.distance(person["position"], centre), _player_number(person["id"])) < mine:
                return True
        return False

    def _share_claim(self, observation: ThreeBranchesObservation) -> None:
        """Tell each villager in earshot who could want our target, once, that we're on it."""
        target = self.role_target
        for person in people.nearby(observation):
            player_id = person["id"]
            if player_id in self.claim_told or not self._interested(player_id, target["type"]):
                continue
            self.claim_told.add(player_id)
            self._outbox.append({"to": player_id, "text": _CLAIM_TEXT.format(target["prop_id"])})

    def _mid_toggle(self) -> bool:
        """Whether a toggle hold is under way; interrupting it would flip the prop back."""
        return self.active_task is not None and self.active_task["type"] in _TOGGLE_TYPES

    def _cooldown_key(self, player_id: str, tick: int) -> str:
        """The stretch of the day one greeting per pair covers: the phase, except that for family
        dawn and morning count as one (see _FAMILY_GREET_PHASES)."""
        phase = _phase_at_tick(tick)
        if self.relation(player_id) == "family":
            return _FAMILY_GREET_PHASES.get(phase, phase)
        return phase

    def _cooled(self, player_id: str, tick: int) -> bool:
        """Whether we're free to greet this person again: never yet, or a whole phase has passed since
        our last exchange (as long as the phase it happened in) and it's a new phase -- so never twice
        a phase, and never twice in quick succession across a phase boundary either."""
        entry = self.interactions.get(player_id)
        if entry is None:
            return True
        return tick - entry["since"] >= _phase_length(entry["since"]) and entry["key"] != self._cooldown_key(player_id, tick)

    def _record_interaction(self, player_id: str, tick: int, initiated_by: str) -> None:
        """Log a greeting or reply; a greeting and its reply count as one exchange, which starts the wait."""
        entry = self.interactions.get(player_id)
        if entry is None:
            entry = self.interactions[player_id] = {"since": tick, "key": None, "count": 0, "initiated_by": initiated_by}
        if self._cooled(player_id, tick) or entry["count"] == 0:
            entry["count"] += 1
            entry["initiated_by"] = initiated_by
            entry["since"] = tick
            entry["key"] = self._cooldown_key(player_id, tick)
        if people.is_visitor(player_id):
            # A greeting that opened with a startle: we never saw who it was until we'd jumped
            startled = self.pending_greet is not None and self.pending_greet["id"] == player_id and self.pending_greet.get("startled")
            if not (startled and self._news_is("scared", tick)):
                self._set_news("saw", tick)

    def _set_news(self, kind: str, tick: int) -> None:
        """Have news to pass on in greetings for the rest of this phase: "saw" or "scared"."""
        if self.visitor_news != (kind, _phase_at_tick(tick)):
            self.visitor_news = (kind, _phase_at_tick(tick))
            self.news_told = set()

    def _news_is(self, kind: str, tick: int) -> bool:
        return self.visitor_news == (kind, _phase_at_tick(tick))

    def _greeting_text(self, kind: str, player_id: str, tick: int) -> str:
        """The greeting line for the time of day, plus any visitor news this phase for anyone not
        yet told, plus a role's own word for the visitor."""
        text = _GREETING_LINES[kind].format(**_PHASE_WORDS[_phase_at_tick(tick)])
        news = self.visitor_news
        if kind != "visitor" and news is not None and news[1] == _phase_at_tick(tick) and player_id not in self.news_told:
            self.news_told.add(player_id)
            text += _VISITOR_NEWS[news[0]]
        if kind == "visitor":
            text += _VISITOR_EXTRA_LINES.get(self.role, "")
        return text

    def _reply_text(self, kind: str, tick: int) -> str:
        return _REPLY_LINES[kind].format(**_PHASE_WORDS[_phase_at_tick(tick)])

    def _greeting_me(self, person, here) -> bool:
        """A wave or nod from someone close by who is facing us."""
        if person["expression"]["type"] not in ("wave", "nod"):
            return False
        if geometry.distance(person["position"], here) > _GREETING_NOTICE_RANGE:
            return False
        toward_me = geometry.heading_to(person["position"], here)
        gap = abs((float(person["heading"]) - toward_me + 180.0) % 360.0 - 180.0)
        return gap <= _GREETING_FACING_DEGREES

    def relation(self, player_id: str) -> str:
        """visitor, family (same home), friend (same role, once seen), acquaintance (has greeted or
        been greeted), else stranger."""
        if people.is_visitor(player_id):
            return "visitor"
        if self.home != "none" and self.homes.get(player_id) == self.home:
            return "family"
        if player_id in self.seen_ever and people.is_npc(player_id) and _role_of(player_id) == self.role:
            return "friend"
        entry = self.interactions.get(player_id)
        if entry is not None and entry["count"] > 0:
            return "acquaintance"
        return "stranger"

    def _may_greet(self, player_id: str, tick: int) -> bool:
        if self.relation(player_id) == "family" and _phase_at_tick(tick) not in _FAMILY_GREET_PHASES:
            return False  # family don't greet each other at midday
        return self._cooled(player_id, tick)

    def _greeted_by(self, player_id: str, tick: int) -> None:
        """They greeted us first (callers check it's a new exchange): log it now, so the wait before
        greeting them again starts whether or not our answer gets through, then turn our own
        pending greeting into the answer, or owe them one."""
        if self.relation(player_id) == "family" and _phase_at_tick(tick) not in _FAMILY_GREET_PHASES:
            return  # family have no exchanges at midday: a nod read as a greeting starts nothing
        self._record_interaction(player_id, tick, "them")
        if self.pending_greet is not None and self.pending_greet["id"] == player_id:
            self.pending_greet["reply"] = True
            self.pending_greet["ready_tick"] = min(self.pending_greet["ready_tick"], tick + _REPLY_DELAY_TICKS)
        else:
            self.reply_due.setdefault(player_id, tick)

    def _goes_second(self, person, here) -> bool:
        """When two villagers face each other, the lower player number speaks first."""
        if not people.is_npc(person["id"]) or "heading" not in person:
            return False
        toward_me = geometry.heading_to(person["position"], here)
        gap = abs((float(person["heading"]) - toward_me + 180.0) % 360.0 - 180.0)
        return gap <= geometry.VISION_DEGREES / 2 and _player_number(person["id"]) < self.number

    def _say(self, player_id: str, text: str, nearby_ids, reply: bool = False) -> None:
        """Speak to someone directly, only if they're in earshot (otherwise it would be dropped).
        A reply is marked so chat() can hold it for the next tick rather than lose it."""
        if player_id in nearby_ids:
            self._outbox.append({"to": player_id, "text": text, "reply": reply})

    def _react_to_people(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction | None:
        """Stop and turn toward whoever just showed up, or None when nobody is due a reaction.

        Reactions outrank every task, so this runs before the phase logic. Pausing mid-use costs
        nothing: the use counter simply doesn't advance this tick and resumes on the next one.
        """
        here = me.position(observation)
        tick = day.tick(observation)
        seen = people.seen(observation)
        nearby = people.nearby(observation)
        if self._greet_only_near is not None:
            # At the bell muster, greet only those who've gathered there too, not anyone still on the way
            spot = self._greet_only_near
            seen = tuple(person for person in seen if geometry.distance(person["position"], spot) <= _MUSTER_REACH + 1.0)
            nearby = tuple(person for person in nearby if geometry.distance(person["position"], spot) <= _MUSTER_REACH + 1.0)
        seen_ids = {person["id"] for person in seen}
        # Chat is only delivered to someone in hearing range, so speaking needs this, not sight
        nearby_ids = {person["id"] for person in nearby}
        self.seen_ever.update(seen_ids)

        def find(player_id: str):
            """That person's current record, from sight or hearing, or None once they are gone."""
            for person in seen:
                if person["id"] == player_id:
                    return person
            for person in nearby:
                if person["id"] == player_id:
                    return person
            return None

        # Someone waving or nodding at us: inside the cooldown it's their reply (it still counts and
        # restarts the cooldown); outside it, we owe them an answer
        for person in seen:
            if not self._greeting_me(person, here):
                continue
            if self._cooled(person["id"], tick):
                self._greeted_by(person["id"], tick)
            else:
                self._record_interaction(person["id"], tick, "them")

        # Already noticed someone: walk to them if they started close by, then greet
        if self.pending_greet is not None:
            person = find(self.pending_greet["id"])
            if person is None:
                self.pending_greet = None  # they left before we could say anything
            else:
                facing = geometry.heading_to(here, person["position"])
                dist_to_person = geometry.distance(here, person["position"])

                # If the visitor started within approach range, walk over before greeting
                if self.pending_greet.get("approach_player") and dist_to_person > _VISITOR_GREET_DISTANCE:
                    return action.walk(self._walk_toward(person["position"], observation), 1.0, "none")

                if tick < self.pending_greet["ready_tick"]:
                    return action.stand(facing, "none")

                player_id = person["id"]
                reply = self.pending_greet.get("reply")
                kind = self.relation(player_id)  # read before recording, so a first meeting is still "stranger"
                if player_id not in nearby_ids:
                    # Too far to be heard: a silent wave or nod, and no exchange is logged, so a real
                    # one still happens when they come into earshot
                    self.pending_greet = None
                    self.distant_waves[player_id] = tick
                    return action.stand(facing, "wave" if kind in _WAVE_RELATIONS else "nod")
                self._record_interaction(player_id, tick, "them" if reply else "me")
                self.pending_greet = None
                text = self._reply_text(kind, tick) if reply else self._greeting_text(kind, player_id, tick)
                self._say(player_id, text, nearby_ids, reply=bool(reply))
                reaction = _VISITOR_REACTIONS.get(self.role) if kind == "visitor" else None
                if reaction == "point":
                    stalls = [prop for prop in props.all(observation) if prop["type"] == "stall"]
                    if stalls:
                        stall = min(stalls, key=lambda prop: geometry.distance(here, _prop_centre(prop)))
                        return action.stand(geometry.heading_to(here, _prop_centre(stall)), "point")
                if reaction == "follow":
                    self.following = {"id": player_id, "until": tick + _FOLLOW_TICKS}
                return action.stand(facing, "wave" if kind in _WAVE_RELATIONS else "nod")

        def greet_due(player_id: str) -> bool:
            if tick - self.distant_waves.get(player_id, -_DISTANT_WAVE_COOLDOWN_TICKS) < _DISTANT_WAVE_COOLDOWN_TICKS:
                return False
            return self._may_greet(player_id, tick)

        def in_passing(player_id: str) -> bool:
            """Greet without stopping: friends always, family after midday, a farmer meeting the
            visitor, and anyone hurrying somewhere."""
            kind = self.relation(player_id)
            return (
                self._hurrying
                or kind == "friend"
                or (kind == "family" and _phase_at_tick(tick) in _FAMILY_PASSING_PHASES)
                or (kind == "visitor" and _VISITOR_REACTIONS.get(self.role) == "nod")
            )

        def greet_in_passing(player_id: str, reply: bool) -> None:
            """Say the line and wave or nod while carrying on with whatever we're doing."""
            kind = self.relation(player_id)
            self._record_interaction(player_id, tick, "them" if reply else "me")
            text = self._reply_text(kind, tick) if reply else self._greeting_text(kind, player_id, tick)
            self._say(player_id, text, nearby_ids, reply=reply)
            nodding = kind not in _WAVE_RELATIONS or (kind == "visitor" and _VISITOR_REACTIONS.get(self.role) == "nod")
            self._passing_emote = "nod" if nodding else "wave"

        def startle_due(player_id: str) -> bool:
            return tick - self.startled.get(player_id, -_STARTLE_COOLDOWN_TICKS) >= _STARTLE_COOLDOWN_TICKS

        def notice(person, expression: str, reply: bool = False):
            """Turn toward someone and start the beat that ends in a wave or a nod."""
            approach = (
                not reply
                and people.is_visitor(person["id"])
                and geometry.distance(here, person["position"]) <= _VISITOR_APPROACH_RANGE
            )
            delay = _REPLY_DELAY_TICKS if reply else _GREET_DELAY_TICKS
            if not reply and self._goes_second(person, here):
                delay += _GREET_DELAY_TICKS  # let them speak first; we greet anyway if they don't
            self.pending_greet = {
                "id": person["id"],
                "ready_tick": tick + delay,
                "approach_player": approach,
                "reply": reply,
            }
            return action.stand(geometry.heading_to(here, person["position"]), expression)

        # Answer anyone who greeted us first, while they're still around. The answer belongs to the
        # exchange they started, so the wait between exchanges doesn't hold it back.
        for player_id, since in list(self.reply_due.items()):
            del self.reply_due[player_id]
            person = find(player_id)
            if person is None or tick - since > _REPLY_WINDOW_TICKS:
                continue
            if self.relation(player_id) == "family" and _phase_at_tick(tick) not in _FAMILY_GREET_PHASES:
                continue  # family don't greet at midday, so a nod read as one gets no answer either
            if in_passing(player_id):
                if player_id in nearby_ids:
                    greet_in_passing(player_id, reply=True)
                    return None
                continue
            return notice(person, "none", reply=True)

        # The visitor comes first (unless it's a greeting in passing, below: hurrying, or a farmer)
        visitor = next((person for person in seen if people.is_visitor(person["id"])), None)
        if visitor is not None and not in_passing(visitor["id"]) and greet_due(visitor["id"]):
            return notice(visitor, "none")

        # In-passing greetings for those in earshot; anyone farther off waits until they're closer
        for person in nearby:
            if not in_passing(person["id"]) or not greet_due(person["id"]):
                continue
            sighted = find(person["id"]) if person["id"] in seen_ids else person
            if sighted is not person and self._goes_second(sighted, here):
                continue  # they'll greet us first and we answer in passing
            greet_in_passing(person["id"], reply=False)
            return None

        for person in seen:
            player_id = person["id"]
            if people.is_visitor(player_id) or in_passing(player_id):
                continue
            if greet_due(player_id):
                return notice(person, "none")

        # Heard but not seen means they are in the blind spot behind us, so the beat opens with a
        # startle instead. It doesn't use up the greet cooldown, so the wave or nod still follows.
        for person in nearby:
            if person["id"] in seen_ids:
                continue
            if in_passing(person["id"]):
                continue
            if startle_due(person["id"]) and greet_due(person["id"]):
                self.startled[person["id"]] = tick
                if people.is_visitor(person["id"]):
                    self._set_news("scared", tick)  # in earshot but out of sight: no idea who that was
                startle = notice(person, "startle")
                self.pending_greet["startled"] = True
                return startle

        return None

    def _family(self) -> list[str]:
        return [pid for pid, home in self.homes.items() if home == self.home and pid != self.player_id]

    def _family_evening_due(self, observation: ThreeBranchesObservation) -> bool:
        """This household's evening slot has come, it hasn't happened yet, and it's not too late."""
        return (
            not self.family_evening_done
            and not self.asleep
            and self.home_doorway is not None
            and bool(self._family())
            and _reached(observation, self.family_time)
            and not _reached(observation, _FAMILY_GIVE_UP)
        )

    def _family_evening_tick(self, observation: ThreeBranchesObservation, heading: float, here, tick: int):
        """Walk to the home doorway, wait for family, spend a little time together, then move on
        (usually to bed, which is just inside)."""
        meet = self.family_evening

        def finish():
            self.family_evening = None
            self.family_evening_done = True
            self._path = None
            return action.stand(heading, "wave")

        if _reached(observation, _FAMILY_GIVE_UP) and meet["stage"] != "together":
            return finish()

        if meet["stage"] == "going":
            # Family already standing at the door (and blocking it) counts as having arrived, but
            # only near home -- bumping into each other in town isn't the family evening
            near_family = geometry.distance(here, self.home_doorway) <= _FAMILY_MEET_DISTANCE + 1.0 and any(
                person is not None and geometry.distance(here, person["position"]) <= _FAMILY_MEET_DISTANCE
                for person in (people.find(observation, player_id) for player_id in self._family())
            )
            if not near_family and geometry.distance(here, self.home_doorway) > 1.5:
                walk_heading = self._walk_toward(self.home_doorway, observation)
                if self._walk_failed:
                    return finish()
                return action.walk(walk_heading, 1.0, "none")
            meet["stage"], meet["since"] = "waiting", tick

        if meet["stage"] == "waiting":
            for player_id in self._family():
                person = people.find(observation, player_id)
                if person is not None and geometry.distance(here, person["position"]) <= _FAMILY_MEET_DISTANCE:
                    meet.update(stage="together", since=tick, partner=player_id)
                    return action.stand(geometry.heading_to(here, person["position"]), "wave")
            return action.stand(heading, "none")  # wait until they come, or until _FAMILY_GIVE_UP

        # Together: face each other, with the odd laugh or nod
        if tick - meet["since"] >= _FAMILY_TIME_TICKS:
            return finish()
        partner = people.find(observation, meet["partner"])
        facing = geometry.heading_to(here, partner["position"]) if partner is not None else heading
        expression = self.rng.choice(("laugh", "nod")) if (tick - meet["since"]) % 12 == 6 else "none"
        return action.stand(facing, expression)

    def _gather_tick(self, observation: ThreeBranchesObservation, heading: float, here, tick: int):
        """Walk to the group's gathering spot and spend the rest of the window there with whoever of
        the group turns up: facing them, with the odd laugh or nod. Then back to work."""
        meet = self.gathering
        if tick >= meet["end"]:
            self.gathering = None
            return action.stand(heading, "wave")
        if geometry.distance(here, meet["spot"]) > _GATHER_REACH:
            walk_heading = self._walk_toward(meet["spot"], observation)
            if self._walk_failed:
                self._walk_failed = False
                self.gathering = None  # can't get there: back to work
                return action.stand(heading, "shrug")
            return action.walk(walk_heading, 1.0, "none")
        mates = [
            person
            for person in people.seen(observation)
            if people.is_npc(person["id"])
            and _GROUPS.get(_role_of(person["id"])) == self.group
            and geometry.distance(here, person["position"]) <= _GATHER_REACH + 2.0
        ]
        if not mates:
            return action.stand(heading, "none")  # first here: wait for the others
        nearest = min(mates, key=lambda person: geometry.distance(here, person["position"]))
        expression = self.rng.choice(("laugh", "nod")) if tick % 12 == self.number % 12 else "none"
        return action.stand(geometry.heading_to(here, nearest["position"]), expression)

    def _inn_layout(self, observation: ThreeBranchesObservation):
        """The inn's floor cells, the spots just outside its doorway, and its bench, worked out once."""
        if self._inn is not None:
            return self._inn or None
        inn = next((item for item in layout.buildings(observation) if item["type"] == "inn"), None)
        door = layout.doorway(observation, inn["id"]) if inn is not None else None
        if inn is None or door is None:
            self._inn = {}
            return None
        centre = layout.building_center(observation, inn["id"])
        centre_cell = layout.cell_at(observation, centre)
        inside, outside = [], []
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                cell = {"x": centre_cell["x"] + dx, "y": centre_cell["y"] + dy}
                if not layout.walkable(observation, cell):
                    continue
                point = _cell_centre(cell)
                if layout.building_at(observation, point) == inn["id"]:
                    inside.append(point)
                elif geometry.distance(point, door) <= _INN_OUTSIDE_RANGE and layout.ground_at(observation, cell) != "doorway":
                    outside.append(point)
        for dx in range(-int(_INN_OUTSIDE_RANGE), int(_INN_OUTSIDE_RANGE) + 1):
            for dy in range(-int(_INN_OUTSIDE_RANGE), int(_INN_OUTSIDE_RANGE) + 1):
                cell = layout.cell_at(observation, {"x": door["x"] + dx, "y": door["y"] + dy})
                if cell is None or not layout.walkable(observation, cell):
                    continue
                point = _cell_centre(cell)
                if (
                    point not in outside
                    and layout.building_at(observation, point) is None
                    and layout.ground_at(observation, cell) != "doorway"
                    and geometry.distance(point, door) <= _INN_OUTSIDE_RANGE
                ):
                    outside.append(point)
        benches = [prop for prop in props.all(observation) if prop["type"] == "bench"]
        bench = min(benches, key=lambda prop: geometry.distance(_prop_centre(prop), door), default=None)
        self._inn = {"id": inn["id"], "door": door, "inside": inside, "outside": outside or [door], "bench": bench}
        return self._inn

    def _gathering_in_view(self, observation: ThreeBranchesObservation, inn) -> dict[str, float] | None:
        """Two idle villagers I can see standing together at the inn (not in its doorway): a spot
        beside them to join, or None. (After the course's Neighbor guide.)"""
        here = me.position(observation)
        idle = []
        for person in people.seen(observation):
            if not people.is_npc(person["id"]) or person["moved"] >= 0.05:
                continue
            if person["expression"]["type"] not in ("none", "laugh", "nod"):
                continue
            at_inn = (
                layout.building_at(observation, person["position"]) == inn["id"]
                or geometry.distance(person["position"], inn["door"]) <= _CLUSTER_NEAR_INN
            )
            cell = layout.cell_at(observation, person["position"])
            if at_inn and cell is not None and layout.ground_at(observation, cell) != "doorway":
                idle.append(person)
        for first in idle:
            for second in idle:
                if first is not second and geometry.distance(first["position"], second["position"]) <= _CLUSTER_PAIR_RANGE:
                    away = geometry.distance(first["position"], here) or 1.0
                    beside = {
                        "x": first["position"]["x"] + 1.2 * (here["x"] - first["position"]["x"]) / away,
                        "y": first["position"]["y"] + 1.2 * (here["y"] - first["position"]["y"]) / away,
                    }
                    cell = layout.cell_at(observation, beside)
                    return _cell_centre(_nearest_walkable_cell(observation, cell)) if cell is not None else None
        return None

    def _pick_social_spot(self, observation: ThreeBranchesObservation, inn) -> dict:
        """The next spot at the inn: the occupancy tender takes the inn bench, the building wanderer
        hosts out front, everyone else joins a group they can see or picks a free spot in or out."""
        linger = self.rng.randint(*_SOCIAL_LINGER_TICKS)
        if self.role == "occupancy_tender" and inn["bench"] is not None:
            bench = inn["bench"]
            spot = _cell_centre(_nearest_walkable_cell(observation, bench["cell"]))
            return {"spot": spot, "since": None, "linger": 10_000, "sit": bench["id"]}
        if self.role == "building_wanderer":
            return {"spot": self.rng.choice(inn["outside"]), "since": None, "linger": linger}
        joined = self._gathering_in_view(observation, inn)
        if joined is not None:
            return {"spot": joined, "since": None, "linger": linger}
        spots = inn["inside"] if inn["inside"] and self.rng.random() < _INN_INSIDE_SHARE else inn["outside"]
        return {"spot": self.rng.choice(spots), "since": None, "linger": linger}

    def _social_tick(self, observation: ThreeBranchesObservation, heading: float, here, tick: int):
        """The evening at the inn: go to a spot, spend a while there facing whoever's close, with the
        odd laugh or nod, then drift to another spot or a group."""
        inn = self._inn_layout(observation)
        if inn is None:
            return self._task_tick(observation, heading, here)  # no inn in this village: keep working
        social = self.social
        if social is None or (social["since"] is not None and tick - social["since"] >= social["linger"]):
            social = self.social = self._pick_social_spot(observation, inn)
            self._path = None
        reach = geometry.PROP_REACH if social.get("sit") else _SOCIAL_REACH
        if geometry.distance(here, social["spot"]) > reach:
            walk_heading = self._walk_toward(social["spot"], observation)
            if self._walk_failed:
                self._walk_failed = False
                self.social = None  # unreachable spot: pick another next tick
                return action.stand(heading, "shrug")
            return action.walk(walk_heading, 1.0, "none")
        if social["since"] is None:
            social["since"] = tick
        if social.get("sit"):
            usable = props.usable(observation)
            if usable is not None and usable["id"] == social["sit"]:
                return action.stand(heading, "use")  # sitting on the inn bench
        company = [
            person
            for person in people.seen(observation)
            if geometry.distance(here, person["position"]) <= _CLUSTER_PAIR_RANGE + 1.0
        ]
        if not company:
            return action.stand(geometry.heading_to(here, inn["door"]), "none")
        nearest = min(company, key=lambda person: geometry.distance(here, person["position"]))
        moods = ("laugh", "nod", "shrug") if _reached(observation, _LATE_EVENING) else ("laugh", "nod")  # getting tired
        expression = self.rng.choice(moods) if tick % 15 == self.number % 15 else "none"
        return action.stand(geometry.heading_to(here, nearest["position"]), expression)

    def _task_tick(self, observation: ThreeBranchesObservation, heading: float, here) -> ThreeBranchesAction:
        """One tick of role-driven work: answer the bell, grab a passing light, else do the job."""

        # Everyone, whatever their role, fixes a lantern/hearth they pass that's wrong for the time of day
        if self.active_task is None:
            light = self._opportunistic_toggle(observation)
            if light is not None:
                self.active_task = {
                    "prop_id": light["id"],
                    "type": light["type"],
                    "ticks_remaining": self._prop_use_ticks(light["type"]),
                }
                self.use_counter = 0
            elif self.role_target is not None and self.role_target["prop_id"] is not None:
                if self._arrived(here):
                    check = self._arrival_check(observation, heading, here)
                    if check is not None:
                        return check
                    usable = props.usable(observation)
                    if usable is None or usable["id"] != self.role_target["prop_id"]:
                        return self._reposition(observation, heading, here)
                    self.active_task = {
                        "prop_id": self.role_target["prop_id"],
                        "type": self.role_target["type"],
                        "ticks_remaining": self._prop_use_ticks(self.role_target["type"]),
                    }
                    self.use_counter = 0

        if self.active_task is not None:
            if self.use_counter >= self.active_task["ticks_remaining"]:
                # Back in the pool once it reverts (timed) or the time of day flips it (toggles)
                self._mark_done(
                    self.active_task["prop_id"], self.active_task["type"], day.tick(observation), _phase(observation)
                )
                if self.role_target is not None and self.role_target["prop_id"] == self.active_task["prop_id"]:
                    self.role_target = None
                    self.hop_point = None
                    # A break after real work; a round of toggles (lighting up, closing stalls) runs straight through
                    if self.active_task["type"] not in _TOGGLE_TYPES:
                        self.break_ticks_remaining = self.rng.randint(_BREAK_MIN_TICKS, _BREAK_MAX_TICKS)
                self.active_task = None
                self.use_counter = 0
            else:
                usable = props.usable(observation)
                if usable is not None and usable["id"] == self.active_task["prop_id"]:
                    self.use_counter += 1
                    return action.stand(heading, "use")
                # Walked off mid-task (to greet the visitor, say): "use" here would work whatever prop
                # is nearest -- once it rang the bell all day. Drop the hold; the target walks us back.
                self.active_task = None
                self.use_counter = 0

        # Break after completing a task: idle-hop for a random duration (still lighting lamps en route)
        if self.break_ticks_remaining > 0:
            self.break_ticks_remaining -= 1
            return self._idle_hop(observation, heading, here)

        if self.role == "building_wanderer":
            return self._advance_building(observation, heading, here)

        # The dawn bell: the one target of the "bell" activity, rung like any timed prop
        if self.activity == "bell" and self.role_target is None and self.bell is not None:
            self.role_target = {
                "prop_id": self.bell["id"],
                "type": "bell",
                "position": _cell_centre(_nearest_walkable_cell(observation, self.bell["cell"])),
                "direct": True,  # straight there, no hops: dawn is short
            }
            self.hop_point = None
            self.reposition_tries = 0

        # No target, or only walking rounds: take any real work there is, else rounds or a wander
        if self.role_target is None or self.role_target["prop_id"] is None:
            prop = self._pick_available_prop(observation, _ROLE_TARGET_TYPES[self.role])
            if prop is None:
                if self._rounds_due(observation):
                    return self._advance_rounds(observation, heading, here)
                return self._idle_hop(observation, heading, here)  # nothing left for now; keep it lively
            self.role_target = {
                "prop_id": prop["id"],
                "type": prop["type"],
                "position": _cell_centre(_nearest_walkable_cell(observation, prop["cell"])),
            }
            self.hop_point = None
            self.reposition_tries = 0
            self.claim_told = set()

        target_prop = props.find(observation, self.role_target["prop_id"])
        if self._someone_else_has(observation, target_prop, here, day.tick(observation)):
            self._clear_target()  # someone else has it; pick the next one next tick
            return action.stand(heading, "nod")
        self._share_claim(observation)

        if self._arrived(here):
            return action.stand(heading, "none")  # arrived; active_task starts from here next tick

        if self.role_target.get("direct"):
            walk_heading = self._walk_toward(self.role_target["position"], observation)
            if self._walk_failed:
                self._walk_failed = False
                return self._give_up_target(observation, heading)
            return action.walk(walk_heading, 1.0, "none")
        return self._advance_hop(observation, heading, here, self.role_target["position"])

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Choose this tick's action, adding a wave if we greeted a friend on the move."""
        self._passing_emote = None
        observation = {**observation, "village": self._village}  # the day's one copy of the map (see reset)
        chosen = self._decide(observation)
        if self._passing_emote is not None and int(chosen["action"]) == 0:
            return action.walk(float(chosen["heading"]), float(chosen["speed"]), self._passing_emote)
        return chosen

    def _decide(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Choose one action from current observation and persistent day state."""

        here = me.position(observation)
        heading = me.heading(observation)
        tick = day.tick(observation)
        self._nearby_ids = {person["id"] for person in people.nearby(observation)}  # in earshot this tick
        for seen_prop in props.seen(observation):
            self.prop_memory[seen_prop["prop"]] = (seen_prop["state"], tick)

        # A toggle hold finishes first whatever the clock says; switching mid-hold would flip it back
        mid_toggle = self._mid_toggle()
        activity = self.activity if mid_toggle else self._assign(observation)
        if activity != self.activity:
            self._start_activity(activity, tick)
        if self.asleep:
            return action.stand(heading, "sleep")  # sleepers don't greet, reply or startle

        # Dawn: still in bed until the bell (or our turn to get up); on the way to the bell, greet nobody
        self._greet_only_near = None
        if activity == "wake":
            stage = self._wake_stage(observation, here, tick)
            if stage in ("asleep", "stagger"):
                return action.stand(heading, "sleep")
            if stage == "to_bell":
                walk_heading = self._walk_toward(self.wake["spot"], observation)
                if not self._walk_failed:
                    return action.walk(walk_heading, 1.0, "none")
                self._walk_failed = False
                self.wake["stage"] = "up"  # no way to the bell: just start the day
            if self.wake["stage"] == "at_bell":
                self._greet_only_near = self.wake["spot"]

        # Announce role once, on first being up and about
        if not self.announced_role and activity != "sleep":
            self.announced_role = True
            role = self.role.replace("_", " ")
            article = "an" if role[0] in "aeiou" else "a"
            self._outbox.append({"to": None, "text": f"I'm {article} {role}."})
            return action.stand(heading, "none")

        # On the way somewhere the clock is waiting on -- home (to family or bed), the group gathering,
        # the inn, or a toggle round (stalls, lights) -- greetings happen on the move so the trip
        # doesn't stall; once there, people stop and greet properly
        self._hurrying = (
            activity == "sleep"
            or (activity == "family" and self.family_evening is not None and self.family_evening["stage"] == "going")
            or (
                activity == "gather"
                and self.gathering is not None
                and geometry.distance(here, self.gathering["spot"]) > _GATHER_REACH
            )
            or (activity == "social" and (self.social is None or self.social["since"] is None))
            or (self.role_target is not None and self.role_target["type"] in _URGENT_TYPES)
        )
        if not mid_toggle:
            reaction = self._react_to_people(observation)
            if reaction is not None:
                return reaction
            following = self._follow_tick(observation, activity, here, tick)
            if following is not None:
                return following

        if activity == "family":
            return self._family_evening_tick(observation, heading, here, tick)
        if activity == "sleep":
            return self._sleep_tick(observation, heading, here)
        if activity == "wake" and self.wake["stage"] == "at_bell":
            return action.stand(geometry.heading_to(here, _prop_centre(self.bell)), "none")  # gathered at the bell
        if activity == "gather":
            return self._gather_tick(observation, heading, here, tick)
        if activity == "social":
            return self._social_tick(observation, heading, here, tick)
        return self._task_tick(observation, heading, here)

    def _follow_tick(self, observation: ThreeBranchesObservation, activity: str, here, tick: int):
        """The building wanderer walking along with the visitor after greeting them, like a guide,
        for a little while -- unless the visitor's gone or the day needs it somewhere (family, bed)."""
        follow = self.following
        if follow is None:
            return None
        person = people.find(observation, follow["id"])
        if person is None or tick >= follow["until"] or activity not in ("work", "social", "gather"):
            self.following = None
            self._path = None
            return None
        if geometry.distance(here, person["position"]) > _FOLLOW_DISTANCE:
            return action.walk(self._walk_toward(person["position"], observation), 1.0, "none")
        return action.stand(geometry.heading_to(here, person["position"]), "none")

    def _assign(self, observation: ThreeBranchesObservation) -> str:
        """This tick's activity; the first rule that matches wins."""
        if self.family_evening is not None or self._family_evening_due(observation):
            return "family"  # before bedtime, so a late family evening isn't cut short
        if _reached(observation, self.bedtime):
            return "sleep"
        if self.activity == "bell" and not self._bell_rung_or_missed(observation):
            return "bell"  # a bell trip still under way carries on past the end of dawn, to the deadline
        if self.activity == "wake" and self.wake["stage"] != "up":
            return "wake"  # still getting up (a late bell, the muster) carries on past the end of dawn
        phase = _phase(observation)
        activity = _SCHEDULES[self.role][_PHASE_NAMES.index(phase)]
        if activity == "wake" and self.wake["stage"] == "up":
            activity = "work"
        if activity == "bell" and self._bell_rung_or_missed(observation):
            return "sleep"  # the dawn bell has rung (or it's too late to): off to bed
        if activity == "work" and (self.gathering is not None or self._join_gathering(observation, phase)):
            return "gather"
        if activity == "social" and self._evening_duty_left(observation):
            return "work"
        return activity

    def _join_gathering(self, observation: ThreeBranchesObservation, phase: str) -> bool:
        """Join this phase's group gathering if it's on, we haven't been yet, and we're between tasks
        (the gathering comes after finishing whatever we're tending). Sets self.gathering."""
        if self.group is None or phase not in _GATHER_SPOTS[self.group] or phase in self.gathered:
            return False
        walking_to_work = self.role_target is not None and self.role_target["prop_id"] is not None
        if self.active_task is not None or (walking_to_work and self.role not in _GATHER_ON_THE_WAY):
            return False
        start = _GATHER_START[self.group]
        end = _tick_at((phase, start + _GATHER_LENGTH))
        if not _reached(observation, (phase, start)) or day.tick(observation) > end - _GATHER_JOIN_MARGIN_TICKS:
            return False
        self.gathered.add(phase)
        spot = _gathering_spot(observation, self.group, phase)
        if spot is None:
            return False
        self.gathering = {"spot": spot, "end": end}
        return True

    def _evening_duty_left(self, observation: ThreeBranchesObservation) -> bool:
        """Before going to the inn: a last round of work until this role's _SOCIAL_FROM time, and
        any stalls to close or lights to light. Once at the inn, stay (no flip-flopping)."""
        if self.activity == "social":
            return False
        if not _reached(observation, ("evening", _SOCIAL_FROM.get(self.role, 0.0))):
            return True
        types = _ROLE_TARGET_TYPES.get(self.role, ())
        if not types or not all(prop_type in _TOGGLE_TYPES for prop_type in types):
            return False
        if self.active_task is not None or (self.role_target is not None and self.role_target["prop_id"] is not None):
            return True  # finish the one in hand
        return bool(self._available_props(observation, types))

    def _wake_stage(self, observation: ThreeBranchesObservation, here, tick: int) -> str:
        """Advance the dawn wake-up and return its stage: asleep or stagger (both still in bed),
        to_bell, at_bell, or up (the day has started)."""
        wake = self.wake
        if wake["stage"] == "asleep":
            rung = self.bell is not None and day.bell_ringing(observation)
            if rung or _reached(observation, _WAKE_LATEST):
                spot = _cell_centre(_nearest_walkable_cell(observation, self.bell["cell"])) if rung else None
                if rung and geometry.distance(here, spot) <= _BELL_RANGE:
                    wake.update(stage="to_bell", until=tick + _MUSTER_GIVE_UP_TICKS, spot=spot)
                else:
                    dawn_ticks = _PHASES[0][2] - _PHASES[0][1] + 1
                    stagger = _WAKE_STAGGER * dawn_ticks * ((self.number - 1) % len(_ROLE_ORDER)) / (len(_ROLE_ORDER) - 1)
                    wake.update(stage="stagger", until=tick + round(stagger))
        if wake["stage"] == "stagger" and tick >= wake["until"]:
            wake["stage"] = "up"
        if wake["stage"] == "to_bell":
            if geometry.distance(here, wake["spot"]) <= _MUSTER_REACH:
                wake.update(stage="at_bell", until=tick + _MUSTER_TICKS)
            elif tick >= wake["until"]:
                wake["stage"] = "up"
        if wake["stage"] == "at_bell" and tick >= wake["until"]:
            wake["stage"] = "up"
        return wake["stage"]

    def _bell_rung_or_missed(self, observation: ThreeBranchesObservation) -> bool:
        """The dawn bell is done: we rang it, someone else is ringing it, or it's past the deadline."""
        return (
            self.bell is None
            or self.bell["id"] in self.timed_done
            or (day.bell_ringing(observation) and self.active_task is None)
            or _reached(observation, _BELL_DEADLINE)
        )

    def _start_activity(self, activity: str, tick: int) -> None:
        """Drop whatever the last activity was doing and start fresh on this one."""
        self.activity = activity
        self.active_task = None
        self.use_counter = 0
        self._clear_target()
        self._path = None
        self.stuck_ticks = 0
        self.break_ticks_remaining = 0
        self.asleep = False
        self.social = None
        if activity == "family":
            self.family_evening = {"stage": "going", "since": tick}

    def _sleep_tick(self, observation: ThreeBranchesObservation, heading: float, here) -> ThreeBranchesAction:
        """Walk home to the spot this villager woke up in (inside, clear of the doorway) and sleep."""
        if self.bed_spot is None:
            self.asleep = True  # no home (e.g. the visitor seat): sleep where we are
            return action.stand(heading, "sleep")
        distance = geometry.distance(here, self.bed_spot)
        self.stuck_ticks = self.stuck_ticks + 1 if me.moved(observation) < 0.05 else 0
        if distance <= _BED_REACH or (distance <= _BED_STUCK_RANGE and self.stuck_ticks >= 10):
            self.asleep = True
            return action.stand(heading, "sleep")
        if self.stuck_ticks >= _STUCK_TICKS_LIMIT:
            self._path = None  # blocked on the way: plan again
            self.stuck_ticks = 0
        walk_heading = self._walk_toward(self.bed_spot, observation)
        if self._walk_failed:
            self._walk_failed = False
            if self.home_doorway is not None and self.bed_spot != self.home_doorway:
                self.bed_spot = self.home_doorway  # no route to bed: the doorway will do
            else:
                self.asleep = True
                return action.stand(heading, "sleep")
        return action.walk(walk_heading, 1.0, "none")

    # Optional: messaging. On your turn, chat receives messages addressed to your player since
    # its previous turn. Return messages with a recipient and text, or nothing to stay silent.
    # Use None as the recipient to broadcast. Every message is recorded and shown in replays.
    #
    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """Note task claims from other villagers, then send every message act() queued."""
        for message in inbox or ():
            sender = message.get("from")
            if not isinstance(sender, str) or not people.is_npc(sender):
                continue
            text = str(message.get("text", ""))
            match = _CLAIM_PATTERN.fullmatch(text)
            if match:
                self.claims[match.group(1)] = (sender, int(message.get("tick", 0)))
            released = _RELEASE_PATTERN.fullmatch(text)
            if released and self.claims.get(released.group(1), (None,))[0] == sender:
                del self.claims[released.group(1)]
        for message in inbox or ():
            sender = message.get("from")
            if not isinstance(sender, str) or message.get("to") != self.player_id:
                continue
            text = str(message.get("text", ""))
            if _CLAIM_PATTERN.fullmatch(text) or _RELEASE_PATTERN.fullmatch(text):
                continue
            sent = int(message.get("tick", 0))
            if self._cooled(sender, sent):
                self._greeted_by(sender, sent)  # spoken to directly: answer them
        if not self._outbox:
            return None
        # The harness takes one message per recipient (plus one broadcast) per tick and drops the
        # rest. Send the first for each; hold back a leftover reply for next tick, drop anything else.
        messages, held, recipients = [], [], set()
        for message in self._outbox:
            if message["to"] not in recipients:
                recipients.add(message["to"])
                messages.append({"to": message["to"], "text": message["text"]})
            elif message.get("reply"):
                held.append(message)
        self._outbox = held
        return messages
