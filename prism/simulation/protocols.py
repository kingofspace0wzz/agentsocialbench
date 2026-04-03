# prism/simulation/protocols.py
"""Named protocol presets for the graph-driven simulation engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationProtocol:
    turn_policy: str        # "round_robin" | "coordinator_first"
    message_routing: str    # "mixed" (group + private)
    termination: str        # "consensus" | "coordinator_decides" | "max_rounds"
    max_rounds: int


PROTOCOLS = {
    "multi_party_group": SimulationProtocol(
        turn_policy="round_robin",
        message_routing="mixed",
        termination="consensus",
        max_rounds=15,
    ),
    "hub_and_spoke": SimulationProtocol(
        turn_policy="coordinator_first",
        message_routing="mixed",
        termination="coordinator_decides",
        max_rounds=12,
    ),
    "competitive": SimulationProtocol(
        turn_policy="round_robin",
        message_routing="mixed",
        termination="coordinator_decides",
        max_rounds=12,
    ),
    "affinity_modulated": SimulationProtocol(
        turn_policy="round_robin",
        message_routing="mixed",
        termination="consensus",
        max_rounds=15,
    ),
}


def get_protocol(category: str) -> SimulationProtocol:
    if category not in PROTOCOLS:
        raise ValueError(f"Unknown category: {category}. Valid: {list(PROTOCOLS.keys())}")
    return PROTOCOLS[category]
