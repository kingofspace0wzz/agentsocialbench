# prism/simulation/social_graph.py
"""Social graph model for multi-party scenarios."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Edge:
    from_user: str
    to_user: str
    context: str
    affinity: str  # close | friend | acquaintance | stranger


class SocialGraph:
    """Scenario relational model: participants + directed edges with context and affinity."""

    def __init__(self, participants: list[dict], edges: list[Edge]):
        self._participants = {p["user_name"]: p for p in participants}
        self._participants_lower = {p["user_name"].lower(): p for p in participants}
        self._edges = edges
        # Index edges by from_user (lowercase)
        self._edges_from: dict[str, list[Edge]] = {}
        for e in edges:
            self._edges_from.setdefault(e.from_user, []).append(e)

    @classmethod
    def from_scenario(cls, scenario: dict) -> SocialGraph:
        participants = scenario.get("participants", [])
        raw_edges = scenario.get("social_graph", {}).get("edges", [])
        edges = [
            Edge(
                from_user=e["from"].lower(),
                to_user=e["to"].lower(),
                context=e.get("context", ""),
                affinity=e.get("affinity", "stranger"),
            )
            for e in raw_edges
        ]
        return cls(participants, edges)

    def get_affinity(self, from_user: str, to_user: str) -> str | None:
        for e in self._edges_from.get(from_user.lower(), []):
            if e.to_user == to_user.lower():
                return e.affinity
        return None

    def get_relationship(self, from_user: str, to_user: str) -> dict | None:
        for e in self._edges_from.get(from_user.lower(), []):
            if e.to_user == to_user.lower():
                return {"context": e.context, "affinity": e.affinity}
        return None

    def get_edges_for(self, user: str) -> list[Edge]:
        return self._edges_from.get(user.lower(), [])

    def get_coordinator(self) -> dict | None:
        for p in self._participants.values():
            if p.get("task_role") == "coordinator":
                return p
        return None

    def get_all_participants(self) -> list[str]:
        return list(self._participants.keys())

    def get_participant(self, name: str) -> dict | None:
        return self._participants.get(name) or self._participants_lower.get(name.lower())

    def resolve_broadcast_recipients(self) -> list[str]:
        return [p["agent_name"] for p in self._participants.values()]
