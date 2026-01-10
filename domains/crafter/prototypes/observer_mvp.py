"""
MVP: Observer watching Crafter gameplay, building ontology from observations.

Key features:
- Evidence has annotation field for freeform notes
- Single evidence can update multiple claims across different premises
- Beliefs evolve as evidence accumulates or contradicts
- Evidentiary time: can replay belief evolution at any point
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# EVIDENCE SYSTEM
# =============================================================================


class EvidenceType(Enum):
    OBSERVATION = "observation"  # Saw something happen
    MEASUREMENT = "measurement"  # Counted/measured something
    INFERENCE = "inference"  # Derived from other beliefs
    CONTRADICTION = "contradiction"  # Saw something that conflicts


@dataclass
class Evidence:
    """A piece of evidence that can support multiple claims."""

    id: str
    type: EvidenceType
    source: str  # e.g., "step_1234"
    observation: str  # What was observed
    time: int = 0  # Evidentiary time (step number)
    annotation: str = ""  # Freeform notes for clarity
    weight: float = 1.0

    def __repr__(self):
        return f"Evidence(t={self.time}, {self.observation[:40]}...)"


@dataclass
class Claim:
    subject: str  # e.g., "zombie"
    predicate: str  # e.g., "damage"
    value: Any
    confidence: float = 0.0
    relevance: float = 1.0  # How important/useful is this fact? (0-1)
    created_at: int = 0  # Evidentiary time when claim was created
    updated_at: int = 0  # Last update time
    evidence_ids: List[str] = field(default_factory=list)  # Supporting evidence
    contradiction_ids: List[str] = field(default_factory=list)  # Contradicting evidence
    superseded_by: Optional[str] = None
    superseded_at: Optional[int] = None  # When it was superseded

    @property
    def key(self) -> str:
        return f"{self.subject}.{self.predicate}"

    @property
    def status(self) -> str:
        if self.superseded_by:
            return "superseded"
        if len(self.contradiction_ids) > len(self.evidence_ids):
            return "disproven"
        if self.confidence >= 0.7:
            return "confident"
        if self.confidence >= 0.4:
            return "probable"
        return "uncertain"

    @property
    def score(self) -> float:
        """Combined score = confidence * relevance."""
        return self.confidence * self.relevance


@dataclass
class RelevanceEdge:
    """Pairwise relevance between two claims/subjects."""

    source: str  # claim key or subject name
    target: str  # claim key or subject name
    weight: float  # 0-1, how related are these?
    relation: str = ""  # e.g., "damages", "drops", "crafts_into"


class OntologyRecord:
    """Evidence-based ontology where evidence can link to multiple claims."""

    def __init__(self):
        self.claims: Dict[str, Claim] = {}  # key = f"{subject}.{predicate}"
        self.evidence: Dict[str, Evidence] = {}  # id -> Evidence
        self.history: List[Dict] = []  # Full history with timestamps
        self.current_time: int = 0  # Current evidentiary time
        self._evidence_counter = 0

        # For replay: store superseded claims
        self.superseded_claims: Dict[str, List[Claim]] = {}  # key -> [old claims]

        # Relevance graph
        self.relevance_edges: List[RelevanceEdge] = []
        self._relevance_index: Dict[str, List[RelevanceEdge]] = {}  # subject -> edges

    def set_time(self, t: int):
        """Set current evidentiary time."""
        self.current_time = t

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"ev_{self._evidence_counter}"

    def _claim_key(self, subject: str, predicate: str) -> str:
        return f"{subject}.{predicate}"

    def _update_confidence(self, claim: Claim, at_time: Optional[int] = None):
        """Compute confidence from evidence, optionally only using evidence up to at_time."""
        if at_time is None:
            evidence_ids = claim.evidence_ids
            contradiction_ids = claim.contradiction_ids
        else:
            evidence_ids = [eid for eid in claim.evidence_ids if self.evidence[eid].time <= at_time]
            contradiction_ids = [
                eid for eid in claim.contradiction_ids if self.evidence[eid].time <= at_time
            ]

        supporting = sum(self.evidence[eid].weight for eid in evidence_ids)
        contradicting = sum(self.evidence[eid].weight for eid in contradiction_ids)
        claim.confidence = supporting / (supporting + contradicting + 1.0)

    def add_evidence(self, evidence: Evidence) -> str:
        """Add evidence to the store."""
        if not evidence.id:
            evidence.id = self._next_evidence_id()
        if evidence.time == 0:
            evidence.time = self.current_time
        self.evidence[evidence.id] = evidence
        return evidence.id

    def observe(self, subject: str, predicate: str, value: Any, evidence: Evidence) -> Claim:
        """Record an observation, creating or updating a claim."""
        key = self._claim_key(subject, predicate)
        t = self.current_time

        # Ensure evidence is stored
        if evidence.id not in self.evidence:
            self.add_evidence(evidence)

        if key not in self.claims:
            # New claim
            claim = Claim(
                subject=subject, predicate=predicate, value=value, created_at=t, updated_at=t
            )
            claim.evidence_ids.append(evidence.id)
            self._update_confidence(claim)
            self.claims[key] = claim
            self.history.append(
                {
                    "time": t,
                    "action": "new_claim",
                    "key": key,
                    "value": value,
                    "confidence": claim.confidence,
                    "evidence_id": evidence.id,
                    "source": evidence.source,
                }
            )
        else:
            claim = self.claims[key]
            claim.updated_at = t
            if claim.value == value:
                # Confirming evidence
                if evidence.id not in claim.evidence_ids:
                    claim.evidence_ids.append(evidence.id)
                self._update_confidence(claim)
                self.history.append(
                    {
                        "time": t,
                        "action": "confirm",
                        "key": key,
                        "confidence": claim.confidence,
                        "evidence_id": evidence.id,
                        "source": evidence.source,
                    }
                )
            else:
                # Contradicting evidence!
                if evidence.id not in claim.contradiction_ids:
                    claim.contradiction_ids.append(evidence.id)
                self._update_confidence(claim)
                self.history.append(
                    {
                        "time": t,
                        "action": "contradict",
                        "key": key,
                        "old_value": claim.value,
                        "observed_value": value,
                        "confidence": claim.confidence,
                        "evidence_id": evidence.id,
                        "source": evidence.source,
                    }
                )

                # If heavily contradicted, create superseding claim
                if claim.confidence < 0.3:
                    # Store old claim for replay
                    if key not in self.superseded_claims:
                        self.superseded_claims[key] = []
                    old_claim = deepcopy(claim)
                    old_claim.superseded_at = t
                    self.superseded_claims[key].append(old_claim)

                    # Create new claim
                    new_claim = Claim(
                        subject=subject,
                        predicate=predicate,
                        value=value,
                        created_at=t,
                        updated_at=t,
                    )
                    new_claim.evidence_ids.append(evidence.id)
                    self._update_confidence(new_claim)
                    claim.superseded_by = key
                    claim.superseded_at = t
                    self.claims[key] = new_claim
                    self.history.append(
                        {
                            "time": t,
                            "action": "supersede",
                            "key": key,
                            "old_value": old_claim.value,
                            "new_value": value,
                            "evidence_id": evidence.id,
                            "source": evidence.source,
                        }
                    )
                    return new_claim

        return self.claims[key]

    def observe_multiple(self, observations: List[Dict], evidence: Evidence) -> List[Claim]:
        """
        Single evidence updates multiple claims.
        observations = [{"subject": "zombie", "predicate": "sleep_damage", "value": 7}, ...]
        """
        # Store evidence once
        if evidence.id not in self.evidence:
            self.add_evidence(evidence)

        claims = []
        for obs in observations:
            claim = self.observe(
                subject=obs["subject"],
                predicate=obs["predicate"],
                value=obs["value"],
                evidence=evidence,
            )
            claims.append(claim)
        return claims

    def compile(self, min_confidence: float = 0.4) -> Dict[str, Any]:
        """Compile current best knowledge."""
        ontology = {}
        uncertain = []

        for claim in self.claims.values():
            if claim.superseded_by:
                continue

            if claim.confidence >= min_confidence:
                if claim.subject not in ontology:
                    ontology[claim.subject] = {}
                ontology[claim.subject][claim.predicate] = {
                    "value": claim.value,
                    "confidence": round(claim.confidence, 2),
                    "status": claim.status,
                    "evidence_count": len(claim.evidence_ids),
                }
            else:
                uncertain.append(
                    {
                        "claim": f"{claim.subject}.{claim.predicate}={claim.value}",
                        "confidence": round(claim.confidence, 2),
                    }
                )

        return {"knowledge": ontology, "uncertain": uncertain}

    def get_evidence_for_claim(self, subject: str, predicate: str) -> List[Evidence]:
        """Get all evidence for a claim."""
        key = self._claim_key(subject, predicate)
        if key not in self.claims:
            return []
        claim = self.claims[key]
        return [self.evidence[eid] for eid in claim.evidence_ids]

    # =========================================================================
    # REPLAY METHODS
    # =========================================================================

    def compile_at_time(self, at_time: int, min_confidence: float = 0.4) -> Dict[str, Any]:
        """Compile ontology as it existed at a specific time."""
        ontology = {}
        uncertain = []

        for key, claim in self.claims.items():
            # Skip claims created after at_time
            if claim.created_at > at_time:
                continue

            # Check if this claim was superseded before at_time
            # If so, we need to use the old claim
            if key in self.superseded_claims:
                # Find the claim that was active at at_time
                active_claim = None
                for old_claim in self.superseded_claims[key]:
                    if old_claim.created_at <= at_time and (
                        old_claim.superseded_at is None or old_claim.superseded_at > at_time
                    ):
                        active_claim = old_claim
                        break
                if active_claim is None and claim.created_at <= at_time:
                    active_claim = claim
                if active_claim is None:
                    continue
                claim = active_claim

            # Recompute confidence using only evidence up to at_time
            evidence_ids = [eid for eid in claim.evidence_ids if self.evidence[eid].time <= at_time]
            contradiction_ids = [
                eid for eid in claim.contradiction_ids if self.evidence[eid].time <= at_time
            ]

            if not evidence_ids:
                continue

            supporting = sum(self.evidence[eid].weight for eid in evidence_ids)
            contradicting = sum(self.evidence[eid].weight for eid in contradiction_ids)
            confidence = supporting / (supporting + contradicting + 1.0)

            if confidence >= min_confidence:
                if claim.subject not in ontology:
                    ontology[claim.subject] = {}
                ontology[claim.subject][claim.predicate] = {
                    "value": claim.value,
                    "confidence": round(confidence, 2),
                    "evidence_count": len(evidence_ids),
                }
            else:
                uncertain.append(
                    {
                        "claim": f"{claim.subject}.{claim.predicate}={claim.value}",
                        "confidence": round(confidence, 2),
                    }
                )

        return {"time": at_time, "knowledge": ontology, "uncertain": uncertain}

    def get_history_up_to(self, at_time: int) -> List[Dict]:
        """Get history entries up to a specific time."""
        return [h for h in self.history if h["time"] <= at_time]

    def replay(self, times: List[int] = None, min_confidence: float = 0.4) -> List[Dict]:
        """
        Replay belief evolution at specified times.
        If times is None, uses all unique times from history.
        """
        if times is None:
            times = sorted({h["time"] for h in self.history})

        snapshots = []
        for t in times:
            snapshot = self.compile_at_time(t, min_confidence)
            snapshot["events_at_time"] = [h for h in self.history if h["time"] == t]
            snapshots.append(snapshot)

        return snapshots

    def print_replay(self, times: List[int] = None, min_confidence: float = 0.3):
        """Print a visual replay of belief evolution."""
        snapshots = self.replay(times, min_confidence)

        print("\n" + "=" * 70)
        print("BELIEF EVOLUTION REPLAY")
        print("=" * 70)

        for snap in snapshots:
            t = snap["time"]
            print(f"\n{'─' * 70}")
            print(f"TIME = {t}")
            print(f"{'─' * 70}")

            # Show events at this time
            for event in snap["events_at_time"]:
                action = event["action"].upper()
                if action == "NEW_CLAIM":
                    print(
                        f"  + NEW: {event['key']} = {event['value']} (conf: {event['confidence']:.2f})"
                    )
                elif action == "CONFIRM":
                    print(f"  ✓ CONFIRM: {event['key']} (conf: {event['confidence']:.2f})")
                elif action == "CONTRADICT":
                    print(
                        f"  ✗ CONTRADICT: {event['key']}: expected {event['old_value']}, saw {event['observed_value']}"
                    )
                elif action == "SUPERSEDE":
                    print(
                        f"  ⟳ SUPERSEDE: {event['key']}: {event['old_value']} → {event['new_value']}"
                    )

            # Show current beliefs
            if snap["knowledge"]:
                print(f"\n  Beliefs at t={t}:")
                for subject, props in sorted(snap["knowledge"].items()):
                    for pred, data in props.items():
                        conf_bar = "█" * int(data["confidence"] * 10) + "░" * (
                            10 - int(data["confidence"] * 10)
                        )
                        print(f"    {subject}.{pred} = {data['value']}")
                        print(
                            f"      [{conf_bar}] {data['confidence']:.0%} ({data['evidence_count']} evidence)"
                        )

    def get_belief_timeline(self, subject: str, predicate: str) -> List[Dict]:
        """Get the timeline of a specific belief."""
        key = self._claim_key(subject, predicate)
        timeline = []

        # Get all history for this claim
        for h in self.history:
            if h["key"] == key:
                timeline.append(h)

        return timeline

    # =========================================================================
    # RELEVANCE METHODS
    # =========================================================================

    def set_relevance(self, subject: str, predicate: str, relevance: float):
        """Set relevance score for a claim."""
        key = self._claim_key(subject, predicate)
        if key in self.claims:
            self.claims[key].relevance = relevance

    def add_relevance_edge(self, source: str, target: str, weight: float, relation: str = ""):
        """Add a relevance edge between two subjects."""
        edge = RelevanceEdge(source=source, target=target, weight=weight, relation=relation)
        self.relevance_edges.append(edge)

        # Index both directions
        if source not in self._relevance_index:
            self._relevance_index[source] = []
        self._relevance_index[source].append(edge)

        if target not in self._relevance_index:
            self._relevance_index[target] = []
        self._relevance_index[target].append(edge)

    def get_related_subjects(
        self, subject: str, min_weight: float = 0.0
    ) -> List[Tuple[str, float, str]]:
        """Get subjects related to this one, sorted by relevance weight."""
        if subject not in self._relevance_index:
            return []

        related = []
        for edge in self._relevance_index[subject]:
            other = edge.target if edge.source == subject else edge.source
            if edge.weight >= min_weight:
                related.append((other, edge.weight, edge.relation))

        return sorted(related, key=lambda x: -x[1])

    def expand_from_seeds(
        self, seeds: List[str], hops: int = 1, min_weight: float = 0.3
    ) -> Set[str]:
        """Expand from seed subjects following relevance edges."""
        subjects = set(seeds)
        frontier = set(seeds)

        for _ in range(hops):
            new_frontier = set()
            for subj in frontier:
                for other, _weight, _ in self.get_related_subjects(subj, min_weight):
                    if other not in subjects:
                        new_frontier.add(other)
                        subjects.add(other)
            frontier = new_frontier

        return subjects

    # =========================================================================
    # COMPILE TO TEXT
    # =========================================================================

    def compile_to_text(
        self,
        min_confidence: float = 0.4,
        min_relevance: float = 0.0,
        subjects: Set[str] = None,
        include_evidence: bool = False,
    ) -> str:
        """Compile ontology to readable text format."""
        lines = []
        lines.append("# Ontology")
        lines.append("")

        # Group by subject
        by_subject: Dict[str, List[Claim]] = {}
        for claim in self.claims.values():
            if claim.superseded_by:
                continue
            if claim.confidence < min_confidence:
                continue
            if claim.relevance < min_relevance:
                continue
            if subjects and claim.subject not in subjects:
                continue

            if claim.subject not in by_subject:
                by_subject[claim.subject] = []
            by_subject[claim.subject].append(claim)

        # Sort subjects by max relevance of their claims
        def subject_score(subj):
            claims = by_subject[subj]
            return max(c.score for c in claims)

        for subject in sorted(by_subject.keys(), key=subject_score, reverse=True):
            claims = sorted(by_subject[subject], key=lambda c: -c.score)
            lines.append(f"## {subject}")

            for claim in claims:
                conf_pct = int(claim.confidence * 100)
                rel_pct = int(claim.relevance * 100)
                lines.append(
                    f"- {claim.predicate}: {claim.value}  [conf={conf_pct}%, rel={rel_pct}%]"
                )

                if include_evidence:
                    for ev_id in claim.evidence_ids[:3]:  # Limit to 3
                        ev = self.evidence[ev_id]
                        lines.append(f"    evidence: {ev.observation}")
                        if ev.annotation:
                            lines.append(f"    note: {ev.annotation}")

            # Show related subjects
            related = self.get_related_subjects(subject, min_weight=0.3)
            if related:
                rel_str = ", ".join(f"{r[0]}({r[1]:.0%})" for r in related[:5])
                lines.append(f"  related: {rel_str}")

            lines.append("")

        return "\n".join(lines)

    def compile_to_file(
        self,
        filepath: str,
        min_confidence: float = 0.4,
        min_relevance: float = 0.0,
        subjects: Set[str] = None,
        include_evidence: bool = False,
    ):
        """Write compiled ontology to a text file."""
        text = self.compile_to_text(min_confidence, min_relevance, subjects, include_evidence)
        with open(filepath, "w") as f:
            f.write(text)
        return filepath

    def compile_tiered(self, base_path: str, seeds: List[str] = None):
        """
        Compile to three tiers:
        - small.txt: High relevance only (or seed-local if seeds provided)
        - medium.txt: Medium+ relevance
        - large.txt: Everything confident
        """
        if seeds:
            # Conditional compilation from seeds
            small_subjects = self.expand_from_seeds(seeds, hops=1, min_weight=0.5)
            medium_subjects = self.expand_from_seeds(seeds, hops=2, min_weight=0.3)

            self.compile_to_file(
                f"{base_path}_small.txt", min_confidence=0.5, subjects=small_subjects
            )
            self.compile_to_file(
                f"{base_path}_medium.txt", min_confidence=0.4, subjects=medium_subjects
            )
            self.compile_to_file(
                f"{base_path}_large.txt", min_confidence=0.3, include_evidence=True
            )
        else:
            # Relevance-based tiers
            self.compile_to_file(f"{base_path}_small.txt", min_confidence=0.6, min_relevance=0.7)
            self.compile_to_file(f"{base_path}_medium.txt", min_confidence=0.5, min_relevance=0.4)
            self.compile_to_file(
                f"{base_path}_large.txt", min_confidence=0.3, include_evidence=True
            )

        return [f"{base_path}_small.txt", f"{base_path}_medium.txt", f"{base_path}_large.txt"]

    # =========================================================================
    # LENGTH-CONSTRAINED COMPILATION
    # =========================================================================

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (~4 chars per token for English)."""
        return len(text) // 4

    def _format_claim(self, claim: Claim, include_evidence: bool = False) -> str:
        """Format a single claim as text."""
        lines = []
        conf_pct = int(claim.confidence * 100)
        rel_pct = int(claim.relevance * 100)
        lines.append(
            f"- {claim.subject}.{claim.predicate}: {claim.value}  [conf={conf_pct}%, rel={rel_pct}%]"
        )

        if include_evidence:
            for ev_id in claim.evidence_ids[:2]:
                ev = self.evidence[ev_id]
                lines.append(f"    ({ev.observation})")

        return "\n".join(lines)

    def _get_ranked_claims(
        self, min_confidence: float = 0.3, subjects: Set[str] = None
    ) -> List[Claim]:
        """Get claims sorted by score (confidence * relevance)."""
        claims = []
        for claim in self.claims.values():
            if claim.superseded_by:
                continue
            if claim.confidence < min_confidence:
                continue
            if subjects and claim.subject not in subjects:
                continue
            claims.append(claim)

        return sorted(claims, key=lambda c: -c.score)

    def compile_to_length(
        self,
        max_chars: int = None,
        max_tokens: int = None,
        seeds: List[str] = None,
        min_confidence: float = 0.3,
        include_evidence: bool = False,
    ) -> str:
        """
        Compile ontology up to a character/token limit.

        Args:
            max_chars: Maximum characters (mutually exclusive with max_tokens)
            max_tokens: Maximum tokens (~4 chars each)
            seeds: If provided, prioritize these subjects and their neighbors
            min_confidence: Minimum confidence threshold
            include_evidence: Include evidence text (uses more space)

        Returns:
            Compiled text fitting within the limit
        """
        if max_tokens and not max_chars:
            max_chars = max_tokens * 4

        if not max_chars:
            max_chars = 10000  # Default 10k chars

        # Get subjects to include
        if seeds:
            # Expand from seeds, prioritizing closer nodes
            priority_subjects = list(seeds)
            for hop in range(1, 4):  # Up to 3 hops
                expanded = self.expand_from_seeds(seeds, hops=hop, min_weight=0.3)
                for subj in expanded:
                    if subj not in priority_subjects:
                        priority_subjects.append(subj)
        else:
            priority_subjects = None

        # Get ranked claims
        claims = self._get_ranked_claims(
            min_confidence, set(priority_subjects) if priority_subjects else None
        )

        # If we have priority subjects, re-sort to respect that order
        if priority_subjects:

            def priority_key(claim):
                try:
                    subj_priority = priority_subjects.index(claim.subject)
                except ValueError:
                    subj_priority = 999
                return (subj_priority, -claim.score)

            claims = sorted(claims, key=priority_key)

        # Build output incrementally
        lines = ["# Ontology", ""]
        current_subject = None
        char_count = len("\n".join(lines))

        for claim in claims:
            # Format the claim
            if claim.subject != current_subject:
                subject_header = f"\n## {claim.subject}"
                claim_text = self._format_claim(claim, include_evidence)
                new_text = subject_header + "\n" + claim_text
                current_subject = claim.subject
            else:
                new_text = self._format_claim(claim, include_evidence)

            # Check if adding this would exceed limit
            if char_count + len(new_text) + 1 > max_chars:
                # Try without evidence if we were including it
                if include_evidence:
                    new_text_short = self._format_claim(claim, include_evidence=False)
                    if claim.subject != current_subject:
                        new_text_short = f"\n## {claim.subject}\n" + new_text_short
                    if char_count + len(new_text_short) + 1 <= max_chars:
                        lines.append(new_text_short)
                        char_count += len(new_text_short) + 1
                        continue
                break

            lines.append(new_text)
            char_count += len(new_text) + 1

        result = "\n".join(lines)

        # Add metadata footer
        footer = f"\n\n[{len(result)} chars, ~{self._estimate_tokens(result)} tokens]"
        if len(result) + len(footer) <= max_chars + 50:  # Allow slight overflow for footer
            result += footer

        return result

    def compile_to_file_with_limit(
        self, filepath: str, max_chars: int = None, max_tokens: int = None, **kwargs
    ) -> str:
        """Compile to file with length limit."""
        text = self.compile_to_length(max_chars=max_chars, max_tokens=max_tokens, **kwargs)
        with open(filepath, "w") as f:
            f.write(text)
        return text

    def get_context(self, budget: str, seeds: List[str] = None) -> str:
        """
        Convenience method with human-friendly budget strings.

        Examples:
            record.get_context("500 chars")
            record.get_context("1k tokens")
            record.get_context("small")  # ~500 chars
            record.get_context("medium") # ~2000 chars
            record.get_context("large")  # ~8000 chars
        """
        budget = budget.lower().strip()

        # Named sizes
        if budget == "small" or budget == "sm":
            return self.compile_to_length(max_chars=500, seeds=seeds)
        elif budget == "medium" or budget == "med":
            return self.compile_to_length(max_chars=2000, seeds=seeds, include_evidence=True)
        elif budget == "large" or budget == "lg":
            return self.compile_to_length(max_chars=8000, seeds=seeds, include_evidence=True)
        elif budget == "full":
            return self.compile_to_text(min_confidence=0.3, include_evidence=True)

        # Parse numeric budgets
        import re

        match = re.match(r"(\d+\.?\d*)\s*(k)?\s*(chars?|tokens?|c|t)?", budget)
        if match:
            num = float(match.group(1))
            multiplier = 1000 if match.group(2) else 1
            unit = match.group(3) or "chars"

            value = int(num * multiplier)

            if unit.startswith("t"):
                return self.compile_to_length(max_tokens=value, seeds=seeds)
            else:
                return self.compile_to_length(max_chars=value, seeds=seeds)

        raise ValueError(
            f"Unknown budget format: {budget}. Try '500 chars', '1k tokens', 'small', 'medium', 'large'"
        )


# =============================================================================
# CRAFTER OBSERVER
# =============================================================================


class CrafterObserver:
    """Simulates an agent observing Crafter gameplay and building beliefs."""

    def __init__(self):
        self.record = OntologyRecord()
        self.step = 0

    def _make_evidence(
        self, observation: str, etype: EvidenceType = EvidenceType.OBSERVATION, annotation: str = ""
    ) -> Evidence:
        ev_id = f"ev_{self.step}_{self.record._evidence_counter + 1}"
        return Evidence(
            id=ev_id,
            type=etype,
            source=f"step_{self.step}",
            observation=observation,
            time=self.step,
            annotation=annotation,
        )

    def observe_event(self, event: Dict):
        """Process a gameplay event and update beliefs."""
        self.step = event.get("step", self.step + 1)
        self.record.set_time(self.step)  # Sync evidentiary time
        event_type = event["type"]

        if event_type == "damage_dealt":
            attacker = event["attacker"]
            damage = event["damage"]
            context = event.get("context", "normal")
            predicate = "sleep_damage" if context == "sleeping" else "damage"

            self.record.observe(
                subject=attacker,
                predicate=predicate,
                value=damage,
                evidence=self._make_evidence(
                    f"{attacker} dealt {damage} damage ({context})",
                    annotation=event.get("annotation", ""),
                ),
            )

        elif event_type == "resource_collected":
            source = event["source"]
            item = event["item"]
            amount = event.get("amount", 1)

            self.record.observe(
                subject=source,
                predicate="yields",
                value=item,
                evidence=self._make_evidence(
                    f"Collected {amount} {item} from {source}",
                    annotation=event.get("annotation", ""),
                ),
            )

        elif event_type == "creature_died":
            creature = event["creature"]
            hits = event["hits"]
            weapon_damage = event.get("weapon_damage", 1)
            inferred_health = hits * weapon_damage

            self.record.observe(
                subject=creature,
                predicate="health",
                value=inferred_health,
                evidence=self._make_evidence(
                    f"{creature} died after {hits} hits with {weapon_damage} damage weapon",
                    etype=EvidenceType.INFERENCE,
                    annotation=f"Health inferred: {hits} hits × {weapon_damage} dmg = {inferred_health} HP",
                ),
            )

        elif event_type == "crafting_observed":
            recipe = event["recipe"]
            inputs = event["inputs"]
            output = event["output"]

            # Single evidence, multiple claims!
            ev = self._make_evidence(
                f"Crafted {output} using {inputs}", annotation=f"Recipe discovered: {recipe}"
            )
            self.record.observe_multiple(
                [
                    {"subject": recipe, "predicate": "inputs", "value": inputs},
                    {"subject": recipe, "predicate": "output", "value": output},
                ],
                ev,
            )

        elif event_type == "sleep_attack":
            # Complex event: single observation updates MULTIPLE premises
            attacker = event["attacker"]
            damage = event["damage"]
            normal_damage = event.get("known_normal_damage")

            ev = self._make_evidence(
                f"While sleeping, {attacker} attacked for {damage} damage",
                annotation=event.get("annotation", "Critical discovery about sleep vulnerability!"),
            )

            observations = [
                {"subject": attacker, "predicate": "sleep_damage", "value": damage},
            ]

            # If we know normal damage, we can also infer the multiplier
            if normal_damage:
                multiplier = round(damage / normal_damage, 1)
                observations.append(
                    {"subject": "sleep", "predicate": "damage_multiplier", "value": multiplier}
                )
                observations.append(
                    {"subject": "player", "predicate": "sleep_is_dangerous", "value": True}
                )

            self.record.observe_multiple(observations, ev)

        elif event_type == "spawn_observed":
            creature = event["creature"]
            location = event["location"]
            time_of_day = event.get("time", "unknown")

            self.record.observe(
                subject=creature,
                predicate="spawn_location",
                value=location,
                evidence=self._make_evidence(
                    f"{creature} spawned at {location} during {time_of_day}",
                    annotation=f"Time: {time_of_day}",
                ),
            )


# =============================================================================
# SIMULATION
# =============================================================================


def simulate_gameplay_session():
    """Simulate an observer watching a Crafter gameplay session."""

    print("=" * 70)
    print("CRAFTER OBSERVER SIMULATION")
    print("=" * 70)

    observer = CrafterObserver()

    # =========================================================================
    # EARLY GAME: Basic observations
    # =========================================================================
    print("\n[EARLY GAME - Steps 1-100]")
    print("-" * 40)

    events = [
        {
            "step": 10,
            "type": "resource_collected",
            "source": "tree",
            "item": "wood",
            "annotation": "First tree I found",
        },
        {"step": 15, "type": "resource_collected", "source": "tree", "item": "wood"},
        {"step": 20, "type": "resource_collected", "source": "tree", "item": "wood"},
        {
            "step": 50,
            "type": "crafting_observed",
            "recipe": "place_table",
            "inputs": {"wood": 2},
            "output": "table",
        },
        {
            "step": 55,
            "type": "crafting_observed",
            "recipe": "make_wood_pickaxe",
            "inputs": {"wood": 1},
            "output": "wood_pickaxe",
        },
    ]

    for event in events:
        observer.observe_event(event)
        print(f"  Step {event['step']}: {event['type']}")

    print("\n  Current beliefs:")
    ontology = observer.record.compile(min_confidence=0.3)
    for subject, props in ontology["knowledge"].items():
        for pred, data in props.items():
            print(
                f"    {subject}.{pred} = {data['value']} (conf: {data['confidence']}, n={data['evidence_count']})"
            )

    # =========================================================================
    # FIRST COMBAT: Learning about zombies
    # =========================================================================
    print("\n[FIRST COMBAT - Steps 100-150]")
    print("-" * 40)

    events = [
        {
            "step": 110,
            "type": "damage_dealt",
            "attacker": "zombie",
            "damage": 2,
            "annotation": "First zombie encounter - scary!",
        },
        {"step": 115, "type": "damage_dealt", "attacker": "zombie", "damage": 2},
        {"step": 120, "type": "creature_died", "creature": "zombie", "hits": 5, "weapon_damage": 1},
    ]

    for event in events:
        observer.observe_event(event)
        print(f"  Step {event['step']}: {event['type']}")

    print("\n  Zombie beliefs:")
    ontology = observer.record.compile(min_confidence=0.3)
    if "zombie" in ontology["knowledge"]:
        for pred, data in ontology["knowledge"]["zombie"].items():
            print(f"    zombie.{pred} = {data['value']} (conf: {data['confidence']})")

    # =========================================================================
    # SLEEPING INCIDENT: Single evidence updates MULTIPLE beliefs!
    # =========================================================================
    print("\n[SLEEPING INCIDENT - Step 200]")
    print("-" * 40)
    print("  >>> Single observation will update MULTIPLE beliefs! <<<")

    # First, form weak hypothesis
    observer.step = 199
    observer.record.set_time(199)
    observer.record.observe(
        subject="zombie",
        predicate="sleep_damage",
        value=2,  # Wrong assumption!
        evidence=Evidence(
            id="ev_hypothesis_sleep",
            type=EvidenceType.INFERENCE,
            source="step_199",
            observation="Assumed sleep damage equals normal damage",
            time=199,
            annotation="No direct evidence - just a guess based on normal combat",
            weight=0.3,
        ),
    )
    print("  Hypothesis: zombie.sleep_damage = 2 (weak, conf ~0.23)")

    # Now the real observation - updates multiple beliefs at once!
    observer.step = 200
    observer.observe_event(
        {
            "step": 200,
            "type": "sleep_attack",
            "attacker": "zombie",
            "damage": 7,
            "known_normal_damage": 2,
            "annotation": "OUCH! Woke up to massive damage. Sleep is dangerous!",
        }
    )

    print("\n  After sleep attack, ONE evidence updated THREE beliefs:")
    ontology = observer.record.compile(min_confidence=0.3)
    for subject in ["zombie", "sleep", "player"]:
        if subject in ontology["knowledge"]:
            for pred, data in ontology["knowledge"][subject].items():
                print(f"    {subject}.{pred} = {data['value']} (conf: {data['confidence']})")

    # Show the shared evidence
    print("\n  Evidence details:")
    for ev in observer.record.evidence.values():
        if "sleep" in ev.observation.lower() or "sleeping" in ev.observation.lower():
            print(f"    [{ev.id}] {ev.observation}")
            if ev.annotation:
                print(f"           Note: {ev.annotation}")

    # =========================================================================
    # MORE OBSERVATIONS
    # =========================================================================
    print("\n[EXTENDED PLAY - Steps 300-500]")
    print("-" * 40)

    events = [
        {"step": 310, "type": "damage_dealt", "attacker": "zombie", "damage": 2},
        {"step": 350, "type": "damage_dealt", "attacker": "zombie", "damage": 2},
        {"step": 400, "type": "creature_died", "creature": "zombie", "hits": 5, "weapon_damage": 1},
        {
            "step": 420,
            "type": "damage_dealt",
            "attacker": "skeleton",
            "damage": 1,
            "annotation": "Skeleton arrows hurt less than zombie melee",
        },
        {
            "step": 425,
            "type": "creature_died",
            "creature": "skeleton",
            "hits": 3,
            "weapon_damage": 1,
        },
        {
            "step": 450,
            "type": "spawn_observed",
            "creature": "skeleton",
            "location": "mountain",
            "time": "night",
        },
        {"step": 470, "type": "creature_died", "creature": "cow", "hits": 3, "weapon_damage": 1},
        {
            "step": 480,
            "type": "resource_collected",
            "source": "cow",
            "item": "food",
            "amount": 6,
            "annotation": "Cows drop a lot of food!",
        },
    ]

    for event in events:
        observer.observe_event(event)

    print("  Processed 8 more observations...")

    # =========================================================================
    # FINAL COMPILED ONTOLOGY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPILED ONTOLOGY (min_confidence=0.5)")
    print("=" * 70)

    ontology = observer.record.compile(min_confidence=0.5)

    print("\n[CONFIDENT KNOWLEDGE]")
    for subject, props in sorted(ontology["knowledge"].items()):
        print(f"\n  {subject}:")
        for pred, data in props.items():
            status_icon = "✓" if data["status"] == "confident" else "~"
            print(
                f"    {status_icon} {pred}: {data['value']} (conf: {data['confidence']}, evidence: {data['evidence_count']})"
            )

    if ontology["uncertain"]:
        print("\n[UNCERTAIN - NEEDS MORE EVIDENCE]")
        for item in ontology["uncertain"]:
            print(f"  ? {item['claim']} (conf: {item['confidence']})")

    # =========================================================================
    # EVIDENCE WITH ANNOTATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVIDENCE LOG (with annotations)")
    print("=" * 70)

    for ev in observer.record.evidence.values():
        print(f"\n  [{ev.id}] {ev.type.value.upper()}")
        print(f"    Source: {ev.source}")
        print(f"    Observation: {ev.observation}")
        if ev.annotation:
            print(f"    Annotation: {ev.annotation}")

    # =========================================================================
    # SET RELEVANCE SCORES
    # =========================================================================
    print("\n" + "=" * 70)
    print("SETTING RELEVANCE SCORES")
    print("=" * 70)

    # Combat-related facts are highly relevant
    observer.record.set_relevance("zombie", "damage", 0.9)
    observer.record.set_relevance("zombie", "sleep_damage", 1.0)  # Critical!
    observer.record.set_relevance("zombie", "health", 0.8)
    observer.record.set_relevance("skeleton", "damage", 0.9)
    observer.record.set_relevance("skeleton", "health", 0.8)
    observer.record.set_relevance("sleep", "damage_multiplier", 1.0)
    observer.record.set_relevance("player", "sleep_is_dangerous", 1.0)

    # Resource facts are medium relevance
    observer.record.set_relevance("tree", "yields", 0.5)
    observer.record.set_relevance("cow", "yields", 0.6)
    observer.record.set_relevance("cow", "health", 0.4)

    # Crafting is lower relevance (can look up)
    observer.record.set_relevance("place_table", "inputs", 0.3)
    observer.record.set_relevance("place_table", "output", 0.3)
    observer.record.set_relevance("make_wood_pickaxe", "inputs", 0.3)
    observer.record.set_relevance("make_wood_pickaxe", "output", 0.3)

    print("  Set relevance: combat facts=0.8-1.0, resources=0.4-0.6, crafting=0.3")

    # =========================================================================
    # ADD RELEVANCE EDGES (semantic relationships)
    # =========================================================================
    print("\n  Adding relevance edges...")

    # Combat relationships
    observer.record.add_relevance_edge("zombie", "player", 0.9, "attacks")
    observer.record.add_relevance_edge("skeleton", "player", 0.9, "attacks")
    observer.record.add_relevance_edge("zombie", "sleep", 0.95, "bonus_damage_when")
    observer.record.add_relevance_edge("sleep", "player", 0.8, "state_of")

    # Resource relationships
    observer.record.add_relevance_edge("tree", "wood", 0.8, "yields")
    observer.record.add_relevance_edge("cow", "food", 0.8, "yields")
    observer.record.add_relevance_edge("wood", "place_table", 0.6, "used_by")
    observer.record.add_relevance_edge("wood", "make_wood_pickaxe", 0.6, "used_by")

    # Creature relationships
    observer.record.add_relevance_edge("zombie", "skeleton", 0.5, "similar_to")

    print("  Added edges: zombie-player, zombie-sleep, tree-wood, etc.")

    # =========================================================================
    # COMPILE TO TEXT FILES
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPILING TO TEXT FILES")
    print("=" * 70)

    # Compile tiered (by relevance)
    files = observer.record.compile_tiered("ontology")
    for f in files:
        with open(f) as fp:
            lines = len(fp.readlines())
        print(f"  {f}: {lines} lines")

    # Also compile conditional on "zombie" (combat-focused)
    print("\n  Compiling zombie-focused subset...")
    observer.record.compile_tiered("ontology_combat", seeds=["zombie", "skeleton"])

    # =========================================================================
    # LENGTH-CONSTRAINED CONTEXT
    # =========================================================================
    print("\n" + "=" * 70)
    print("LENGTH-CONSTRAINED CONTEXT")
    print("=" * 70)

    # Different budget formats
    print("\n--- get_context('200 chars') ---")
    print(observer.record.get_context("200 chars"))

    print("\n--- get_context('500 chars', seeds=['zombie']) ---")
    print(observer.record.get_context("500 chars", seeds=["zombie"]))

    print("\n--- get_context('100 tokens') ---")
    print(observer.record.get_context("100 tokens"))

    print("\n--- get_context('small') ---")
    print(observer.record.get_context("small"))

    print("\n--- get_context('1k chars', seeds=['skeleton']) ---")
    print(observer.record.get_context("1k chars", seeds=["skeleton"]))

    # =========================================================================
    # REPLAY: Watch beliefs evolve over time
    # =========================================================================
    observer.record.print_replay(times=[10, 110, 199, 200, 400], min_confidence=0.2)

    # =========================================================================
    # SINGLE BELIEF TIMELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("TIMELINE: zombie.sleep_damage")
    print("=" * 70)

    timeline = observer.record.get_belief_timeline("zombie", "sleep_damage")
    for entry in timeline:
        t = entry["time"]
        action = entry["action"]
        if action == "new_claim":
            print(f"  t={t}: CREATED with value={entry['value']}, conf={entry['confidence']:.2f}")
        elif action == "contradict":
            print(
                f"  t={t}: CONTRADICTED - expected {entry['old_value']}, saw {entry['observed_value']}, conf={entry['confidence']:.2f}"
            )
        elif action == "supersede":
            print(f"  t={t}: SUPERSEDED - {entry['old_value']} → {entry['new_value']}")


if __name__ == "__main__":
    simulate_gameplay_session()
