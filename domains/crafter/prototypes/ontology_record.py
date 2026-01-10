"""
Ontology Record System

Instead of static ontology, we have:
- Claims: assertions with confidence
- Evidence: observations supporting/refuting claims
- Provenance: source of knowledge (code reading, gameplay, documentation, etc.)
- Supersession: when claims are disproven and replaced

The OntologyEngine compiles current best knowledge from the evidence graph.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
import json


# =============================================================================
# EVIDENCE TYPES
# =============================================================================


class EvidenceType(Enum):
    SOURCE_CODE = "source_code"  # Read from actual code (highest confidence)
    DOCUMENTATION = "documentation"  # Official docs
    GAMEPLAY_OBSERVATION = "gameplay"  # Observed during play
    INFERENCE = "inference"  # Derived from other claims
    HYPOTHESIS = "hypothesis"  # Untested belief
    CONTRADICTION = "contradiction"  # Evidence against a claim


class ClaimStatus(Enum):
    ACTIVE = "active"  # Current best belief
    SUPERSEDED = "superseded"  # Replaced by newer claim
    DISPROVEN = "disproven"  # Evidence contradicts
    UNCERTAIN = "uncertain"  # Conflicting evidence


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================


@dataclass
class Evidence:
    """A piece of evidence supporting or refuting a claim."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: EvidenceType = EvidenceType.HYPOTHESIS
    source: str = ""  # e.g., "crafter/objects.py:142" or "episode_1234_step_567"
    observation: str = ""  # What was observed
    timestamp: datetime = field(default_factory=datetime.now)
    weight: float = 1.0  # How much this evidence should count
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Evidence({self.type.value}: {self.observation[:50]}...)"


@dataclass
class Claim:
    """An assertion about the world with confidence and evidence."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # The claim itself
    subject: str = ""  # e.g., "entities.zombie"
    predicate: str = ""  # e.g., "has_property"
    object: str = ""  # e.g., "damage"
    value: Any = None  # e.g., 2

    # Confidence and status
    confidence: float = 0.0  # 0.0 to 1.0, computed from evidence
    status: ClaimStatus = ClaimStatus.UNCERTAIN

    # Evidence chain
    supporting_evidence: List[str] = field(default_factory=list)  # Evidence IDs
    contradicting_evidence: List[str] = field(default_factory=list)

    # Provenance
    created_at: datetime = field(default_factory=datetime.now)
    superseded_by: Optional[str] = None  # Claim ID that replaced this
    supersedes: Optional[str] = None  # Claim ID this replaced

    # For inference chains
    derived_from: List[str] = field(default_factory=list)  # Claim IDs used to derive this

    def __repr__(self):
        return f"Claim({self.subject}.{self.predicate}={self.value}, conf={self.confidence:.2f}, {self.status.value})"


# =============================================================================
# ONTOLOGY RECORD (the evidence graph)
# =============================================================================


class OntologyRecord:
    """
    The living evidence graph. Stores all claims and evidence,
    including historical/superseded ones.
    """

    def __init__(self):
        self.claims: Dict[str, Claim] = {}
        self.evidence: Dict[str, Evidence] = {}

        # Indexes for efficient lookup
        self._claims_by_subject: Dict[str, List[str]] = {}
        self._active_claims: set = set()

    def add_evidence(self, evidence: Evidence) -> str:
        """Add a piece of evidence."""
        self.evidence[evidence.id] = evidence
        return evidence.id

    def add_claim(self, claim: Claim, evidence_ids: List[str] = None) -> str:
        """Add a claim with optional initial evidence."""
        if evidence_ids:
            claim.supporting_evidence = evidence_ids

        self.claims[claim.id] = claim

        # Index by subject
        if claim.subject not in self._claims_by_subject:
            self._claims_by_subject[claim.subject] = []
        self._claims_by_subject[claim.subject].append(claim.id)

        # Recompute confidence
        self._update_confidence(claim.id)

        return claim.id

    def add_supporting_evidence(self, claim_id: str, evidence: Evidence):
        """Add evidence supporting a claim."""
        ev_id = self.add_evidence(evidence)
        claim = self.claims[claim_id]
        claim.supporting_evidence.append(ev_id)
        self._update_confidence(claim_id)

    def add_contradicting_evidence(self, claim_id: str, evidence: Evidence):
        """Add evidence contradicting a claim."""
        ev_id = self.add_evidence(evidence)
        claim = self.claims[claim_id]
        claim.contradicting_evidence.append(ev_id)
        self._update_confidence(claim_id)

    def supersede_claim(self, old_claim_id: str, new_claim: Claim, reason: str):
        """Replace an old claim with a new one."""
        old_claim = self.claims[old_claim_id]
        old_claim.status = ClaimStatus.SUPERSEDED
        old_claim.superseded_by = new_claim.id

        new_claim.supersedes = old_claim_id
        self.add_claim(new_claim)

        # Add evidence for the supersession
        ev = Evidence(
            type=EvidenceType.CONTRADICTION,
            source="supersession",
            observation=f"Superseded: {reason}",
        )
        self.add_contradicting_evidence(old_claim_id, ev)

    def _update_confidence(self, claim_id: str):
        """Recompute confidence from evidence."""
        claim = self.claims[claim_id]

        # Weight evidence by type
        type_weights = {
            EvidenceType.SOURCE_CODE: 1.0,
            EvidenceType.DOCUMENTATION: 0.9,
            EvidenceType.GAMEPLAY_OBSERVATION: 0.7,
            EvidenceType.INFERENCE: 0.6,
            EvidenceType.HYPOTHESIS: 0.3,
        }

        supporting_weight = 0.0
        for ev_id in claim.supporting_evidence:
            ev = self.evidence[ev_id]
            supporting_weight += ev.weight * type_weights.get(ev.type, 0.5)

        contradicting_weight = 0.0
        for ev_id in claim.contradicting_evidence:
            ev = self.evidence[ev_id]
            contradicting_weight += ev.weight * type_weights.get(ev.type, 0.5)

        # Confidence formula: supporting / (supporting + contradicting + 1)
        # The +1 provides a prior toward uncertainty
        total = supporting_weight + contradicting_weight + 1.0
        claim.confidence = supporting_weight / total

        # Update status
        if contradicting_weight > supporting_weight * 2:
            claim.status = ClaimStatus.DISPROVEN
        elif supporting_weight > 2.0:  # Strong evidence
            claim.status = ClaimStatus.ACTIVE
            self._active_claims.add(claim_id)
        else:
            claim.status = ClaimStatus.UNCERTAIN


# =============================================================================
# ONTOLOGY ENGINE (compiles current best knowledge)
# =============================================================================


class OntologyEngine:
    """
    Compiles the OntologyRecord into usable ontology.
    Resolves conflicts, picks highest-confidence claims,
    outputs clean ontology with confidence annotations.
    """

    def __init__(self, record: OntologyRecord):
        self.record = record

    def compile(self, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Compile current best knowledge into ontology format.
        Only includes claims above min_confidence.
        """
        ontology = {
            "compiled_at": datetime.now().isoformat(),
            "min_confidence": min_confidence,
            "entities": {},
            "actions": {},
            "dynamics": {},
            "materials": {},
            "_meta": {
                "total_claims": len(self.record.claims),
                "active_claims": len(self.record._active_claims),
                "uncertain_claims": [],
            },
        }

        for claim_id, claim in self.record.claims.items():
            if claim.status == ClaimStatus.SUPERSEDED:
                continue

            if claim.confidence < min_confidence:
                ontology["_meta"]["uncertain_claims"].append(
                    {
                        "claim": f"{claim.subject}.{claim.predicate}={claim.value}",
                        "confidence": claim.confidence,
                    }
                )
                continue

            # Route to appropriate section
            subject_parts = claim.subject.split(".")
            category = subject_parts[0] if len(subject_parts) > 1 else "entities"
            entity_name = subject_parts[-1]

            if category not in ontology:
                ontology[category] = {}
            if entity_name not in ontology[category]:
                ontology[category][entity_name] = {"_confidence": {}}

            # Add the claim value with confidence
            ontology[category][entity_name][claim.predicate] = claim.value
            ontology[category][entity_name]["_confidence"][claim.predicate] = claim.confidence

        return ontology

    def get_evidence_chain(self, claim_id: str) -> List[Dict]:
        """Get the full evidence chain for a claim."""
        claim = self.record.claims[claim_id]
        chain = []

        for ev_id in claim.supporting_evidence:
            ev = self.record.evidence[ev_id]
            chain.append(
                {
                    "type": "supporting",
                    "evidence_type": ev.type.value,
                    "source": ev.source,
                    "observation": ev.observation,
                    "weight": ev.weight,
                }
            )

        for ev_id in claim.contradicting_evidence:
            ev = self.record.evidence[ev_id]
            chain.append(
                {
                    "type": "contradicting",
                    "evidence_type": ev.type.value,
                    "source": ev.source,
                    "observation": ev.observation,
                    "weight": ev.weight,
                }
            )

        if claim.supersedes:
            chain.append(
                {
                    "type": "supersedes",
                    "previous_claim": claim.supersedes,
                }
            )

        return chain

    def find_conflicts(self) -> List[Dict]:
        """Find claims that conflict with each other."""
        conflicts = []

        for subject, claim_ids in self.record._claims_by_subject.items():
            # Group by predicate
            by_predicate = {}
            for cid in claim_ids:
                claim = self.record.claims[cid]
                if claim.status == ClaimStatus.SUPERSEDED:
                    continue
                key = claim.predicate
                if key not in by_predicate:
                    by_predicate[key] = []
                by_predicate[key].append(claim)

            # Check for conflicting values
            for predicate, claims in by_predicate.items():
                if len(claims) > 1:
                    values = set(str(c.value) for c in claims)
                    if len(values) > 1:
                        conflicts.append(
                            {
                                "subject": subject,
                                "predicate": predicate,
                                "claims": [
                                    {"id": c.id, "value": c.value, "confidence": c.confidence}
                                    for c in claims
                                ],
                            }
                        )

        return conflicts


# =============================================================================
# DEMO: Building ontology from different sources
# =============================================================================


def demo():
    print("=" * 70)
    print("ONTOLOGY RECORD DEMO")
    print("=" * 70)

    record = OntologyRecord()

    # =========================================================================
    # SCENARIO 1: High-confidence claim from source code
    # =========================================================================
    print("\n[1] Adding claim from SOURCE CODE (high confidence)")

    ev1 = Evidence(
        type=EvidenceType.SOURCE_CODE,
        source="crafter/objects.py:89",
        observation="Zombie class has self.health = 5 in __init__",
        weight=1.0,
    )

    claim1 = Claim(
        subject="entities.zombie",
        predicate="health",
        value=5,
    )

    record.add_claim(claim1, [record.add_evidence(ev1)])
    print(f"  {claim1}")

    # =========================================================================
    # SCENARIO 2: Uncertain claim from gameplay observation
    # =========================================================================
    print("\n[2] Adding claim from GAMEPLAY (lower confidence)")

    ev2 = Evidence(
        type=EvidenceType.GAMEPLAY_OBSERVATION,
        source="episode_1234_step_567",
        observation="Zombie appeared to deal 2 damage when attacking",
        weight=1.0,
    )

    claim2 = Claim(
        subject="entities.zombie",
        predicate="damage",
        value=2,
    )

    record.add_claim(claim2, [record.add_evidence(ev2)])
    print(f"  {claim2}")

    # Add more gameplay evidence
    print("\n[3] Adding MORE gameplay evidence (increases confidence)")

    for i in range(3):
        ev = Evidence(
            type=EvidenceType.GAMEPLAY_OBSERVATION,
            source=f"episode_{2000 + i}_step_{100 + i * 50}",
            observation=f"Zombie dealt 2 damage in combat (observation {i + 2})",
            weight=1.0,
        )
        record.add_supporting_evidence(claim2.id, ev)

    print(f"  {claim2}")

    # =========================================================================
    # SCENARIO 3: Source code confirms gameplay observation
    # =========================================================================
    print("\n[4] SOURCE CODE confirms gameplay observation (high confidence now)")

    ev_code = Evidence(
        type=EvidenceType.SOURCE_CODE,
        source="crafter/objects.py:92",
        observation="Zombie.damage = 2 in class definition",
        weight=1.0,
    )
    record.add_supporting_evidence(claim2.id, ev_code)
    print(f"  {claim2}")

    # =========================================================================
    # SCENARIO 4: Discovering sleep vulnerability (supersedes earlier belief)
    # =========================================================================
    print("\n[5] DISCOVERING that zombie damage changes during sleep")

    # Initial hypothesis from gameplay
    claim3_old = Claim(
        subject="entities.zombie",
        predicate="sleep_damage",
        value=2,  # Initially thought same as regular damage
    )
    ev3 = Evidence(
        type=EvidenceType.HYPOTHESIS,
        source="inference",
        observation="Assumed sleep damage equals regular damage",
        weight=0.5,
    )
    record.add_claim(claim3_old, [record.add_evidence(ev3)])
    print(f"  Initial hypothesis: {claim3_old}")

    # Later: gameplay shows different!
    ev3_contra = Evidence(
        type=EvidenceType.GAMEPLAY_OBSERVATION,
        source="episode_5000_step_200",
        observation="While sleeping, zombie dealt 7 damage (not 2!)",
        weight=1.0,
    )
    record.add_contradicting_evidence(claim3_old.id, ev3_contra)
    print(f"  After contradiction: {claim3_old}")

    # Supersede with correct claim
    claim3_new = Claim(
        subject="entities.zombie",
        predicate="sleep_damage",
        value=7,
    )
    ev3_confirm = Evidence(
        type=EvidenceType.SOURCE_CODE,
        source="crafter/objects.py:95",
        observation="Zombie.sleep_damage = 7 in source",
        weight=1.0,
    )
    record.supersede_claim(claim3_old.id, claim3_new, "Source code shows sleep_damage=7")
    record.add_supporting_evidence(claim3_new.id, ev3_confirm)
    print(f"  New claim: {claim3_new}")
    print(f"  Old claim now: {claim3_old}")

    # =========================================================================
    # COMPILE ONTOLOGY
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPILED ONTOLOGY")
    print("=" * 70)

    engine = OntologyEngine(record)
    ontology = engine.compile(min_confidence=0.5)

    print(json.dumps(ontology, indent=2, default=str))

    # =========================================================================
    # SHOW EVIDENCE CHAIN
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVIDENCE CHAIN for zombie.damage claim")
    print("=" * 70)

    chain = engine.get_evidence_chain(claim2.id)
    for item in chain:
        print(f"  {item}")

    # =========================================================================
    # CHECK FOR CONFLICTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFLICT CHECK")
    print("=" * 70)

    conflicts = engine.find_conflicts()
    if conflicts:
        for c in conflicts:
            print(f"  CONFLICT: {c['subject']}.{c['predicate']}")
            for claim in c["claims"]:
                print(f"    - {claim['value']} (confidence: {claim['confidence']:.2f})")
    else:
        print("  No conflicts found.")


if __name__ == "__main__":
    demo()
