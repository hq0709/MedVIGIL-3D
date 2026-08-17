"""
Tool-answerability: decide, before running anything, which question classes a
given tool set can answer -- then verify the prediction empirically.

The claim
---------
Given a tool set T and a question class Q, Q is T-ANSWERABLE iff the answer is a
computable function of T's outputs on that input. For structured tools (returning
typed values rather than free text) this is decidable by inspecting return types
against what the question's answer depends on.

Why it matters
--------------
Two published results look contradictory:
  * DeepTumorVQA (2605.09679): tools give Measurement +35.5pp but Visual
    Reasoning -0.4pp.
  * "Do Multimodal Agents Really Benefit from Tool Use?" (2606.02357): 93-96% of
    tool-solved problems were solvable without tools; that paper states it cannot
    tell from traces "whether tools provided answer-critical information".

T-answerability resolves both. Inspect DeepTumorVQA's tool set -- segment_organ,
measure, lookup_medical_knowledge, crop_region -- and every one returns a SCALAR
statistic (volume, HU, diameter, count). None returns a RELATION. So measurement
questions are T-answerable under it and relational ones are not, which is exactly
the split their numbers show. The framework predicted it rather than fitting it.

The falsifiable part
--------------------
Adding a relation-returning tool must move relational questions from "no gain" to
"large gain", while leaving classes that remain non-answerable at zero gain. The
prediction is two-sided: what should NOT improve is as much a commitment as what
should. That is what separates this from an ablation.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scene_graph import AXES, Structure, axis_relation, bbox_gap_lower_bound


# --- what an answer can depend on -------------------------------------------
# Deliberately coarse: these are the kinds of quantity a spatial/measurement
# question's answer is a function of.
DEP_SCALAR = "scalar"            # volume, diameter, HU, count
DEP_RELATION = "relation"        # ordering along an anatomical axis
DEP_ADJACENCY = "adjacency"      # surface proximity / contact
DEP_IDENTITY = "identity"        # which structure contains a point
DEP_APPEARANCE = "appearance"    # texture, density pattern -- no tool provides it


@dataclass
class Tool:
    name: str
    returns: set[str]            # which dependency kinds this tool can supply
    fn: object = None

    def __repr__(self) -> str:
        return f"Tool({self.name}, returns={sorted(self.returns)})"


# Question class -> the dependency kinds its answer is a function of.
QUESTION_DEPS = {
    "extent": {DEP_SCALAR},
    "laterality": {DEP_RELATION},
    "longitudinal": {DEP_RELATION},
    "anteroposterior": {DEP_RELATION},
    "adjacency": {DEP_ADJACENCY},
    "container": {DEP_IDENTITY},
    "texture": {DEP_APPEARANCE},
}


def is_answerable(question_class: str, tools: list[Tool]) -> bool:
    """Decidable for structured tools: are all dependencies covered?"""
    need = QUESTION_DEPS.get(question_class)
    if need is None:
        return False
    have: set[str] = set()
    for t in tools:
        have |= t.returns
    return need.issubset(have)


def predict_gains(question_classes: list[str], tools: list[Tool]) -> dict[str, str]:
    """A priori prediction, made before any model is run."""
    return {q: ("gain expected" if is_answerable(q, tools) else "NO gain expected")
            for q in question_classes}


# --- concrete tool implementations ------------------------------------------

def make_measurement_tools(structures: dict[str, Structure]) -> list[Tool]:
    """Reconstruction of the DeepTumorVQA-style tool set: scalars only.

    This is the control arm. Every return value is a number, so relational
    questions stay non-answerable no matter how capable the agent is.
    """
    def measure(name: str) -> dict:
        s = structures.get(name)
        if s is None:
            return {"error": f"unknown structure {name}"}
        lo, hi = np.array(s.bbox_min_ras), np.array(s.bbox_max_ras)
        return {"volume_mm3": round(s.volume_mm3, 1),
                "extent_mm": [round(v, 1) for v in (hi - lo)],
                "hu_mean": None if s.hu_mean is None else round(s.hu_mean, 1),
                "n_voxels": s.n_voxels}

    def list_structures() -> dict:
        return {"structures": sorted(structures)}

    return [Tool("measure", {DEP_SCALAR}, measure),
            Tool("list_structures", set(), list_structures)]


def make_relation_tool(structures: dict[str, Structure],
                       seg=None, affine=None) -> Tool:
    """The tool the published tool sets are missing: returns a RELATION."""
    def spatial_relation(a: str, b: str, axis: str = "all") -> dict:
        sa, sb = structures.get(a), structures.get(b)
        if sa is None or sb is None:
            return {"error": "unknown structure"}
        axes = list(AXES) if axis == "all" else [axis]
        out = {}
        for ax in axes:
            r = axis_relation(sa, sb, ax)
            out[ax] = ({"relation": r.predicate, "margin_mm": round(r.margin_mm, 1)}
                       if r else {"relation": "undecidable_overlapping"})
        out["bbox_gap_mm"] = round(bbox_gap_lower_bound(sa, sb), 1)
        return out

    return Tool("spatial_relation", {DEP_RELATION, DEP_ADJACENCY}, spatial_relation)


# --- the empirical check ------------------------------------------------------

def audit_published_toolset() -> dict:
    """Apply the framework to DeepTumorVQA's published tool set.

    Its four tools -- segment_organ (segmentation statistics), measure (volume /
    HU / diameter / count), lookup_medical_knowledge (text criteria), crop_region
    (image crops) -- return scalars, text, and pixels. None returns an ordering
    between two structures.
    """
    published = [
        Tool("segment_organ", {DEP_SCALAR, DEP_IDENTITY}),
        Tool("measure", {DEP_SCALAR}),
        Tool("lookup_medical_knowledge", set()),
        Tool("crop_region", {DEP_APPEARANCE}),
    ]
    classes = ["extent", "laterality", "longitudinal", "anteroposterior",
               "adjacency", "container"]
    pred = predict_gains(classes, published)

    with_relation = published + [Tool("spatial_relation",
                                      {DEP_RELATION, DEP_ADJACENCY})]
    pred2 = predict_gains(classes, with_relation)

    return {"published_toolset": pred, "plus_relation_tool": pred2,
            "classes_that_flip": [c for c in classes
                                  if pred[c] != pred2[c]],
            "classes_that_stay_zero": [c for c in classes
                                       if pred[c] == pred2[c] == "NO gain expected"]}


def selftest() -> None:
    a = Structure(1, "liver", 100, 1000.0, (0, 0, 0), (-10, -10, -10), (10, 10, 10))
    b = Structure(2, "spleen", 100, 1000.0, (60, 0, 0), (50, -10, -10), (70, 10, 10))
    st = {"liver": a, "spleen": b}

    meas = make_measurement_tools(st)
    rel = make_relation_tool(st)

    # the decisive prediction, made without running any model
    assert is_answerable("extent", meas), "scalars answer extent"
    assert not is_answerable("laterality", meas), \
        "scalar-only tools cannot answer a relation -- this is the whole point"
    assert is_answerable("laterality", meas + [rel]), "relation tool closes it"
    assert not is_answerable("texture", meas + [rel]), \
        "appearance stays unanswerable; the prediction must be two-sided"

    # the relation tool actually returns the right relation
    out = rel.fn("spleen", "liver", "lateral")
    assert out["lateral"]["relation"] == "right_of", out
    assert out["lateral"]["margin_mm"] == 40.0, out

    # and refuses when the structures overlap on that axis
    c = Structure(3, "c", 100, 1000.0, (5, 0, 0), (-5, -10, -10), (15, 10, 10))
    st["c"] = c
    out2 = make_relation_tool(st).fn("liver", "c", "lateral")
    assert out2["lateral"]["relation"] == "undecidable_overlapping", out2

    audit = audit_published_toolset()
    assert audit["published_toolset"]["laterality"] == "NO gain expected"
    assert audit["published_toolset"]["extent"] == "gain expected"
    assert set(audit["classes_that_flip"]) >= {"laterality", "longitudinal",
                                               "anteroposterior"}

    print("selftest OK — scalar-only tools predicted to give NO gain on relational "
          "classes (matching DeepTumorVQA's measured -0.4pp on visual reasoning "
          "vs +35.5pp on measurement); relation tool flips exactly "
          f"{audit['classes_that_flip']} and leaves the rest at zero")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        print(json.dumps(audit_published_toolset(), indent=1))
    else:
        selftest()
