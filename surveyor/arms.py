"""Arm names, matching the rows of the paper's summary table.

The eval drivers take these as --subgoal values and record them in every trace
they write:

    flat              LeWM CEM against the goal image, no subgoal injection
    ffjepa            a drafted subgoal at every replan boundary (FF-JEPA)
    surveyor          CEM + Surveyor: draft a block, verify, serve or redraft
    gcidm             GC-IDM, the amortized goal-reaching baseline
    gcidm+surveyor    the same accept rule with GC-IDM as the executor
    router+surveyor   c* routes the episode, then retires the drafter on arrival
    paired+surveyor   the accept rule verified in a second, frozen encoder

    oracle            true demo waypoints on a schedule (contaminated ceiling)
    voracle           true demo waypoints consumed under the accept rule
    dspark            the ported DSpark head, kept as the negative result
    regressor         deterministic drafter, kept as the negative result
    lerp              data-free interpolated block, the decomposition control
    random            random action policy, the floor
    horizon_gated     the retired encoder-distance gate

LEGACY maps the development names, which appear in the pre-registration
documents and in traces banked before the rename, onto the released ones so
recorded runs stay readable.
"""

LEGACY = {
    "baseline": "flat",
    "gdm": "ffjepa",
    "specaccept": "surveyor",
    "specgcidm": "gcidm+surveyor",
    "unified": "router+surveyor",
    "specpaired": "paired+surveyor",
}


def canonical_arm(name):
    """Released arm name for `name`, which may be a development name."""
    return LEGACY.get(name, name)
