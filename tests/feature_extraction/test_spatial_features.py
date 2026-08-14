"""Tests for the added spatial feature groups: inter-organelle contacts
(colocalization) and per-organelle radial distribution.

These groups are computed default-on in the FE workers with NO try/except guard,
so a real bug must surface (fail loud) rather than silently dropping the group —
these tests are the safety net that proves the groups produce the expected
columns and that malformed input raises instead of returning empty.
"""
import numpy as np
import pytest

from organelle_profiler.feature_extraction.spatial_features import (
    compute_spatial_features,
    compute_contact_features,
)
from organelle_profiler.feature_extraction.localization_features import (
    precompute_boundary_kdtrees,
    compute_localization_kdtree,
    compute_cell_level_localization_summary,
    _radial_distribution_from_df,
)

SPACING = (0.325, 0.325)


def _synthetic_cell(size=80):
    """A round cell with a central nucleus and two labeled organelles, one of
    which overlaps a second — gives a known contact and a known radial layout."""
    yy, xx = np.mgrid[0:size, 0:size]
    cell = np.zeros((size, size), int)
    cell[(yy - size // 2) ** 2 + (xx - size // 2) ** 2 < (size // 2 - 2) ** 2] = 1
    nucleus = np.zeros((size, size), int)
    nucleus[(yy - size // 2) ** 2 + (xx - size // 2) ** 2 < 8 ** 2] = 1
    orgA = np.zeros((size, size), int)
    orgA[20:26, 20:26] = 1          # perinuclear-ish blob
    orgA[54:60, 54:60] = 2          # peripheral blob
    orgB = np.zeros((size, size), int)
    orgB[23:29, 23:29] = 1          # overlaps orgA label 1
    orgB[10:14, 40:44] = 2
    return cell, nucleus, orgA, orgB


# --------------------------------------------------------------------------
# Contacts (colocalization)
# --------------------------------------------------------------------------

def test_contacts_detects_overlap_with_expected_keys():
    cell, nuc, A, B = _synthetic_cell()
    feats = compute_contact_features({"orgA": A, "orgB": B}, SPACING)
    key = "contact_orgA__orgB"
    assert f"{key}_overlap_area" in feats
    assert f"{key}_overlap_frac_a" in feats
    assert f"{key}_overlap_frac_b" in feats
    assert f"{key}_n_contacts" in feats
    assert feats[f"{key}_overlap_area"] > 0          # planted overlap exists
    assert feats[f"{key}_n_contacts"] >= 1
    assert 0 < feats[f"{key}_overlap_frac_a"] <= 1


def test_contacts_pixel_area_uses_spacing():
    # 3x3 overlap block -> 9 px -> 9 * 0.325**2 um^2
    a = np.zeros((20, 20), int); a[5:8, 5:8] = 1
    b = np.zeros((20, 20), int); b[5:8, 5:8] = 1
    feats = compute_contact_features({"a": a, "b": b}, SPACING)
    assert feats["contact_a__b_overlap_area"] == pytest.approx(9 * 0.325 ** 2)
    assert feats["contact_a__b_overlap_frac_a"] == pytest.approx(1.0)


def test_contacts_empty_for_single_organelle():
    a = np.zeros((20, 20), int); a[5:8, 5:8] = 1
    assert compute_contact_features({"a": a}, SPACING) == {}


def test_contacts_excludes_whole_cell_compartment():
    # cell_membrane is a compartment, not an organelle -> not a contact partner
    a = np.zeros((20, 20), int); a[5:8, 5:8] = 1
    cell = np.ones((20, 20), int)
    feats = compute_contact_features({"a": a, "cell_membrane": cell}, SPACING)
    assert feats == {}


def test_contacts_skip_redundant_cross_channel():
    # focus3d_* and phase2d_* are the same structures in two phase reconstructions:
    # cross-channel overlaps are registration artifacts, not biological contacts.
    # All three masks are placed to overlap so the ONLY reason a pair is absent is
    # the cross-channel filter (not lack of overlap).
    blob = np.zeros((20, 20), int); blob[5:12, 5:12] = 1
    masks = {
        "focus3d_tubular": blob.copy(),
        "focus3d_vesicular": blob.copy(),
        "phase2d_tubular": blob.copy(),
        "nuclei": blob.copy(),
    }
    feats = compute_contact_features(masks, SPACING)
    pairs = {k[len("contact_"):].rsplit("_overlap_area", 1)[0]
             for k in feats if k.endswith("_overlap_area")}
    # dropped: focus3d <-> phase2d (same structure, two channels)
    assert "focus3d_tubular__phase2d_tubular" not in pairs
    # kept: within-channel morphology split
    assert "focus3d_tubular__focus3d_vesicular" in pairs
    # kept: distinct-marker contacts (nuclei touches every phase group)
    assert "focus3d_tubular__nuclei" in pairs
    assert "nuclei__phase2d_tubular" in pairs


def test_contacts_no_overlap_is_sparse():
    a = np.zeros((20, 20), int); a[2:4, 2:4] = 1
    b = np.zeros((20, 20), int); b[15:17, 15:17] = 1
    # non-touching pair -> no contact_ keys emitted (sparse output)
    assert compute_contact_features({"a": a, "b": b}, SPACING) == {}


def test_compute_spatial_features_matches_contacts():
    cell, nuc, A, B = _synthetic_cell()
    assert compute_spatial_features({"orgA": A, "orgB": B}, SPACING) == \
        compute_contact_features({"orgA": A, "orgB": B}, SPACING)


# --------------------------------------------------------------------------
# Radial distribution (emitted by the localization summary)
# --------------------------------------------------------------------------

def test_radial_columns_present_and_normalized():
    cell, nuc, A, _ = _synthetic_cell()
    tc = precompute_boundary_kdtrees(cell.astype(np.uint8), nuc.astype(np.uint8), SPACING)
    df = compute_localization_kdtree(A, tc, SPACING)
    assert not df.empty
    summary = compute_cell_level_localization_summary(df, "orgA")
    bins = [k for k in summary if k.startswith("orgA_radial_frac_bin")]
    assert len(bins) == 5, f"expected 5 radial shells, got {bins}"
    assert sum(summary[b] for b in bins) == pytest.approx(1.0)  # fractions sum to 1
    assert "orgA_radial_anisotropy" in summary
    # localization aggregates still present (radial didn't displace them)
    assert "orgA_normalized_radial_position_mean" in summary


def test_radial_absent_without_nucleus():
    # No nucleus -> no normalized_radial_position -> radial returns {} (no crash)
    cell, _, A, _ = _synthetic_cell()
    tc = precompute_boundary_kdtrees(cell.astype(np.uint8), None, SPACING)
    df = compute_localization_kdtree(A, tc, SPACING)
    assert _radial_distribution_from_df(df, "orgA") == {}


# --------------------------------------------------------------------------
# Fail-loud: malformed input must raise, not silently return empty
# --------------------------------------------------------------------------

def test_contacts_raise_on_mismatched_shapes():
    a = np.zeros((20, 20), int); a[5:8, 5:8] = 1
    b = np.zeros((30, 30), int); b[5:8, 5:8] = 1   # different shape
    with pytest.raises(ValueError):
        compute_contact_features({"a": a, "b": b}, SPACING)
