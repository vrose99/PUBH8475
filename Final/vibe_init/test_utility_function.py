#!/usr/bin/env python
"""
Quick validation: compare your physionet_2019_utility() against the official scorer.

This runs several test cases before the full pipeline to catch issues early.
"""
import numpy as np
import sys

# Import both implementations
from fairness_timeseries import physionet_2019_utility
from alt_models.physionet_submission_6.src.external.evaluate_sepsis_score import compute_prediction_utility

def test_simple_case():
    """Test 1: Single patient, simple predictions."""
    print("\n" + "="*60)
    print("TEST 1: Simple case (single patient, obvious sepsis)")
    print("="*60)

    # Patient with sepsis at row 11 (SepsisLabel becomes 1)
    # Labels: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
    labels = [0]*11 + [1]*12

    # Official scorer: t_sepsis = argmax(labels) - dt_optimal = 11 - (-6) = 11 + 6 = 17
    # Your pipeline: hours_until_sepsis is NaN for non-septic rows, then [11, 10, 9, 8, ...]
    # For row 11: hours_until_sepsis = 11, dt = -(11 + 6) = -17 ✓ matches official

    # Test case: predict positive at row 11 (exactly at onset)
    predictions = [0]*11 + [1]*12

    official = compute_prediction_utility(labels, predictions)

    # For pipeline: row 11 is septic (hours=11), predict=1
    # dt = -17, dt_optimal = -6, so dt < dt_optimal
    # u = max(m1 * (-17) + b1, -0.05) where m1 = 1/6, b1 = 2
    # u = max((-1/6)*(-17) + 2, -0.05) = max(2.833, -0.05) = 2.833
    # Wait, that doesn't look right. Let me recalculate...

    # Row 11 is the FIRST septic row. Let me trace through more carefully:
    # For a row at hours_until_sepsis=11: dt = -(11+6) = -17
    # Is dt <= dt_optimal (-6)? Yes, -17 <= -6
    # m1 = 1.0 / ((-6) - (-12)) = 1.0/6 ≈ 0.1667
    # b1 = -m1 * (-12) = -0.1667 * (-12) = 2.0
    # u = max(0.1667 * (-17) + 2.0, -0.05) = max(-2.833 + 2.0, -0.05) = max(-0.833, -0.05) = -0.05

    hours_until_sepsis = np.array([np.nan]*11 + [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    y_pred_binary = np.array(predictions, dtype=int)
    your_score = physionet_2019_utility(hours_until_sepsis, y_pred_binary, patient_ids=None)

    print(f"Official PhysioNet scorer: {official:.6f}")
    print(f"Your implementation:       {your_score:.6f}")
    print(f"Match: {np.isclose(official, your_score)}")

    return np.isclose(official, your_score)

def test_early_detection():
    """Test 2: Early detection (best case)."""
    print("\n" + "="*60)
    print("TEST 2: Early detection (predict at hours_until_sepsis=6)")
    print("="*60)

    labels = [0]*11 + [1]*12
    # Predict at row 5 (hours_until_sepsis=6)
    predictions = [0]*5 + [1]*18

    official = compute_prediction_utility(labels, predictions)

    hours_until_sepsis = np.array([np.nan]*11 + [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    y_pred_binary = np.array(predictions, dtype=int)
    your_score = physionet_2019_utility(hours_until_sepsis, y_pred_binary, patient_ids=None)

    print(f"Official PhysioNet scorer: {official:.6f}")
    print(f"Your implementation:       {your_score:.6f}")
    print(f"Match: {np.isclose(official, your_score)}")

    return np.isclose(official, your_score)

def test_multiple_patients():
    """Test 3: Multiple patients (patient-level normalization)."""
    print("\n" + "="*60)
    print("TEST 3: Multiple patients (patient-level normalization)")
    print("="*60)

    # Create two separate patient segments
    # Patient 1: sepsis at row 11
    labels_p1 = [0]*11 + [1]*12
    preds_p1 = [0]*11 + [1]*12  # predict all positives after onset

    # Patient 2: sepsis at row 8 (in second batch)
    labels_p2 = [0]*8 + [1]*15
    preds_p2 = [0]*8 + [1]*15   # predict all positives after onset

    labels = labels_p1 + labels_p2
    predictions = preds_p1 + preds_p2

    official = compute_prediction_utility(labels, predictions)

    # Build hours_until_sepsis for both patients
    hours_p1 = np.array([np.nan]*11 + [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    hours_p2 = np.array([np.nan]*8 + [8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6], dtype=float)
    hours_until_sepsis = np.concatenate([hours_p1, hours_p2])

    y_pred_binary = np.array(predictions, dtype=int)
    patient_ids = np.array([1]*23 + [2]*23)

    your_score = physionet_2019_utility(hours_until_sepsis, y_pred_binary, patient_ids=patient_ids)

    print(f"Official PhysioNet scorer: {official:.6f}")
    print(f"Your implementation:       {your_score:.6f}")
    print(f"Match: {np.isclose(official, your_score)}")

    return np.isclose(official, your_score)

def test_all_zeros():
    """Test 4: All-zero predictions (inaction baseline)."""
    print("\n" + "="*60)
    print("TEST 4: All-zero predictions (inaction/baseline)")
    print("="*60)

    labels = [0]*11 + [1]*12
    predictions = [0]*23  # never predict positive

    official = compute_prediction_utility(labels, predictions)

    hours_until_sepsis = np.array([np.nan]*11 + [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    y_pred_binary = np.array(predictions, dtype=int)
    your_score = physionet_2019_utility(hours_until_sepsis, y_pred_binary, patient_ids=None)

    print(f"Official PhysioNet scorer: {official:.6f}")
    print(f"Your implementation:       {your_score:.6f}")
    print(f"Match: {np.isclose(official, your_score)}")

    return np.isclose(official, your_score)

def test_no_sepsis():
    """Test 5: Patient with no sepsis (all negatives)."""
    print("\n" + "="*60)
    print("TEST 5: Non-septic patient (no sepsis, all TN)")
    print("="*60)

    labels = [0]*23  # no sepsis
    predictions = [0]*23  # correctly predict all negatives

    official = compute_prediction_utility(labels, predictions)

    hours_until_sepsis = np.array([np.nan]*23)  # all non-septic
    y_pred_binary = np.array(predictions, dtype=int)
    your_score = physionet_2019_utility(hours_until_sepsis, y_pred_binary, patient_ids=None)

    print(f"Official PhysioNet scorer: {official:.6f}")
    print(f"Your implementation:       {your_score:.6f}")
    print(f"Match: {np.isclose(official, your_score)}")

    return np.isclose(official, your_score)

def test_false_alarms():
    """Test 6: False alarms on non-septic patient."""
    print("\n" + "="*60)
    print("TEST 6: False alarms (non-septic patient, predict positive)")
    print("="*60)

    labels = [0]*23  # no sepsis
    predictions = [1]*23  # incorrectly predict all positive (false alarms)

    official = compute_prediction_utility(labels, predictions)

    hours_until_sepsis = np.array([np.nan]*23)
    y_pred_binary = np.array(predictions, dtype=int)
    your_score = physionet_2019_utility(hours_until_sepsis, y_pred_binary, patient_ids=None)

    print(f"Official PhysioNet scorer: {official:.6f}")
    print(f"Your implementation:       {your_score:.6f}")
    print(f"Match: {np.isclose(official, your_score)}")

    return np.isclose(official, your_score)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UTILITY FUNCTION VALIDATION TEST")
    print("="*60)
    print("Comparing your physionet_2019_utility() vs official scorer")

    tests = [
        test_simple_case,
        test_early_detection,
        test_multiple_patients,
        test_all_zeros,
        test_no_sepsis,
        test_false_alarms,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your utility function is correct.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check implementation before running full pipeline.")
        sys.exit(1)
