#!/usr/bin/env python3
"""Validate the generated result files from the JQSRT application sweeps for physical sanity and schema formatting."""

import os
import glob
import json
import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(TEST_DIR, "..", "experiments", "jqsrt_application", "results")
CASES_PATH = os.path.join(TEST_DIR, "..", "experiments", "jqsrt_application", "cases.json")


def test_jqsrt_application_results_validation():
    """Find and validate all generated JSON result files."""
    json_pattern = os.path.join(RESULTS_DIR, "*.json")
    result_files = glob.glob(json_pattern)

    # Exclude manifest if it exists
    result_files = [f for f in result_files if "run_manifest" not in os.path.basename(f)]

    if not result_files:
        pytest.skip("No JQSRT application results found in experiments/jqsrt_application/results. Skipping verification.")

    if os.path.exists(CASES_PATH):
        with open(CASES_PATH, "r") as f:
            cases_payload = json.load(f)
        cases = cases_payload.get("cases", cases_payload) if isinstance(cases_payload, dict) else cases_payload
        expected_case_ids = {case["case_id"] for case in cases}
        actual_case_ids = {os.path.splitext(os.path.basename(f))[0] for f in result_files}
        assert actual_case_ids == expected_case_ids, (
            f"Result files do not match cases.json. Missing={sorted(expected_case_ids - actual_case_ids)}, "
            f"Unexpected={sorted(actual_case_ids - expected_case_ids)}"
        )

    print(f"\nValidating {len(result_files)} result files:")
    for filepath in result_files:
        filename = os.path.basename(filepath)
        print(f"  Validating {filename}...")

        with open(filepath, "r") as f:
            data = json.load(f)

        # 1. Schema check (mandatory root keys)
        assert "case_id" in data, f"{filename} missing 'case_id'"
        assert "input_parameters" in data, f"{filename} missing 'input_parameters'"
        assert "environment_metadata" in data, f"{filename} missing 'environment_metadata'"
        assert "timing_metadata" in data, f"{filename} missing 'timing_metadata'"
        assert "observables" in data, f"{filename} missing 'observables'"
        assert "theta_deg" in data, f"{filename} missing 'theta_deg'"
        assert "mueller" in data, f"{filename} missing 'mueller'"

        # 2. Key components shapes
        theta = np.array(data["theta_deg"])
        assert len(theta) == 181, f"{filename} 'theta_deg' length is {len(theta)} instead of 181"
        assert np.all(np.diff(theta) > 0), f"{filename} 'theta_deg' angles are not sorted/increasing"

        # Mueller matrices checks
        mueller = data["mueller"]
        mueller_keys = ["F11", "F12_over_F11", "F22_over_F11", "F33_over_F11", "F34_over_F11", "F44_over_F11"]
        for key in mueller_keys:
            assert key in mueller, f"{filename} mueller key '{key}' missing"
            arr = np.array(mueller[key])
            assert len(arr) == 181, f"{filename} mueller '{key}' length is {len(arr)} instead of 181"
            assert np.all(np.isfinite(arr)), f"{filename} mueller '{key}' has non-finite values"

        # 3. Physical boundaries checks
        obs = data["observables"]
        assert "Qext" in obs, f"{filename} missing 'Qext'"
        assert "Qsca" in obs, f"{filename} missing 'Qsca'"
        assert "Qabs" in obs, f"{filename} missing 'Qabs'"
        assert "SSA" in obs, f"{filename} missing 'SSA'"
        assert "g" in obs, f"{filename} missing 'g'"

        Qext = obs["Qext"]
        Qsca = obs["Qsca"]
        Qabs = obs["Qabs"]
        SSA = obs["SSA"]
        g = obs["g"]

        assert Qext > 0.0, f"{filename} Qext={Qext} <= 0"
        assert Qsca >= 0.0, f"{filename} Qsca={Qsca} < 0"
        assert Qabs >= 0.0, f"{filename} Qabs={Qabs} < 0"
        
        # Energy conservation check: Qext = Qsca + Qabs
        assert abs(Qext - (Qsca + Qabs)) < 1e-6, f"{filename} energy conservation failed: Qext={Qext}, sum={Qsca+Qabs}"
        
        assert 0.0 <= SSA <= 1.0, f"{filename} SSA={SSA} out of bounds [0, 1]"
        assert -1.0 <= g <= 1.0, f"{filename} g={g} out of bounds [-1, 1]"

    print(f"All {len(result_files)} result files successfully validated!")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
