from factorlab.discovery.controls import run_controls


def test_pipeline_recovers_signal_and_does_not_promote_independent_ar_noise(tmp_path):
    result = run_controls(tmp_path / 'controlled.json')
    assert result['passed']
    assert result['comparisons']['known_signal']['ci_lower'] > 0.02
    assert result['comparisons']['independent_noise']['status'] != 'supported'
