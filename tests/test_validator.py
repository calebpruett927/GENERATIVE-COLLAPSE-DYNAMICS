from umcp.validator import RootFileValidator, get_root_validator


def test_validator_init():
    validator = RootFileValidator()
    assert isinstance(validator, RootFileValidator)
    assert hasattr(validator, "root")


def test_validate_all_runs(tmp_path):
    validator = RootFileValidator(root_dir=tmp_path)
    result = validator.validate_all()
    assert isinstance(result, dict)
    assert "status" in result


def test_get_root_validator():
    validator = get_root_validator()
    assert isinstance(validator, RootFileValidator)


def test_validate_weights_empty(tmp_path):
    validator = RootFileValidator(root_dir=tmp_path)
    weights_path = tmp_path / "weights.csv"
    weights_path.write_text("w_a,w_b\n")
    validator._validate_weights()
    assert any("empty" in e for e in validator.errors)


def test_validate_trace_bounds_empty(tmp_path):
    validator = RootFileValidator(root_dir=tmp_path)
    trace_path = tmp_path / "derived"
    trace_path.mkdir()
    trace_file = trace_path / "trace.csv"
    trace_file.write_text("c_1,c_2\n")
    validator._validate_trace_bounds()
    assert any("empty" in e for e in validator.errors)


def test_validate_invariant_identities_warning(tmp_path):
    validator = RootFileValidator(root_dir=tmp_path)
    inv_path = tmp_path / "outputs"
    inv_path.mkdir()
    inv_file = inv_path / "invariants.csv"
    inv_file.write_text("omega,F,kappa,IC\n0.1,0.9,0.0,1.0\n")
    validator._validate_invariant_identities()
    assert any("identity satisfied" in p or "slightly off" in p for p in validator.passed + validator.warnings)
