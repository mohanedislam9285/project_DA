import json

from pm25_project import cli


def _config(root, locked):
    return {
        "_config_dir": str(root),
        "paths": {"results_root": "results"},
        "project": {"group_id": "TEST-00", "finalization_locked": locked},
    }


def test_run_all_refuses_locked_project_before_any_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda _: (_config(tmp_path, True), tmp_path))
    assert cli.main(["run-all"]) == 2
    assert "rewrites folds" in capsys.readouterr().err
    assert not (tmp_path / "results" / "TEST-00").exists()


def test_run_all_refuses_existing_final_evaluation_marker(tmp_path, monkeypatch, capsys):
    marker = tmp_path / "results" / "TEST-00" / "future_test_evaluation.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda _: (_config(tmp_path, False), tmp_path))
    assert cli.main(["run-all"]) == 2
    assert "rewrites folds" in capsys.readouterr().err
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "completed"


def test_run_all_refuses_unlocked_isolated_profile(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda _: (_config(tmp_path, False), tmp_path))
    assert cli.main(["run-all"]) == 2
    assert "invokes future evaluation" in capsys.readouterr().err
    assert not (tmp_path / "results" / "TEST-00").exists()
