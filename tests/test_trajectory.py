from src.data.trajectory import Trajectory, Step, save_trajectories, load_trajectories
import tempfile, os


def make_trajectory(safe=True):
    return Trajectory(
        trajectory_id="test_001",
        task_type="delete_record",
        task_instance_id="instance_001",
        steps=[
            Step(step_idx=0, action="GET /records", observation="[{id:1, name:'Alice'}]"),
            Step(step_idx=1, action="DELETE /records/1", observation="204 No Content"),
        ],
        is_safe=safe,
        source="test",
    )


def test_to_text():
    traj = make_trajectory()
    text = traj.to_text()
    assert "[ACTION]" in text
    assert "[OBS]" in text


def test_serialization_roundtrip():
    traj = make_trajectory()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        fname = f.name
    try:
        save_trajectories([traj], fname)
        loaded = load_trajectories(fname)
        assert len(loaded) == 1
        assert loaded[0].trajectory_id == traj.trajectory_id
        assert loaded[0].steps[0].action == traj.steps[0].action
    finally:
        os.unlink(fname)


def test_from_dict_roundtrip():
    traj = make_trajectory(safe=False)
    d = traj.to_dict()
    restored = Trajectory.from_dict(d)
    assert restored.is_safe == False
    assert len(restored.steps) == 2
