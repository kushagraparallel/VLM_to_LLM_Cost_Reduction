from app.services.experiment_store import save_comparison, save_feedback


def test_feedback_is_saved_against_a_comparison_run(tmp_path) -> None:
    database_path = str(tmp_path / "results.sqlite3")
    result = {
        "run_id": "run-test-001",
        "user_prompt": "Summarize this image.",
        "direct_vlm": {"text": "Answer A"},
        "structured_pipeline": {"answer": "Answer B"},
    }

    save_comparison(database_path, result)
    save_feedback(
        database_path,
        run_id="run-test-001",
        preferred_path="structured_pipeline",
        direct_vlm_rating=3,
        structured_pipeline_rating=5,
        comment="The structured answer was more precise.",
    )

    import sqlite3

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT preferred_path, direct_vlm_rating, structured_pipeline_rating, comment "
            "FROM comparison_feedback WHERE run_id = ?",
            ("run-test-001",),
        ).fetchone()

    assert row == (
        "structured_pipeline",
        3,
        5,
        "The structured answer was more precise.",
    )
