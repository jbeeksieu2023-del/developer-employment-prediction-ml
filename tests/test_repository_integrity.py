from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_documents_and_dependency_manifest_exist():
    expected = [
        ROOT / "reports" / "final_report.pdf",
        ROOT / "slides" / "final_presentation.pdf",
        ROOT / "results" / "reported_model_results.csv",
        ROOT / "requirements.txt",
    ]

    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
