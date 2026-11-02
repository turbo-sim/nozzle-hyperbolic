"""Audit the offline Lettieri report without running numerical cases."""
from pathlib import Path
from development.flux_methods.shared.audit_offline_report import audit

if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[1])
