"""Archive immutable solves and derive explicitly labeled configuration views.

Changing a configuration representation does not execute a new numerical solve.
Original bytes remain in the archive; derived rows bind to their SHA-256 and
retain every non-configuration numerical value exactly.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import yaml
from .common import save,digest,identity,study_path,utc


CONFIGURATION_FIELDS={"id","variant","method","flux","scheme","schema_version","record_type",
                      "quadrature_points","integration","custom_jump_correction","custom_endpoint_envelope","migration"}


def numerical_payload(row):
    """All native state/history/work/error/reference data, without API labels."""
    return {key:value for key,value in row.items() if key not in CONFIGURATION_FIELDS}


def normalized_scheme(original,options):
    result=dict(original)
    result.pop("custom_jump_correction",None)
    result.pop("custom_endpoint_envelope",None)
    result.update(flux="osher",**options)
    return result


class OriginArchive:
    def __init__(self,campaign):
        self.campaign=Path(campaign)
        self.origin=json.loads((self.campaign/"migration_origin.json").read_text(encoding="utf-8"))
        self.path=(self.campaign/self.origin["archive"]).resolve()
        if digest(self.path)!=self.origin["archive_sha256"]:
            raise ValueError("Original evidence archive SHA differs")
        self.archive=zipfile.ZipFile(self.path)
        self.index=json.loads(self.archive.read("member_manifest.json"))
        self.by_sha={row["sha256"]:name for name,row in self.index.items() if name.startswith("rows/")}

    def original(self,row):
        sha=row["migration"]["original_row_sha256"]
        if sha not in self.by_sha:
            raise ValueError("Derived row has no original archive member")
        blob=self.archive.read(self.by_sha[sha])
        if hashlib.sha256(blob).hexdigest()!=sha:
            raise ValueError("Original row SHA differs")
        return json.loads(blob)

    def verify(self,row):
        original=self.original(row)
        options=self.origin["configurations"][row["variant"]]
        return dict(
            recorded_source_matches_origin=row["source_sha256"]==original["source_sha256"]
                and row["implementation_sha256"]==original["implementation_sha256"],
            numeric_payload_matches_origin=numerical_payload(row)==numerical_payload(original)
                and identity(numerical_payload(row))==row["migration"]["numerical_payload_sha256"],
            configuration_matches_origin=row["scheme"]==normalized_scheme(original["scheme"],options)
                and row["flux"]==row["method"]=="osher"
                and row["quadrature_points"]==options["quadrature_points"]
                and row["integration"]==options["integration"],
        )

    def close(self):
        self.archive.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",default="osher",choices=("osher",))
    args=parser.parse_args()
    study=study_path(args.study)
    campaign=study/"outputs/verification"
    profile_path=study/"verification_profile.yaml"
    profile=yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    summary=json.loads((campaign/"summary.json").read_text(encoding="utf-8"))
    settings=json.loads((campaign/"settings.json").read_text(encoding="utf-8"))
    if summary.get("record_type")=="api_migration_view":
        raise ValueError("This campaign is already a derived configuration view")
    if len(settings["profile"]["variants"])!=len(profile["variants"]):
        raise ValueError("Configuration populations differ")
    mapping=dict(zip(settings["profile"]["variants"],profile["variants"],strict=True))
    archived=study/"outputs/archives"
    archived.mkdir(parents=True,exist_ok=True)
    archive_path=archived/"recorded_execution.zip"
    old_directory=archived/"recorded_execution"
    if archive_path.exists() or old_directory.exists():
        raise ValueError("Never overwrite an existing historical archive")
    files={p.relative_to(campaign).as_posix():p for p in campaign.rglob("*") if p.is_file()}
    manifest={name:dict(bytes=p.stat().st_size,sha256=digest(p)) for name,p in files.items()}
    with zipfile.ZipFile(archive_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for name,p in files.items():
            zipped.write(p,name)
        zipped.writestr("member_manifest.json",json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(archive_path) as zipped:
        for name,expected in manifest.items():
            blob=zipped.read(name)
            if len(blob)!=expected["bytes"] or hashlib.sha256(blob).hexdigest()!=expected["sha256"]:
                raise ValueError("Archive member verification failed")
    # Both resolved targets are explicitly inside this study; this is a reversible
    # relocation of preserved generated evidence, never a recursive deletion.
    if not campaign.resolve().is_relative_to(study.resolve()) or not old_directory.resolve().is_relative_to(study.resolve()):
        raise ValueError("Evidence relocation escaped the study directory")
    campaign.rename(old_directory)
    campaign.mkdir()
    archive_sha=digest(archive_path)
    origin=dict(schema_version=2,study=args.study,record_type="api_migration_origin",created_utc=utc(),
         archive="../archives/recorded_execution.zip",archive_sha256=archive_sha,
         original_summary_sha256=manifest["summary.json"]["sha256"],
         original_settings_sha256=manifest["settings.json"]["sha256"],
         archive_members_verified=len(manifest),configurations=profile["configurations"],
         recorded_execution_source_sha256=settings["source_sha256"],
         recorded_execution_clients_sha256=settings["implementation_sha256"],
         case_sha256=settings["case_sha256"],new_native_solves=0,
         interpretation="Only method/configuration representation changed. Numerical fields, states, histories, native metrics and errors are immutable derived values; current-API saved-state evaluation is separate.")
    save(campaign/"migration_origin.json",origin)
    records=[]
    for entry in summary["records"]:
        original=json.loads((old_directory/entry["file"]).read_text(encoding="utf-8"))
        if digest(old_directory/entry["file"])!=entry["sha256"]:
            raise ValueError("Original summary/row SHA mismatch")
        row=dict(original)
        variant=mapping[original["variant"]]
        options=profile["configurations"][variant]
        row_id=f'{row["case"].replace("/","__")}__n{row["cells"]}__{row["reconstruction"]}__{variant}__a{row.get("alignment",0):g}'
        row.update(id=row_id,variant=variant,method="osher",flux="osher",schema_version=2,
             record_type="api_migration_view",**options,scheme=normalized_scheme(original["scheme"],options))
        row.pop("custom_jump_correction",None)
        row.pop("custom_endpoint_envelope",None)
        row["migration"]=dict(original_row_sha256=entry["sha256"],numerical_payload_sha256=identity(numerical_payload(original)),
                              archive_sha256=archive_sha,new_native_solve=False)
        target=campaign/"rows"/(row_id+".json")
        save(target,row)
        records.append(dict(id=row_id,file=target.relative_to(campaign).as_posix(),sha256=digest(target),
             case=row["case"],cells=row["cells"],reconstruction=row["reconstruction"],variant=variant,
             success=row["success"],reason=row["reason"]))
    with_source=dict(schema_version=2,study=args.study,record_type="api_migration_view",
         profile=profile,profile_sha256=digest(profile_path),source_sha256=settings["source_sha256"],
         implementation_sha256=settings["implementation_sha256"],case_sha256=settings["case_sha256"],alignment=settings["alignment"],
         migration_origin_sha256=digest(campaign/"migration_origin.json"),new_native_solves=0)
    save(campaign/"settings.json",with_source)
    save(campaign/"summary.json",dict(schema_version=2,study=args.study,record_type="api_migration_view",records=records,
         requested=[dict(case=c,cells=n,reconstruction=r,variant=v) for c in profile["cases"] for n in profile["grids"] for r in profile["reconstructions"] for v in profile["variants"]],
         requested_count=len(summary["requested"]),completed_count=len(records),completed=summary["completed"],final=summary["final"],
         source_before=summary["source_before"],source_after=summary["source_after"],source_unchanged=summary["source_unchanged"],
         case_before=summary["case_before"],case_after=summary["case_after"],cases_unchanged=summary["cases_unchanged"],
         alignment=summary.get("alignment",0),new_native_solves=0,migration_origin_sha256=digest(campaign/"migration_origin.json"),updated_utc=utc()))
    verifier=OriginArchive(campaign)
    checks=[verifier.verify(json.loads((campaign/r["file"]).read_text(encoding="utf-8"))) for r in records]
    verifier.close()
    if not all(all(r.values()) for r in checks):
        raise ValueError("Derived numerical data differs from original")
    save(campaign/"migration_audit.json",dict(passed=True,records_verified=len(records),archive_sha256=archive_sha,
         original_numerical_payloads_unchanged=True,new_native_solves=0,schema_version=2))
    print("Archived and verified",len(records),"unchanged numerical payloads; new configurable API view created")


if __name__=="__main__":
    main()
