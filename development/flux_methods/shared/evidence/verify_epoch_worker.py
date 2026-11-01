"""Run an unchanged study entry point and record actual numerical-worker origins.

Copy this orchestration-only script into the isolated workspace root. Normal
script-directory discovery selects the flat recorded package. No sys.path,
environment, source, numerical client or controller is altered.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from datetime import datetime,timezone


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path=Path(sys.argv[1]).resolve()
    proof_path=Path(sys.argv[2]).resolve()
    module=sys.argv[3]
    arguments=sys.argv[4:]
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    root=Path(manifest["isolated_root"]).resolve()
    assert Path(__file__).resolve().parent==root
    before={name:digest(root/name) for name in manifest["deployed_files"]}
    assert before==manifest["deployed_files"]
    sys.argv=[module,*arguments]
    namespace=runpy.run_module(module,run_name="__main__")
    main_path=Path(namespace["__file__"]).resolve()
    assert digest(main_path)==manifest["deployed_files"][main_path.relative_to(root).as_posix()]
    origins={}
    client_origins={}
    for name,item in list(sys.modules.items()):
        if name.startswith("nozzle_hyperbolic") and getattr(item,"__file__",None):
            path=Path(item.__file__).resolve()
            assert path.is_relative_to(root/"nozzle_hyperbolic"),(name,path)
            relative=path.relative_to(root).as_posix()
            assert digest(path)==manifest["source_sha256"]["src/"+relative]
            origins[name]=dict(path=str(path),sha256=digest(path))
        elif name.startswith("development.flux_methods.shared") and getattr(item,"__file__",None):
            path=Path(item.__file__).resolve()
            assert path.is_relative_to(root/"development/flux_methods/shared")
            assert digest(path)==manifest["deployed_files"][path.relative_to(root).as_posix()]
            client_origins[name]=dict(path=str(path),sha256=digest(path))
    after={name:digest(root/name) for name in manifest["deployed_files"]}
    assert after==before==manifest["deployed_files"]
    result=dict(category="actual_frozen_numerical_worker_origin_verification",passed=True,
         utc=datetime.now(timezone.utc).isoformat(),module=module,arguments=arguments,
         isolated_root=str(root),python_executable=sys.executable,worker_wrapper_sha256=digest(Path(__file__)),
         source_isolation_manifest_sha256=digest(manifest_path),
         before_after_deployed_hashes_identical=True,deployed_file_count=len(after),
         verified_loaded_module_count=len(origins),loaded_module_origins=origins,
         numerical_client_origins=client_origins,main_module_path=str(main_path),main_module_sha256=digest(main_path),
         import_path_modified=False,environment_modified=False,numerical_clients_modified=False,
         interpretation="Actual worker imports resolved to byte-verified flat source files. Duplicate src files, clients, resources and markers remained identical before/after this invocation.")
    proof_path.parent.mkdir(parents=True,exist_ok=True)
    proof_path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("Verified actual frozen numerical-worker origins:",len(origins))


if __name__=="__main__":
    main()
