"""Saved-data loader and evidence packaging for the solver comparison page.

This module never imports a fluid, flow model or numerical driver. Existing
Lettieri inputs keep their original manifest; new comparison evidence has an
independent manifest and is added to the same portable report download.
"""

import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np


def _fields(source):
    aliases={"p":("p","pressure"),"u":("u","v","velocity"),
             "rho":("rho","d","density"),"T":("T","temperature"),
             "h":("h","enthalpy"),"h0":("h0",),"M":("M","Ma","M_euler"),
             "quality":("quality","Q","quality_mass"),"mdot":("mdot","mass_flow")}
    output={}
    for target,names in aliases.items():
        for name in names:
            if name in source:
                output[target]=list(source[name])
                break
    return output


def _coordinates(row,fields):
    value=row.get("x_m",fields.get("x_m",fields.get("x",[])))
    return list(value)


def load_comparison(study,records):
    """Normalize optional comparison evidence, keeping every saved value."""
    study=Path(study)
    path=study/"outputs/solver_comparison/campaign.json"
    if not path.exists():
        return None,[]
    data=json.loads(path.read_text(encoding="utf-8"))
    variants=data.get("variants",{})
    if not isinstance(variants,dict):
        raise ValueError("Solver comparison requires a variants mapping")
    normalized={}
    for physics,variant in variants.items():
        cases={}
        for case_id,entry in variant.get("cases",{}).items():
            entry=deepcopy(entry)
            space=entry.get("space",{})
            sf=space.get("fields",{})
            space["x_m"]=_coordinates(space,sf)
            space["fields"]=_fields(sf)
            x=np.asarray(space["x_m"],dtype=float)
            if x.size and (not np.isfinite(x).all() or np.any(np.diff(x)<=0)):
                raise ValueError(f"Nonmonotone space-marching coordinates: {physics}/{case_id}")
            for key,values in space["fields"].items():
                if len(values)!=len(x):
                    raise ValueError(f"Space field length: {physics}/{case_id}/{key}")
            supplied=entry.get("fvm",[])
            fvm=[]
            for index,source in enumerate(supplied):
                row=deepcopy(source)
                source_fields=row.get("fields",{})
                row["x_m"]=_coordinates(row,source_fields)
                row["fields"]=_fields(source_fields)
                row.setdefault("id",f"{physics}_{case_id}_N{row.get('cells','unknown')}_{index}")
                row.setdefault("stretch",1.0)
                row.setdefault("source_epoch",row.get("source",{}))
                for key,values in row["fields"].items():
                    if len(values)!=len(row["x_m"]):
                        raise ValueError(f"FVM field length: {physics}/{case_id}/{key}")
                matched=_fields(row.get("reference_fields",row.get("space_fields",{})))
                if matched:
                    for key,values in matched.items():
                        if len(values)!=len(row["x_m"]):
                            raise ValueError(f"Matched reference length: {physics}/{case_id}/{key}")
                    mode="Space marcher sampled at the FVM cell centers"
                else:
                    matched={}
                    # This is presentation interpolation only, never a flow seed.
                    # Outside the reference interval, return gaps rather than extrapolate.
                    if x.size>1:
                        for key,values in space["fields"].items():
                            matched[key]=np.interp(row["x_m"],x,np.asarray(values,dtype=float),
                                                   left=np.nan,right=np.nan).tolist()
                    mode="Space profile linearly interpolated to the FVM cell centers"
                row["reference_fields"]=matched
                row["reference_sampling"]=mode
                fvm.append(row)
            entry.update(space=space,fvm=fvm)
            cases[case_id]=entry
        normalized[physics]=dict(variant,cases=cases)
    data["variants"]=normalized
    evidence=[path]
    sensitivity=path.with_name("reference_sensitivity.json")
    if sensitivity.is_file():
        evidence.append(sensitivity)
    previous=data.get("previous_campaign")
    if previous:
        archived=study/previous["file"]
        if sha256(archived.read_bytes()).hexdigest()!=previous["sha256"]:
            raise ValueError("The preserved comparison archive changed")
        evidence.append(archived)
    manifest={p.relative_to(study).as_posix():sha256(p.read_bytes()).hexdigest() for p in evidence}
    data["input_sha256"]=manifest
    qa=study/"outputs/report_presentation"
    qa.mkdir(parents=True,exist_ok=True)
    (qa/"comparison_input_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return data,evidence


def combined_application(study,assets):
    """Embed the comparison component before the preserved legacy application."""
    assets=Path(assets)
    target=Path(study)/"outputs/report_presentation/combined_app.js"
    code=(assets/"comparison.js").read_text(encoding="utf-8")+"\n"+(assets/"app.js").read_text(encoding="utf-8")
    target.write_text(code,encoding="utf-8",newline="\n")
    return target


def attach_comparison_evidence(study,files):
    """Append new numerical evidence without changing the original 49 digests."""
    if not files:
        return
    study=Path(study)
    page=study/"REPORT.html"
    document=page.read_text(encoding="utf-8")
    pattern=r'(<a download="report_data_and_evidence\.zip" data-sha256=")[^"]+(" href="data:application/zip;base64,)([^"]+)(">)'
    match=re.search(pattern,document)
    if not match:
        raise ValueError("Portable evidence download is missing")
    archive=BytesIO(base64.b64decode(match[3]))
    with ZipFile(archive,"a",ZIP_DEFLATED) as z:
        manifest={}
        for path in files:
            path=Path(path)
            relative=path.relative_to(study).as_posix()
            manifest[relative]=sha256(path.read_bytes()).hexdigest()
            z.write(path,"original/"+relative)
        z.writestr("comparison_input_manifest.json",json.dumps(manifest,indent=2))
    raw=archive.getvalue()
    replacement=match[1]+sha256(raw).hexdigest()+match[2]+base64.b64encode(raw).decode()+match[4]
    document=document[:match.start()]+replacement+document[match.end():]
    page.write_text(document,encoding="utf-8",newline="\n")
    record=study/"outputs/report_presentation/render.json"
    saved=json.loads(record.read_text(encoding="utf-8"))
    saved.update(html_sha256=sha256(document.encode()).hexdigest(),comparison_input_sha256=manifest)
    record.write_text(json.dumps(saved,indent=2)+"\n",encoding="utf-8")
