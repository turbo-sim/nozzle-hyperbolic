"""Read-only checks for competing Python research and test workloads.

These checks do not stop or reprioritize other applications. They identify
known development/test commands, not every possible source of desktop noise.
Checks occur outside the timed solver calls, before and after each sample block.
"""

from datetime import datetime
import json
import os
import re
import subprocess
import time
from zoneinfo import ZoneInfo


def inspect_environment():
    checked=datetime.now(ZoneInfo("Europe/Paris")).isoformat()
    try:
        if os.name=="nt":
            script=("$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new(); "
                    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='python3.exe' OR Name='pythonw.exe'\" "
                    "| Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Depth 3 -Compress")
            result=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",script],capture_output=True,text=True,encoding="utf-8-sig",timeout=20)
            if result.returncode:raise RuntimeError(result.stderr.strip())
            rows=json.loads(result.stdout.strip() or "[]")
            if isinstance(rows,dict):rows=[rows]
            processes=[dict(pid=int(r["ProcessId"]),ppid=int(r["ParentProcessId"]),command=r.get("CommandLine") or "") for r in rows]
        else:
            result=subprocess.run(["ps","-eo","pid=,ppid=,args="],capture_output=True,text=True,check=True,timeout=20)
            processes=[]
            for line in result.stdout.splitlines():
                row=line.strip().split(None,2)
                if len(row)==3 and "python" in row[2].split()[0].lower():
                    processes.append(dict(pid=int(row[0]),ppid=int(row[1]),command=row[2]))
        owned={os.getpid()}
        for _ in range(len(processes)+1):
            expanded=owned|{p["pid"] for p in processes if p["ppid"] in owned}
            if expanded==owned:break
            owned=expanded
        competing=[]
        unreadable=[]
        for process in processes:
            if process["pid"] in owned:continue
            command=process["command"].replace("\\","/")
            if not command:unreadable.append(process["pid"]);continue
            if re.search(r"development[./].*(?:scripts[./]|\.py\b)|(?:-m\s+pytest\b|pytest(?:\.exe)?\s)",command):
                competing.append(process)
        status="contended" if competing else "unverified" if unreadable else "clean"
        return dict(status=status,checked_at=checked,competing_processes=competing,unreadable_python_pids=unreadable,
                    method="read-only Python command inventory; own process tree excluded",
                    limitations="Before/after block probes detect known development/test jobs; ordinary OS activity and short intervening workloads remain possible.")
    except Exception as error:
        return dict(status="unverified",checked_at=checked,error=str(error),
                    details="Unable to inspect competing processes; timing is not certified clean.")


def wait_until_idle(*,stable_seconds=45,poll_seconds=5):
    quiet_since=None;last_notice=0
    while True:
        state=inspect_environment();now=time.monotonic()
        if state["status"]=="unverified":
            raise RuntimeError("Cannot verify timing environment: "+str(state))
        if state["status"]=="clean":
            if quiet_since is None:quiet_since=now
            if now-quiet_since>=stable_seconds:
                state["quiet_seconds"]=now-quiet_since
                return state
        else:quiet_since=None
        if now-last_notice>=30:
            print(f'Waiting for an idle timing environment: {len(state.get("competing_processes",[]))} competing jobs; quiet interval {0 if quiet_since is None else int(now-quiet_since)}/{stable_seconds}s.',flush=True)
            last_notice=now
        time.sleep(poll_seconds)
