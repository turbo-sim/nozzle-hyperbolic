"""Build the editable vector map and its standalone offline browser companion.

Run from the repository root:
    poetry run python -B docs/architecture/_scripts/build_architecture.py

The Markdown guide is maintained independently. This script reads no numerical
results and does not import production solvers or run simulations.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from architecture_viewer import render_viewer

DOCS = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1800, 2490
INK = "#142C44"
MUTED = "#5C7083"
LINE = "#D7E1EA"
PHYSICS = "#127D85"
NUMERICS = "#2F62C4"
CONTROL = "#7253A5"
FUTURE = "#A47024"
parts: list[str] = []


def raw(value):
    parts.append(value)


def text(x, y, value, size=18, color=INK, weight=400, **attrs):
    attributes = " ".join(f'{key.replace("_", "-")}="{escape(str(v))}"' for key, v in attrs.items())
    raw(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" {attributes}>{escape(value)}</text>')


def rect(x,y,w,h,fill="white",stroke=LINE,radius=18,**attrs):
    attributes=" ".join(f'{key.replace("_","-")}="{escape(str(v))}"' for key,v in attrs.items())
    raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" {attributes}/>')


def line(x1,y1,x2,y2,color=LINE,width=2,arrow=False):
    raw(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')


def lines(x,y,items,size=18,leading=27,color=MUTED):
    for index,item in enumerate(items):
        text(x,y+index*leading,item,size,color)


def pill(x,y,label,kind="default",width=None):
    colors={"default":("#FFF3CB","#775818","#E8D495"),"planned":("#FBF4E8",FUTURE,"#CEB88D"),"available":("#EAF5F0","#247557","#C3DFD1")}
    bg,fg,border=colors[kind]
    w=width or 20+len(label)*7.4
    rect(x,y,w,25,bg,border,12,stroke_dasharray="5 3" if kind=="planned" else "none")
    text(x+10,y+17,label,12.5,fg,700)


def card(topic,x,y,w,h,title,description,color,tag=None):
    card_id = f"architecture-card-{topic}"
    raw(f'<g id="{card_id}" data-topic="{topic}" role="button" tabindex="0" aria-labelledby="{card_id}-title" aria-describedby="{card_id}-description">')
    raw(f'<title id="{card_id}-title">{escape(title)}</title><desc id="{card_id}-description">{escape(description)}</desc>')
    rect(x,y,w,h)
    rect(x+1,y+18,4,h-36,color,color,2)
    text(x+23,y+36,title,21,INK,700)
    if tag:
        pill(x+w-23-(20+len(tag)*7.4),y+17,tag,"default" if tag=="DEFAULT" else "available")


def endcard():
    raw('</g>')


def band(y,h,number,title,subtitle,color=NUMERICS):
    fill={PHYSICS:"#EFF7F6",NUMERICS:"#F0F4FB",CONTROL:"#F4F1F9"}[color]
    rect(50,y,1700,h,fill,"none",24)
    text(76,y+35,f'{number}  /  {title}',17,color,700,letter_spacing="1.1")
    text(750,y+35,subtitle,16,MUTED)


def build_svg():
    parts.clear()
    raw(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="architecture-title architecture-description">
<title id="architecture-title">nozzle-hyperbolic: model-agnostic finite-volume architecture</title>
<desc id="architecture-description">A source-verified architecture map. Physical models supply EOS, geometry, closures, storage, fluxes and boundary equations to a common finite-volume engine. Numerical choices include constant or MUSCL reconstruction, characteristic FVS, implemented HLL and Rusanov alternatives, planned Roe FDS, coupled or segregated boundaries, explicit Euler and SSPRK3, backward Euler, automatic differentiation, direct linear solves, Newton policy, coarse-to-fine initialization and convergence audits. Dashed amber elements are planned or deferred.</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="#8A9CAD"/></marker></defs>
<style>text{{font-family:Arial,Helvetica,sans-serif}} [data-topic]{{cursor:pointer}} [data-topic]:focus{{outline:none}} [data-topic]:focus>rect:first-of-type{{stroke:#D59B2F;stroke-width:3}}</style>
<rect width="1800" height="{HEIGHT}" fill="#FFFFFF"/>
''')
    text(70,54,"NOZZLE-HYPERBOLIC  /  PACKAGE MAP",15,NUMERICS,700,letter_spacing="2")
    text(70,104,"Model-owned physics. Shared numerical machinery.",38,INK,700,letter_spacing="-0.8")
    text(70,139,"A fixed-mesh conservative FVM architecture · source checked 31 October 2026",18,MUTED)
    pill(1255,126,"DEFAULT",width=92)
    text(1360,144,"current default",14,MUTED)
    pill(1490,126,"PLANNED",kind="planned",width=99)
    text(1600,144,"future / deferred",14,MUTED)

    card("api",70,180,1660,115,"User configuration and public entry points",
        "Construct a physical model and fixed mesh. Choose Scheme, SteadySettings and ResidualPolicy. Bind an Operator or call the steady, grid-sequence or physical-time drivers. These are independent selections, not a model-specific numerical backend.",CONTROL)
    text(95,250,"Model + mesh + fluid / closures",19,INK,600)
    text(610,250,"Scheme · SteadySettings · ResidualPolicy",19,INK,600)
    text(1220,250,"Operator / solve_* drivers",19,INK,600)
    endcard()
    line(900,295,900,330,arrow=True)

    band(330,340,"01","MODEL-OWNED PHYSICS","Model callbacks define the equations; solver choices stay independent.",PHYSICS)
    card("models",70,390,400,253,"Physical models",
        "Supported shared models are quasi-one-dimensional Euler nozzle, annular Euler with swirl, scalar advection and Burgers, and user-written conservative models. The legacy two-fluid backend exists separately; integration into this shared contract is deferred.",PHYSICS)
    text(94,458,"Euler nozzle / duct",18,INK,700)
    text(94,481,"(p, u, h) · 3 conservative balances",15,MUTED)
    text(94,512,"Annular Euler + swirl",18,INK,700)
    text(94,535,"(p, v_m, ell, h) · ell = r v_theta",15,MUTED)
    text(94,566,"Advection · Burgers · user model",16,INK)
    pill(94,592,"TWO-FLUID PORT",kind="planned",width=161)
    text(267,609,"legacy separate",13,MUTED)
    endcard()

    card("eos",490,390,400,253,"Thermodynamics",
        "Euler evolution obtains state and thermodynamic derivatives through JAXProp get_state. Native JAX fluids pass through; configured host-only fluids use JAXProp's callback/custom-JVP adapter. The numerical engine contains no EOS assumptions. The finite-wave outlet is the deliberate perfect-gas exception in the evolution path. Separately scoped analytical verification references may assume a perfect gas.",PHYSICS)
    text(514,461,"JAXProp fluid.get_state()",20,PHYSICS,700)
    lines(514,495,["General EOS / configured fluid backend","Native JAX or host-fluid bridge","Properties, derivatives and domain checks"],16.5,28)
    rect(513,583,354,38,"#FFF8E8","#EDDBAE",9)
    text(525,607,"Finite-wave outlet: perfect-gas exception",14,"#815F21",600)
    endcard()

    card("geometry",910,390,400,253,"Fixed geometry + mesh",
        "Geometry builders preprocess stationary nozzle and annular geometries into physical cells, endpoint traces, face measures, cell volumes and integrated source geometry. Annular NURBS construction is host preprocessing; geometry is not evolved in time.",PHYSICS)
    lines(934,459,["Nozzle A(x) / constant-area duct","Annular NURBS midline + height","Physical cells + endpoint traces","Face areas · volumes · source measures","Stationary 1D geometry"],16.5,29)
    endcard()

    card("closures",1330,390,400,253,"Sources + closure laws",
        "Physical models inject wall shear, heat transfer and other sources. Shared closure helpers include prescribed wall data, Haaland/Darcy friction and retained mixture correlations. Some annular empirical laws remain placeholders. Closures can declare dependency reach and explicit source timestep restrictions.",PHYSICS)
    lines(1354,459,["Prescribed wall shear and heat","Haaland / Darcy friction","Retained mixture correlations","Model-owned integrated sources","Optional source timestep restriction"],16.5,29)
    text(1354,616,"Annular empirical laws include placeholders",13.5,MUTED)
    endcard()
    line(900,670,900,708,arrow=True)

    card("contract",70,710,1660,127,"02  /  Structural model contract",
        "A user-written model supplies square conservative storage, face fluxes/conserved states/wave bounds, optional characteristic projectors, integrated sources, boundary equations, admissibility, scales and dependency stencils. Callbacks are JAX traceable; no inheritance, EOS or fixed equation count is required. Nonconservative or rectangular systems require an explicit future contract extension.",NUMERICS)
    pill(1438,729,"MODEL AGNOSTIC",kind="available",width=265)
    text(95,782,"Storage Q(W) · face flux / wave bounds · optional eigenstructure · integrated sources",18,INK)
    text(95,812,"Boundary B(W,t) · admissibility · scales · dependency reach · initial state · optional fused face data",17,MUTED)
    endcard()
    line(900,837,900,873,arrow=True)

    band(875,370,"03","SHARED SPATIAL DISCRETIZATION","One conservative assembly for all compatible physical models.")
    card("reconstruction",70,935,400,280,"Reconstruction",
        "The shared Scheme selects constant reconstruction, MUSCL with the Van Leer limiter, or its smooth Van Leer alternative. Reconstruction uses physical coordinates and zero-slope endpoint traces, including nonuniform meshes; it has no EOS branches.",NUMERICS)
    text(95,1007,"Constant / first order",18,INK,700)
    pill(306,989,"DEFAULT",width=139)
    text(95,1045,"MUSCL · Van Leer",18,INK,700)
    text(95,1075,"Smooth MUSCL · optional C2 limiter",16.5,MUTED)
    lines(95,1122,["Coordinate-based; nonuniform meshes","Endpoint traces have zero slope"],16,28)
    text(95,1190,"Scheme.reconstruction",14,NUMERICS,600)
    endcard()
    line(470,1074,488,1074,arrow=True)

    card("fluxes",490,935,640,280,"Numerical interface flux",
        "The default characteristic FVS uses model projectors/waves, Harten regularization and conservative sonic viscosity. HLL and Rusanov approximate-Riemann fluxes are already implemented and use signal bounds without requiring an eigensystem. Roe FDS is planned and is not the same as the present FVS averaging. A conservative fourth-difference filter is optional and off by default.",NUMERICS)
    text(515,1007,"Characteristic FVS",19,INK,700)
    pill(738,989,"DEFAULT",width=139)
    text(515,1035,"Model eigenstructure · Harten regularization · sonic viscosity",15.5,MUTED)
    text(515,1073,"HLL + Rusanov",19,INK,700)
    pill(707,1055,"IMPLEMENTED",kind="available",width=163)
    text(515,1099,"Approximate-Riemann alternatives; model signal bounds",15.5,MUTED)
    rect(514,1116,590,43,"#FBF7EF","#CEB88D",9,stroke_dasharray="6 4")
    text(528,1143,"Roe FDS",18,FUTURE,700)
    pill(666,1124,"PLANNED",kind="planned",width=106)
    text(515,1190,"Optional conservative fourth-difference filter · default OFF",15,MUTED)
    endcard()
    line(1130,1074,1148,1074,arrow=True)

    card("spatial",1150,935,580,280,"Conservative residual assembly",
        "The engine combines reconstructed numerical face fluxes and physical face measures, applies the model's frame transform and integrated sources, and assembles signed cell balances. Cell storage may be nonlinear in primitive variables. Boundary constraints have no control-volume storage. Geometry is applied exactly once by its declared owner.",NUMERICS)
    lines(1175,1007,["Face flux → transported frame → face area","Flux divergence − integrated source","Nonlinear conservative storage Q(W)","Zero-storage algebraic endpoint constraints"],17,34)
    rect(1174,1143,530,48,"#EFF4FC","none",10)
    text(1190,1174,"R_i = A_right F_right − A_left F_left − S_i",17,NUMERICS,600)
    endcard()
    line(900,1245,900,1278,arrow=True)

    band(1280,280,"04","BOUNDARY LAWS AND THEIR NUMERICAL TREATMENT","The physical equations are defined once; their solution strategy is selectable.",PHYSICS)
    card("boundary-laws",70,1340,810,195,"Physical boundary conditions",
        "Nozzle reservoir inlet enforces enthalpy, entropy and acoustic compatibility. Annular inlet also prescribes angle or angular momentum, or a complete supersonic state. Outlets select acoustic/entropy or restricted perfect-gas finite-wave laws with pressure, sonic/automatic or fully outgoing regimes. Sod uses transmissive traces. Scalar models supply their own inflow/outflow equations.",PHYSICS)
    lines(95,1409,["Reservoir inlet: h0 / s0 + acoustic compatibility; annular swirl law","Outlet: acoustic / entropy (general EOS) or finite-wave (perfect gas)","Regimes: subsonic · supersonic · automatic pressure / sonic / outgoing","Sod: transmissive · scalar laws: signed inflow / extrapolated outflow"],16,29)
    endcard()
    line(880,1434,898,1434,arrow=True)
    card("boundary-treatment",900,1340,830,195,"Boundary solution strategy",
        "Backward Euler can solve cell variables and zero-storage endpoint equations together in a coupled Newton system, or eliminate fully converged nonlinear boundary roots with implicit-function derivatives. Explicit Euler and SSPRK3 require segregated nonlinear boundary solves at the required stages. Explicit coupled treatment is rejected by configuration.",NUMERICS)
    text(925,1408,"COUPLED · implicit default",17,NUMERICS,700)
    text(1322,1408,"SEGREGATED · implicit or explicit",17,NUMERICS,700)
    lines(925,1438,["Cell + trace unknowns in one system","Boundary rows have zero storage"],15.5,27)
    lines(1322,1438,["Nonlinear trace solve; implicit root AD","Explicit methods: stage completion"],15.5,27)
    text(925,1512,"Physical laws stay model-owned. Explicit + coupled is not supported.",15,MUTED)
    endcard()
    line(900,1560,900,1593,arrow=True)

    band(1595,360,"05","TIME INTEGRATION AND IMPLICIT ALGEBRA","Conservative stages and Newton solves use the same spatial operator.")
    card("time-explicit",70,1655,400,270,"Explicit integration",
        "Forward Euler and three-stage SSPRK3 advance conservative cell storage. Each required stage recovers the model's iteration variables and completes nonlinear boundary equations. Admissibility and wave/source/filter timestep restrictions remain active. Auxiliary recovery and boundary solves retain their safeguards.",NUMERICS)
    lines(95,1725,["Forward Euler","SSPRK3 · three stages"],20,34,color=INK)
    lines(95,1800,["Advance conserved cell contents","Recover primitive / model variables","Complete stage boundary equations","Auxiliary nonlinear solves safeguarded"],15.5,27)
    endcard()
    card("time-implicit",490,1655,400,270,"Implicit integration",
        "Backward Euler combines nonlinear conservative storage differences with the new-state spatial residual. Boundary constraints carry zero storage. Coupled or exactly eliminated boundary treatment enters the same implicit equations. Strict nonlinear convergence is required; failed implicit attempts reject and reduce the timestep.",NUMERICS)
    text(515,1725,"Backward Euler",21,INK,700)
    pill(707,1707,"DEFAULT",width=158)
    lines(515,1771,["Storage difference + dt × spatial R","Coupled traces or nonlinear elimination","Strict inner convergence required","Invalid / failed solve → reject + retry"],15.5,30)
    endcard()
    line(890,1785,908,1785,arrow=True)
    card("jacobian",910,1655,400,270,"Jacobian + linear solve",
        "The implicit residual is differentiated with JAX AD. Default colored JVPs use model-declared stencil reach and extract block bands directly. The direct block-banded solve is the default. Dense AD/dense solve are available for audit or global coupling. There is no Krylov solver or preconditioner option in this API.",NUMERICS)
    lines(935,1725,["Exact JAX AD · JIT / 64-bit","Colored JVPs → block bands","Declared dependency stencils","Block-banded direct solve","Dense AD / dense solve alternative","Dense-versus-colored audit"],16,29)
    endcard()
    line(1310,1785,1328,1785,arrow=True)
    card("newton",1330,1655,400,270,"Newton iteration",
        "Implicit flow Newton uses undamped full corrections by default, while retaining finite/admissible trial checks and strict update plus residual tests. Armijo damping is optional. The default correction cap is ten. Reaching the cap without convergence does not accept the timestep; a failed attempt triggers outer timestep reduction. Boundary and recovery auxiliary solves retain safeguards.",NUMERICS)
    text(1355,1725,"Full corrections",20,INK,700)
    pill(1552,1707,"DEFAULT",width=153)
    lines(1355,1766,["Optional Armijo backtracking","Cap 10 · update 1e−10 + residual 1e−12","Finite / admissible trial checks","Unconverged attempt → timestep retry"],15.5,30)
    endcard()
    line(900,1955,900,1988,arrow=True)

    band(1990,275,"06","INITIALIZATION, DRIVERS AND ACCEPTANCE","Steady convergence and physical transient integration are separate tasks.",CONTROL)
    card("initialization",70,2050,540,190,"Initialization acceleration",
        "Solve one grid or a strictly increasing coarse-to-fine hierarchy. Only successful numerical states are interpolated to finer physical grids. Each grid uses fresh residual normalization; failed warm attempts are retained and may retry from a cold initial guess. This is grid sequencing, not a defect-correction or FAS multigrid cycle.",CONTROL)
    text(95,2120,"Single grid or coarse → fine grid sequencing",18,INK,600)
    lines(95,2154,["Interpolate successful states; fresh norm per grid","Admissibility check + retained cold-retry fallback"],15.5,27)
    text(95,2217,"Initialization acceleration · not FAS multigrid",14,CONTROL,700)
    endcard()
    card("control",630,2050,540,190,"Drivers + timestep control",
        "Steady pseudo-time marching, grid-sequence orchestration and requested-time transient solves share operator kernels. Timestep information combines waves with model/source and filter restrictions. Accepted and rejected attempts are retained; CFL grows or shrinks according to steady settings. Finite/admissible states and inner convergence control acceptance.",CONTROL)
    lines(655,2120,["Steady pseudo-time / physical transient drivers","Wave + source + filter timestep restrictions","Accept → advance / grow; reject → shrink / retry","Requested output times · complete solve histories"],16,28)
    endcard()
    card("outputs",1190,2050,540,190,"Convergence + diagnostics",
        "Steady stopping evaluates original spatial residuals per equation with fixed scales. Select relative, absolute, either or both tests; defaults are relative 1e-6 or absolute 1e-9, plus independent boundary and global balance limits 1e-9. Optional physical monitors and model audits remain separate. Transients track conservative transfer budgets.",CONTROL)
    lines(1215,2120,["Original spatial residual · fixed scales per equation","Relative / absolute / either / both criteria","BC + global balances · model / monitor audits"],15.5,28)
    text(1215,2217,"Default: rel 1e−6 OR abs 1e−9; BC / global 1e−9",14,CONTROL,700)
    endcard()
    line(900,2265,900,2298,arrow=True)

    card("verification",70,2300,1660,115,"Verification and extension boundary",
        "Model-owned analytical references in models/*/analytical provide the canonical verification formulas, independently of numerical fluxes and solver assembly. Production tests and development clients audit fields, spatial residuals, conservation, branches, Jacobians and boundary treatments. Compatible models extend through callbacks. Two-fluid nonconservative structure needs a later contract review; periodic topology and moving geometry are outside the current contract.",CONTROL)
    text(95,2368,"Model-owned analytical references · models/*/analytical · tests/ and development verification clients",17,INK)
    text(95,2397,"Residual / field / conservation / branch audits. Extend through callbacks: fixed 1D, square conservative balances + endpoint traces.",16,MUTED)
    endcard()
    text(70,2461,"Arrows show configuration and data dependencies—not a literal runtime call order. Solid cards are implemented; dashed amber items are planned/deferred.",14,MUTED)
    raw('</svg>')
    return "\n".join(parts)+"\n"


def main():
    svg=build_svg()
    markdown=(DOCS/"ARCHITECTURE.md").read_bytes().decode("utf-8")
    (DOCS/"ARCHITECTURE.svg").write_text(svg,encoding="utf-8",newline="\n")
    (DOCS/"ARCHITECTURE.html").write_text(render_viewer(svg,markdown),encoding="utf-8",newline="\n")
    print("Wrote docs/architecture/ARCHITECTURE.svg and docs/architecture/ARCHITECTURE.html")


if __name__=="__main__":
    main()
