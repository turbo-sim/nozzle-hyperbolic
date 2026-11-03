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
<desc id="architecture-description">A source-verified architecture map. Physical models supply EOS, geometry, closures, rectangular storage, fluxes, nonconservative terms, cell constraints and boundary equations to a common finite-volume engine. Numerical choices include constant or MUSCL reconstruction, Stadtke FVS, implemented HLL, Rusanov and model-owned Roe FDS alternatives, coupled or segregated boundaries, explicit Euler, SSPRK3 and classical RK4, backward Euler, automatic differentiation, direct linear solves, Newton policy, coarse-to-fine initialization and convergence audits. Model compatibility choices are explicit opt-ins, with documented verification limits.</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="#8A9CAD"/></marker></defs>
<style>text{{font-family:Arial,Helvetica,sans-serif}} [data-topic]{{cursor:pointer}} [data-topic]:focus{{outline:none}} [data-topic]:focus>rect:first-of-type{{stroke:#D59B2F;stroke-width:3}}</style>
<rect width="1800" height="{HEIGHT}" fill="#FFFFFF"/>
''')
    text(70,54,"NOZZLE-HYPERBOLIC  /  PACKAGE MAP",15,NUMERICS,700,letter_spacing="2")
    text(70,104,"Model-owned physics. Shared numerical machinery.",38,INK,700,letter_spacing="-0.8")
    text(70,139,"A constrained fixed-mesh FVM architecture · source checked 2 November 2026",18,MUTED)
    pill(1255,126,"DEFAULT",width=92)
    text(1360,144,"current default",14,MUTED)
    pill(1490,126,"MODEL POLICY",kind="available",width=125)
    text(1625,144,"explicit opt-in",14,MUTED)

    card("api",70,180,1660,115,"User configuration and public entry points",
        "Construct a physical model and fixed mesh. Choose Scheme, SteadySettings and ResidualPolicy. Bind an Operator or call the steady, grid-sequence or physical-time drivers. These are independent selections, not a model-specific numerical backend.",CONTROL)
    text(95,250,"Model + mesh + fluid / closures",19,INK,600)
    text(610,250,"Scheme · SteadySettings · ResidualPolicy",19,INK,600)
    text(1220,250,"Operator / solve_* drivers",19,INK,600)
    endcard()
    line(900,295,900,330,arrow=True)

    band(330,340,"01","MODEL-OWNED PHYSICS","Model callbacks define the equations; solver choices stay independent.",PHYSICS)
    card("models",70,390,400,253,"Physical models",
        "Shared models include nozzle Euler, annular Euler with swirl, scalar advection and Burgers, and the eight-state constrained two-fluid model. Two-fluid uses the frozen Alberto compatibility discretization with explicit limits; final physical validation remains separate. User models extend through generic contracts.",PHYSICS)
    text(94,458,"Euler nozzle / duct",18,INK,700)
    text(94,481,"(p, u, h) · 3 conservative balances",15,MUTED)
    text(94,512,"Annular Euler + swirl",18,INK,700)
    text(94,535,"(p, v_m, ell, h) · ell = r v_theta",15,MUTED)
    text(94,566,"Advection · Burgers · user model",16,INK)
    text(94,590,"Two-fluid: 8 states / 6 balances + 2 EOS",16,INK)
    pill(94,607,"COMPATIBILITY",kind="available",width=156)
    text(265,624,"explicit verification limits",13,MUTED)
    endcard()

    card("eos",490,390,400,253,"Thermodynamics",
        "Euler evolution obtains state and thermodynamic derivatives through JAXProp get_state. Native JAX fluids pass through; configured host-only fluids use JAXProp's callback/custom-JVP adapter. The numerical engine contains no EOS assumptions. The finite-wave outlet computes general-EOS PH wave curves and retains closed forms for a perfect-gas backend. Separately scoped analytical verification references may assume a perfect gas.",PHYSICS)
    text(514,461,"JAXProp fluid.get_state()",20,PHYSICS,700)
    lines(514,495,["General EOS / configured fluid backend","Native JAX or host-fluid bridge","Properties, derivatives and domain checks"],16.5,28)
    rect(513,583,354,38,"#FFF8E8","#EDDBAE",9)
    text(525,607,"Finite-wave outlet: general-EOS curves",14,"#815F21",600)
    endcard()

    card("geometry",910,390,400,253,"Fixed geometry + mesh",
        "Geometry builders preprocess stationary nozzle and annular geometries into physical cells, endpoint traces, face measures, cell volumes and integrated source geometry. Annular NURBS construction is host preprocessing; geometry is not evolved in time.",PHYSICS)
    lines(934,459,["Nozzle A(x) / constant-area duct","Annular NURBS midline + height","Physical cells + endpoint traces","Face areas · volumes · source measures","Stationary 1D geometry"],16.5,29)
    endcard()

    card("closures",1330,390,400,253,"Sources + closure laws",
        "Physical models inject wall shear, heat transfer and other sources. Shared closure helpers include prescribed wall data, Haaland/Darcy friction and retained mixture correlations. Two-fluid closures provide the frozen interfacial area, drag, heat and mass exchange laws; upstream TODO wall physics is not invented. Some annular empirical laws remain placeholders. Models declare dependency reach and explicit source timestep restrictions.",PHYSICS)
    lines(1354,459,["Prescribed wall shear and heat","Haaland / Darcy friction","Interphase mass / heat / drag / area","Model-owned integrated sources","Optional source timestep restriction"],16.5,29)
    text(1354,616,"Annular empirical laws include placeholders",13.5,MUTED)
    endcard()
    line(900,670,900,708,arrow=True)

    card("contract",70,710,1660,127,"02  /  Structural model contract",
        "A model declares V state variables, T transported balances and C cell constraints, with V=T+C. Storage can be rectangular. Models own physical fluxes, geometry/source and nonconservative terms, EOS constraints, trace equations, fixed scales and dependency stencils. Conservative models explicitly declare zero nonconservative terms.",NUMERICS)
    pill(1438,729,"MODEL AGNOSTIC",kind="available",width=265)
    text(95,782,"V states = T balances + C constraints; rectangular Q(W); physical flux / waves; integrated source and NC work",18,INK)
    text(95,812,"Boundary B(W,t) · admissibility · scales · dependency reach · initial state · optional fused face data",17,MUTED)
    endcard()
    line(900,837,900,873,arrow=True)

    band(875,370,"03","SHARED SPATIAL DISCRETIZATION","One shared balance assembly for all compatible physical models.")
    card("reconstruction",70,935,400,280,"Reconstruction",
        "Scheme selects constant reconstruction or MUSCL with Van Leer, smooth Van Leer, Minmod, Superbee or sign-preserving Van Albada. Reconstruction uses physical coordinates on uniform/nonuniform meshes and zero-slope endpoint traces. Optional model-owned representation maps recover constrained face states without EOS branches in the engine.",NUMERICS)
    text(95,1007,"Constant / first order",18,INK,700)
    pill(306,989,"DEFAULT",width=139)
    text(95,1045,"MUSCL: Van Leer / smooth Van Leer",17,INK,700)
    text(95,1075,"Minmod / Superbee / Van Albada",17,MUTED)
    lines(95,1122,["Coordinate-aware; nonuniform meshes","Model state map / EOS inverse at faces"],15.5,28)
    text(95,1190,"Scheme.reconstruction",14,NUMERICS,600)
    endcard()
    line(470,1074,488,1074,arrow=True)

    card("fluxes",490,935,640,280,"Numerical interface flux",
        "The default Stadtke FVS uses model projectors/waves, Harten regularization and conservative sonic viscosity. HLL and Rusanov use model signal bounds without requiring an eigensystem. Implemented Roe FDS contracts model-owned finite-jump eigenspaces, with acoustic Harten regularization and exact contact/shear absolute value. Euler uses PH property calls, a centered secant and C2 roundoff blend with a 4096-times-machine-epsilon audit; invalid faces are rejected without fallback. Explicit timesteps also bound interface dissipation speeds. A conservative fourth-difference filter is optional and off by default.",NUMERICS)
    text(515,1007,"Stadtke FVS",19,INK,700)
    pill(738,989,"DEFAULT",width=139)
    text(515,1035,"Model eigenstructure · Harten regularization · sonic viscosity",15.5,MUTED)
    text(515,1073,"HLL / Rusanov / Roe / Osher",18,INK,700)
    pill(909,1055,"IMPLEMENTED",kind="available",width=163)
    text(515,1099,"Model signal bounds, eigenspaces and declared state paths",15.5,MUTED)
    rect(514,1116,590,59,"#EAF5F0","#C3DFD1",9)
    text(528,1140,"Model compatibility flux",18,INK,700)
    pill(821,1124,"EXPLICIT OPT-IN",kind="available",width=178)
    text(528,1162,"Frozen physical/discrete reference; restricted flux combinations",13.5,MUTED)
    text(515,1190,"Optional conservative fourth-difference filter · default OFF",15,MUTED)
    endcard()
    line(1130,1074,1148,1074,arrow=True)

    card("spatial",1150,935,580,280,"Balance residual assembly",
        "The engine assembles flux divergence, model-owned integrated nonconservative work and sources. Compatibility corrections receive the unfiltered numerical flux before additional stabilization. Rectangular storage plus zero-storage cell constraints forms a square system; traces also have zero storage. Geometry is applied once by its declared owner.",NUMERICS)
    lines(1175,1007,["Face flux to transported frame; apply face area","Flux divergence + NC work - integrated source","Rectangular transported storage Q(W)","Zero-storage EOS cells and endpoint traces"],17,34)
    rect(1174,1143,530,48,"#EFF4FC","none",10)
    text(1190,1174,"R_i = A_R F_R - A_L F_L + NC_i - S_i",17,NUMERICS,600)
    endcard()
    line(900,1245,900,1278,arrow=True)

    band(1280,280,"04","BOUNDARY LAWS AND THEIR NUMERICAL TREATMENT","The physical equations are defined once; their solution strategy is selectable.",PHYSICS)
    card("boundary-laws",70,1340,810,195,"Physical boundary conditions",
        "Nozzle reservoir inlet enforces enthalpy, entropy and acoustic compatibility. Annular inlet also prescribes angle or angular momentum, or a complete supersonic state. Outlets select general-EOS acoustic/entropy or finite-wave laws with pressure, sonic/automatic or fully outgoing regimes. Finite-wave curves assume classical wave structure and retain perfect-gas closed forms. Sod uses transmissive traces. Scalar models supply their own inflow/outflow equations. Two-fluid inlet laws prescribe quality, slip and phase enthalpies; characteristic outlets and trace EOS constraints use the same coupled or segregated physical equations.",PHYSICS)
    lines(95,1409,["Reservoir inlet: h0 / s0 + acoustic compatibility; annular swirl law","Outlet: acoustic / entropy or finite-wave curves (both general EOS)","Regimes: subsonic / supersonic / automatic pressure / sonic / outgoing","Two-fluid: quality / slip / phase enthalpy + characteristic outlet"],16,29)
    endcard()
    line(880,1434,898,1434,arrow=True)
    card("boundary-treatment",900,1340,830,195,"Boundary solution strategy",
        "Backward Euler can solve cell variables and zero-storage endpoint equations together in a coupled Newton system, or eliminate fully converged nonlinear boundary roots with implicit-function derivatives. Explicit Euler, SSPRK3 and RK4 require segregated nonlinear boundary solves at the required stages. Explicit coupled treatment is rejected by configuration.",NUMERICS)
    text(925,1408,"COUPLED · implicit default",17,NUMERICS,700)
    text(1322,1408,"SEGREGATED · implicit or explicit",17,NUMERICS,700)
    lines(925,1438,["Cell + trace unknowns in one system","Boundary rows have zero storage"],15.5,27)
    lines(1322,1438,["Nonlinear trace solve; implicit root AD","Explicit methods: stage completion"],15.5,27)
    text(925,1512,"Physical laws stay model-owned. Explicit + coupled is not supported.",15,MUTED)
    endcard()
    line(900,1560,900,1593,arrow=True)

    band(1595,360,"05","TIME INTEGRATION AND IMPLICIT ALGEBRA","Conservative stages and Newton solves use the same spatial operator.")
    card("time-explicit",70,1655,400,270,"Explicit integration",
        "Forward Euler, SSPRK3 and classical RK4 advance transported contents. Constrained initial states and every stage recover all cell variables with EOS constraints and complete nonlinear boundary equations at their stage times. Thermodynamics, sources and fluxes are recomputed. RK4 is not SSP. Auxiliary solves retain safeguards.",NUMERICS)
    lines(95,1725,["Forward Euler","SSPRK3 / classical RK4"],19,34,color=INK)
    lines(95,1800,["Advance transported cell contents","Recover cell constraints at every stage","Recompute properties, sources and BCs","Safeguarded local solves; RK4 is not SSP"],15.5,27)
    endcard()
    card("time-implicit",490,1655,400,270,"Implicit integration",
        "Backward Euler combines nonlinear conservative storage differences with the new-state spatial residual. Cell EOS and boundary constraints carry zero storage. Coupled or exactly eliminated boundary treatment enters the same implicit equations. Strict nonlinear convergence is required; failed implicit attempts reject and reduce the timestep.",NUMERICS)
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
        "Use a single grid, a strictly increasing grid hierarchy, explicit physical-parameter stages or a portable restart. Successful source states seed transported-density interpolation, constrained content recovery and trace completion. Every stage records preparation/solve cost and all failed attempts. This is steady initialization, not conservative physical-time remapping or FAS multigrid.",CONTROL)
    text(95,2120,"Grid / physical parameter stages / restart",18,INK,600)
    lines(95,2154,["Content recovery + cell/trace constraints","Preparation costs and failed attempts retained"],15.5,27)
    text(95,2217,"Initialization acceleration · not FAS multigrid",14,CONTROL,700)
    endcard()
    card("control",630,2050,540,190,"Drivers + timestep control",
        "Steady pseudo-time marching, grid-sequence orchestration and requested-time transient solves share operator kernels. Timestep information combines waves with model/source and filter restrictions. Accepted and rejected attempts are retained; CFL grows or shrinks according to steady settings. Finite/admissible states and inner convergence control acceptance.",CONTROL)
    lines(655,2120,["Steady pseudo-time / physical transient drivers","Wave + source + filter timestep restrictions","Accept → advance / grow; reject → shrink / retry","Requested output times · complete solve histories"],16,28)
    endcard()
    card("outputs",1190,2050,540,190,"Convergence + diagnostics",
        "Every transported equation has fixed-scale residual checks. Cell constraints and trace equations have separate absolute checks; defaults are 1e-9. Relative transport reduction defaults to 1e-6. Budgets include actual NC work. Equation-changing pins report unmodified physical defects separately. Branch/reference checks and model audits remain independent.",CONTROL)
    lines(1215,2120,["Transport residual: fixed scales per equation","Independent cell EOS / trace equation checks","Actual NC budgets; unmodified pin defects"],15.5,28)
    text(1215,2217,"Default: rel 1e-6 OR abs 1e-9; algebraic 1e-9",14,CONTROL,700)
    endcard()
    line(900,2265,900,2298,arrow=True)

    card("verification",70,2300,1660,115,"Verification and extension boundary",
        "Model-owned analytical references in models/*/analytical provide the canonical verification formulas, independently of numerical fluxes and solver assembly. Production tests and development clients audit fields, spatial residuals, conservation, branches, Jacobians and boundary treatments. Compatible models extend through callbacks. Two-fluid compatibility and singular-state limits are explicit. Periodic topology, moving geometry and a new path-conservative two-fluid method remain outside this contract.",CONTROL)
    text(95,2368,"Model-owned analytical references · models/*/analytical · tests/ and development verification clients",17,INK)
    text(95,2397,"Residual / field / conservation / branch audits. Extend through callbacks: fixed 1D, transported balances + algebraic cells and endpoint traces.",16,MUTED)
    endcard()
    text(70,2461,"Arrows show configuration and data dependencies—not a literal runtime call order. Cards describe implemented capabilities; physical validation and method limits remain explicit.",14,MUTED)
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
