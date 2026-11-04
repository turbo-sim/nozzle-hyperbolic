import ast, sys, glob, difflib
O="C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/nozzle_hyperbolic/fvm_solver"
N="C:/Users/alber/projects/nozzle-hyperbolic/src/nozzle_hyperbolic/models/two_fluid"
old_files=[f"{O}/two_fluid/matrices.py",f"{O}/two_fluid/flux.py",f"{O}/two_fluid/sources.py",f"{O}/source_two_fluid.py"]
new_files=[f"{N}/matrices.py",f"{N}/flux.py",f"{N}/sources.py"]+glob.glob(f"{N}/closures/*.py")
class S(ast.NodeTransformer):
    def _strip(self,node):
        self.generic_visit(node)
        b=node.body
        if b and isinstance(b[0],ast.Expr) and isinstance(getattr(b[0],'value',None),ast.Constant) and isinstance(b[0].value.value,str):
            node.body=b[1:] or [ast.Pass()]
        return node
    visit_FunctionDef=_strip; visit_ClassDef=_strip; visit_Module=_strip
def defs(files):
    out={}
    for f in files:
        t=S().visit(ast.parse(open(f,encoding='utf8').read()))
        for n in t.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                out[n.name]=(f.split('/')[-1],ast.unparse(n))
            elif isinstance(n,ast.Assign):
                for tg in n.targets:
                    if isinstance(tg,ast.Name): out[tg.id]=(f.split('/')[-1],ast.unparse(n))
    return out
o=defs(old_files); n=defs(new_files)
print("ONLY OLD:",sorted(set(o)-set(n)))
print("ONLY NEW:",sorted(set(n)-set(o)))
for k in sorted(set(o)&set(n)):
    if o[k][1]!=n[k][1]:
        print(f"\n##### DIFF {k}  ({o[k][0]} -> {n[k][0]})")
        for l in difflib.unified_diff(o[k][1].splitlines(),n[k][1].splitlines(),lineterm='',n=1): print(l)
    else: print("same:",k)
