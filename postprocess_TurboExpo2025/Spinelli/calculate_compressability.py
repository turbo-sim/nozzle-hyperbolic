import CoolProp.CoolProp as cp

fluid_name = "CO2"
fluid = cp.AbstractState("REFPROP",fluid_name) # "BICUBIC&HEOS"

fluid.update(cp.PT_INPUTS, 84.74*1e5, 313.9)
a = fluid.compressibility_factor()

print(a)
