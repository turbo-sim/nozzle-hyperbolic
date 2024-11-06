import CoolProp.CoolProp as CP

# Set the fluid name and desired state variables
fluid = "MDM"  # MDM is the identifier for Octamethyltrisiloxane in CoolProp
temperature = 300.09  # Temperature in Kelvin
pressure = 10e5  # Pressure in Pa (1 atm)


try:
    # Create an AbstractState instance to handle fluid properties
    state = CP.AbstractState("HEOS", fluid)
    
    # Set the state based on temperature and pressure
    state.update(CP.PT_INPUTS, pressure, temperature)
    
    # Retrieve and print some properties
    density = state.rhomass()               # Density in kg/m³
    specific_enthalpy = state.hmass()        # Specific enthalpy in J/kg
    specific_entropy = state.smass()         # Specific entropy in J/kg/K
    specific_heat_capacity = state.cpmass()  # Specific heat capacity at constant pressure in J/kg/K
    
    # Display the properties
    print(f"Properties of {fluid} at T = {temperature} K and P = {pressure} Pa:")
    print(f"Density: {density:.2f} kg/m³")
    print(f"Specific Enthalpy: {specific_enthalpy:.2f} J/kg")
    print(f"Specific Entropy: {specific_entropy:.2f} J/kg/K")
    print(f"Specific Heat Capacity (Cp): {specific_heat_capacity:.2f} J/kg/K")

    # Attempt to retrieve viscosity
    try:
        viscosity = state.viscosity()
        print(f"Viscosity: {viscosity:.5f} Pa.s")
    except ValueError:
        print("Viscosity data is not available for this fluid.")
        
    # Attempt to retrieve thermal conductivity
    try:
        thermal_conductivity = state.conductivity()
        print(f"Thermal Conductivity: {thermal_conductivity:.5f} W/m/K")
    except ValueError:
        print("Thermal conductivity data is not available for this fluid.")

except ValueError as e:
    print(f"Error: {e}")


