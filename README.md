# Project02_46W38
GOAL
building the turbine Turbie & simulating Turbie's response to wind loads

ASSUMPTIONS
- No dynamic inflow, dynamic stall, or tower shadow;
- Turbie’s thrust coefficient Ct is constant for a simulation and can be calculated from the mean wind speed;
- The only aerodynamic forcing is on the blades;
- No spatial variation of turbulence.

HOW
Dynamical systems of the form y'(t)= A y + B 
Matrix A includes mass, stiffness and damping of Turbie
Matrix B includes the input forcing from the wind (Ct constant in calculating thrust force, see assumptions)

SOLVER:
- solve_ivp con RK45
- output in the same time grid of input w/s (wind speed)
- adaptive time step 
- external forcing T(t) based on linear interpolation of wind samples

OUTPUT
- time simulation of displacements
- mean & std devs of blade collective & (hub+nacelle+tower) against w/s

COMMENTS ON OUTPUTS
- effect of damping is not immediately vissible on displacements' oscillations over time
- both displacements' means and std devs show their maxima / max amplitudes at rated w/s, which is phisically correct since the goal of the controller is to maximize power output until rated w/s is reached, menaing higher loads and therefore displacements, then for w/s > rated the goal of the controller becomes keeping power constant at its rated value, which explains the decrease of the std dev for higher w/s
