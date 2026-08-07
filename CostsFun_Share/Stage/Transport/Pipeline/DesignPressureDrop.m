function [DeltaPdesign_Pa_per_m] = DesignPressureDrop(Pinlet_Pa, Poutlet_Pa, Npump, g_m_per_s2, rho_kg_per_m3, z_m, L_m)
%Equation (B.1) in Supplementary Material
%OUTPUT: DeltaPdesign_Pa_per_m = design pressure drop [Pa.m-1]
%INPUT: Poutlet_Pa = outlet pressure [Pa]
%       Pinlet_Pa = inlet pressure [Pa]
%       Npump = number of pumps [-]
%       g_m_per_s2 = gravity constant [m.s-2]
%       rho_kg_per_m3 = CO2 density [kg.m-3]
%       z_m = difference in altitude between start and end of the pipe [m]
%       L_m = length of the pipe [m]

DeltaPdesign_Pa_per_m = ((Pinlet_Pa - Poutlet_Pa).*(Npump + 1) + g_m_per_s2.*rho_kg_per_m3.*z_m)./L_m;

end