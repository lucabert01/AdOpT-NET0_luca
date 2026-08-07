function [v_m_per_s] = Velocity(m_kg_per_s, IDNPS_m, rho_kg_per_m3)
%Equation (B.8) in Supplementary Material
%OUTPUT: v_m_per_s = velocity [m.s-1]
%INPUT: m_kg_per_s = CO2 mass flow [kg.s-1]
%       IDNPS_m = inner diameter of the nominal pipe size [m]
%       rho_kg_per_m3 = CO2 density [kg.m-3]

v_m_per_s = 4*m_kg_per_s./(IDNPS_m.^2.*pi.*rho_kg_per_m3);

end