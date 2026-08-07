function [Re] = ReynoldsNumber(rho_kg_per_m3, IDNPS_m, v_m_per_s, mu_Pas)
%Equation (B.9) in Supplementary Material
%OUTPUT: Re = Reynolds number [-]
%INPUT: rho_kg_per_m3 = CO2 density [kg.m-3]
%       IDNPS_m = inner diameter of the nominal pipe size [m]
%       v_m_per_s = velocity [m.s-1]
%       mu_Pas = CO2 viscosity [Pa.s]

Re = rho_kg_per_m3.*IDNPS_m.*v_m_per_s./mu_Pas;

end