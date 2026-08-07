function [DeltaPact_Pa_per_m] = ActualPressureDrop(f, m_kg_per_s, rho_kg_per_m3, IDNPS_m)
%Equation (B.12) in Supplementary Material
%OUTPUT: DeltaPact_Pa_per_m = actual pressure drop [Pa.m-1]
%INPUT: f = Darcy-Weisbach friction factor [-]
%       m_kg_per_s = CO2 mass flow [kg.s-1]
%       rho_kg_per_m3 = CO2 density [kg.m-3]
%       IDNPS_m = inner diameter of the nominal pipe size [m]

DeltaPact_Pa_per_m = 8.*f.*m_kg_per_s.^2./(pi^2.*rho_kg_per_m3.*IDNPS_m.^5);

end