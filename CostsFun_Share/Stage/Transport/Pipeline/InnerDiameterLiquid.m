function [IDcalcliq_m] = InnerDiameterLiquid(f, m_kg_per_s, rho_kg_per_m3, DeltaPdesign_Pa_per_m)
%Equation (B.4) in Supplementary Material
%OUTPUT: IDcalcliq_m = required inner diameter for liquid transport [m]
%INPUT: f = Darcy-Weisbach friction factor [-]
%       m_kg_per_s = CO2 mass flow [kg.s-1]
%       rho_kg_per_m3 = CO2 density [kg.m-3]
%       DeltaPdesign_Pa_per_m = design pressure drop [Pa.m-1]

IDcalcliq_m = (8.*f.*m_kg_per_s.^2./(pi^2.*rho_kg_per_m3.*DeltaPdesign_Pa_per_m)).^(1/5);

end