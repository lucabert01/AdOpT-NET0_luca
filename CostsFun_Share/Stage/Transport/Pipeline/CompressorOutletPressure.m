function [Poutlet_adapted_gas_Pa, Poutlet_adapted_gas_Pa_wrong] = CompressorOutletPressure(R_J_per_mol_per_K, ...
    Tave_K, m_kg_per_s, f, L_m, g_m_per_s2, Poutlet_Pa, Pinlet_Pa, M_kg_per_mol, ...
    z_m, IDNPS_m, DataFluidProperties277K, DataFluidProperties288K)
%Equation (B.18) in Supplementary Material
%OUTPUT: Poutlet_adapted_gas_Pa = actual outlet pressure of the compressor for gas transport
%INPUT: R_J_per_mol_per_K = universal gas constant [J.mol-1.K-1]
%       T_ave_K = average temperature [K]
%       m_kg_per_s= CO2 mass flow [kg.s-1]
%       f = Darcy-Weisbach friction factor [-]
%       L_m = length of the pipe [m]
%       g_m_per_s2 = gravity constant [m.s-2]
%       Poutlet_Pa = outlet pressure [Pa]
%       Pinlet_Pa = inlet pressure [Pa]
%       z_m = difference in altitude between start and end of the pipe [m]
%       IDNPS_m = inner diameter of the nominal pipe size [m]
%       DataFluidProperties277K = Table with fluid properties at 277 K
%       DataFluidProperties288K = Table with fluid properties at 288 K

Pave_Pa = AveragePressure(Poutlet_Pa, Pinlet_Pa);
Zave = CompressibilityFactor(Pave_Pa, Tave_K, DataFluidProperties277K, DataFluidProperties288K);

Poutlet_adapted_gas_Pa = (16.*Zave.*R_J_per_mol_per_K.*Tave_K.*m_kg_per_s.^2.*f.*L_m./...
    (pi.^2.*IDNPS_m.^5.*M_kg_per_mol) + ...
    2.*g_m_per_s2.*Pave_Pa.^2.*M_kg_per_mol.*z_m./(Zave.*Tave_K.*R_J_per_mol_per_K) + ...
    Poutlet_Pa.^2).^0.5;

Poutlet_adapted_gas_Pa_wrong = (16.*Zave.^2.*R_J_per_mol_per_K.*Tave_K.^2.*m_kg_per_s.*f.*L_m./...
    (pi.^2.*IDNPS_m.*M_kg_per_mol.*Zave.*Tave_K.*R_J_per_mol_per_K) + ...
    2.*g_m_per_s2.*Pave_Pa.^2.*M_kg_per_mol.*z_m./(Zave.*Tave_K.*R_J_per_mol_per_K) + ...
    Pinlet_Pa.^2).^0.5;

end