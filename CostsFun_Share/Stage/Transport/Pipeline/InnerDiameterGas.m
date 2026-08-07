function [IDcalcgas_m] = InnerDiameterGas(Poutlet_Pa, Pinlet_Pa, R_J_per_mol_per_K, ...
    Tave_K, m_kg_per_s, f, L_m, M_kg_per_mol, g_m_per_s2, z_m, DataFluidProperties277K, DataFluidProperties288K)
%Equation (B.2) in Supplementary Material
%OUTPUT: IDcalcgas_m = required inner diameter for gaseous transport [m]
%INPUT: Poutlet_Pa = outlet pressure [Pa]
%       Pinlet_Pa = inlet pressure [Pa]
%       R_J_per_mol_per_K = universal gas constant [J.mol-1.K-1]
%       T_ave_K = average temperature [K]
%       m_kg_per_s= CO2 mass flow [kg.s-1]
%       f = Darcy-Weisbach friction factor [-]
%       L_m = length of the pipe [m]
%       M_kg_per_mol = molar mass of CO2 [kg.mol-1]
%       g_m_per_s2 = gravitation constant [m.s-2]
%       z_m = difference in altitude between start and end of the pipe [m]
%       DataFluidProperties277K = Table with fluid properties at 277 K
%       DataFluidProperties288K = Table with fluid properties at 288 K

Pave_Pa = AveragePressure(Poutlet_Pa, Pinlet_Pa);
Zave = CompressibilityFactor(Pave_Pa, Tave_K, DataFluidProperties277K, DataFluidProperties288K);
IDcalcgas_m = (-16.*Zave.^2.*R_J_per_mol_per_K.^2.*Tave_K.^2.*m_kg_per_s.^2.*f.*L_m./...
    (pi^2.*(M_kg_per_mol.*Zave.*Tave_K.*R_J_per_mol_per_K.*(Poutlet_Pa.^2 - Pinlet_Pa.^2) + 2.*g_m_per_s2.*Pave_Pa.^2.*M_kg_per_mol.^2.*z_m))).^(1/5);

end