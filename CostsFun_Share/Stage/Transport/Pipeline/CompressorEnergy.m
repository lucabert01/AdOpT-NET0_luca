function [Ecomp_kJ_per_kg, Wcomp_MW] = CompressorEnergy(P2_Pa, Pcompin_Pa, m_kg_per_s, Tcomp_K, R_J_per_mol_per_K, M_kg_per_mol, rho_kg_per_m3)
%Equations (B.13)-(B.17) in Supplementary Material
%OUTPUT: Ecomp_kJ_per_kg = compressor specific energy [kJ.kg-1]
%        Wcomp_MW = compressor capacity [MW]   
%INPUT: P2_Pa = outlet pressure [Pa]
%       Pcompin_Pa = inlet pressure [Pa]
%       m_kg_per_s = CO2 mass flow [kg.s-1]
%       Tcomp_K = compression temperature [K]
%       R_J_per_mol_per_K = universal gas constant [J.mol-1.K-1]
%       M_kg_per_mol = CO2 molar mass [kg.mol-1]
%       rho_kg_per_m3 = CO2 density [kg.m-3]

PR = 2.04;
%T1 = 303.15;
kappa = 1.294;
etaIso = 0.8;
etaMech = 0.99;
etaPump = 0.75;

switch Tcomp_K
    case 303.15
        Z = 0.994799474; %capture at 30°C and 1.1 bar (0.11 MPa)
    case 288.15
        Z = 0.910912883; %recompression of onshore gas at 15°C and 1.5 MPa
end

switch P2_Pa > 3e6
    case 0 %gas
        nStage = ceil(log(P2_Pa/Pcompin_Pa)/log(PR));
        P1_Pa = P2_Pa;
        DPpump = 0; % =P2_Pa - P1_Pa;
        PR = nthroot(P1_Pa/Pcompin_Pa,nStage);
    case 1 %liquid
        nStage = floor(log(P2_Pa/Pcompin_Pa)/log(PR));
        P1_Pa = Pcompin_Pa.*PR.^nStage;
        DPpump = P2_Pa - P1_Pa;
end



% Full calculation

% p1x = Pcapture_Pa;
% Ecomp_J_per_kg_fullCalc = 0;
% for x = 1:nStage
%     Zx = interp1(TcompressorIsothermalProperties.Pressure_MPa_,TcompressorIsothermalProperties.Z,p1x./1e6,'linear');
%     kappax = interp1(TcompressorIsothermalProperties.Pressure_MPa_,TcompressorIsothermalProperties.kappa,p1x./1e6,'linear');
%     Ecomp_J_per_kg_fullCalc = Ecomp_J_per_kg_fullCalc + Zx.*R_J_per_mol_per_K.*T1.*kappax.*...
%     (pressureRatio.^((kappax-1)./kappax) - 1)./...
%     (M_kg_per_mol.*etaIso.*etaMech.*(kappax-1));
%     p1x = p1x.*2.04;
% end
% Ecomp_J_per_kg_fullCalc = Ecomp_J_per_kg_fullCalc + DPpump./(etaPump.*rho_kg_per_m3);

% Alternative

% nStage_alt = nStage + 1;
% Ecomp_J_per_kg_alt = Z.*R_J_per_mol_per_K.*T1.*nStage_alt.*kappa.*...
%     ((P2_Pa./Pcapture_Pa).^((kappa-1)./(nStage_alt.*kappa)) - 1)./...
%     (M_kg_per_mol.*etaIso.*etaMech.*(kappa-1));

% Simplified version

Ecomp_J_per_kg = Z.*R_J_per_mol_per_K.*Tcomp_K.*nStage.*kappa.*...
    (PR.^((kappa-1)./kappa) - 1)./...
    (M_kg_per_mol.*etaIso.*etaMech.*(kappa-1)) + DPpump./(etaPump.*rho_kg_per_m3);
Ecomp_kJ_per_kg = Ecomp_J_per_kg./1e3;

Wcomp_kJ_per_s = Ecomp_kJ_per_kg.*m_kg_per_s;

Wcomp_MW = Wcomp_kJ_per_s./1e3;

end