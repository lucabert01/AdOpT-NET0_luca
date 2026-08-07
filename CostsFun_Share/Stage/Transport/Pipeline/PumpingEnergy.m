function [Epump_MJ_per_kg, Wpump_MW, varargout] = PumpingEnergy(Pinlet_MPa, Poutlet_MPa, etaPump, rho_kg_per_m3, m_kg_per_s, phase, terrain, varargin)
%Equation (B.19) & (B.20) in Supplementary Material
%OUTPUT: Epump_MJ_per_kg = pumping specific energy [MJ.kg-1]
%        Wpump_MW = pumping station capacity [MW]
% opt    E_lastpump_MJ_per_kg = pumping specific energy of the last pumping station [MJ.kg-1]
% opt    W_lastpump_MW = capacity of the last pumping station [MW]
%INPUT: Pinlet_MPa = inlet pressure [MPa]
%       Poutlet_MPa = outlet pressure [MPa]
%       etaPump = efficiency of the pumping equipment [-]
%       rho_kg_per_m3 = CO2 density [kg.m-3]
%       m_kg_per_s = CO2 mass flow [kg.s-1]
%       phase = (1) gas (2) liquid
%       terrain = (1) offshore (2) onshore
% opt   Poutlet_last_pump_MPa = outlet pressure of the last pumping station [MPa]
% opt   R_J_per_mol_per_K = universal gas constant [J.mol-1.K-1]
% opt   M_kg_per_mol = molar mass of CO2 [kg.mol-1]

switch nargin
    case 8
        Poutlet_last_pump_MPa = varargin{1}./1e6;
    case 10
        Poutlet_last_pump_MPa = varargin{1}./1e6;
        R_J_per_mol_per_K = varargin{2};
        M_kg_per_mol = varargin{3};
end

switch phase
    case 1 %gas
        switch terrain
            case 1 %offshore
                Ecomp_kJ_per_kg = 0;
                Wcomp_MW = 0;
            case 2 %onshore
                [Ecomp_kJ_per_kg, Wcomp_MW] = CompressorEnergy(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, m_kg_per_s, 15+273.15, R_J_per_mol_per_K, M_kg_per_mol, rho_kg_per_m3);
        end
        Epump_MJ_per_kg = Ecomp_kJ_per_kg./1e3;
        Wpump_MW = Wcomp_MW;

        switch nargout
            case 4
                [Ecomp_last_pump_kJ_per_kg, Wcomp_last_pump_MW] = CompressorEnergy(Poutlet_last_pump_MPa.*1e6, Poutlet_MPa.*1e6, m_kg_per_s, 15+273.15, R_J_per_mol_per_K, M_kg_per_mol, rho_kg_per_m3);
                E_lastpump_MJ_per_kg = Ecomp_last_pump_kJ_per_kg./1e3;
                W_lastpump_MW = Wcomp_last_pump_MW;
                varargout{1} = E_lastpump_MJ_per_kg;
                varargout{2} = W_lastpump_MW;
        end

    case 2 % liquid

        Epump_MJ_per_kg = (Pinlet_MPa - Poutlet_MPa)./(etaPump.*rho_kg_per_m3);
        Wpump_MW = Epump_MJ_per_kg.*m_kg_per_s;

        switch nargout
            case 4
                E_lastpump_MJ_per_kg = (Poutlet_last_pump_MPa - Poutlet_MPa)./(etaPump.*rho_kg_per_m3);
                W_lastpump_MW = E_lastpump_MJ_per_kg.*m_kg_per_s;
                varargout{1} = E_lastpump_MJ_per_kg;
                varargout{2} = W_lastpump_MW;
        end
end

end