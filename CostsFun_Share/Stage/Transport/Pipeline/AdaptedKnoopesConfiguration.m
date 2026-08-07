function [configuration, config] = AdaptedKnoopesConfiguration(m_kg_per_s, timeFrame, ...
    terrain, z_m, L_km, r, F, SteelFactor, Clab_EUR_per_m2, CROW_EUR_per_m, ...
    C_el_EUR_per_MWh, ODNPS_possible)
%Calculates the configurations for pipelines
%INPUT: m_kg_per_s = matrix containing the mass flow of CO2 [kg.s-1]
%       timeFrame = time horizon that is decisive for the steel types available: 1 = short-term, 2 = mid-term, 3 = long-term
%       terrain = terrain type: 1 = offshore, 2 = onshore
%       z_m = elevation difference between two nodes [m]
%       L_km = matrix with distances between two nodes [km]
%       r = discount rate
%       F = design factor for the pipeline
%       SteelFactor = steel factor
%       Clab_EUR_per_m2 = labour cost [EUR.m-2]
%       CROW_EUR_per_m = right-of-way cost [EUR.m-1]
%       C_el_EUR_per_MWh = electricity cost [EUR.MWh-1]
%       ODNPS_possible = possible outer NPS diameters for the pipe [m]
%OUTPUT:    configuration =
%           config = 

%% Data

DataFluidProperties277K = readtable('FluidProperties/CO2IsothermalProperties.xlsx','Sheet','277K');
DataFluidProperties288K = readtable('FluidProperties/CO2IsothermalProperties.xlsx','Sheet','288K');

epsilon_m = 50e-6; %[m]
rho_kg_per_m3 = [DataFluidProperties277K.Density_kg_m3_(find(DataFluidProperties277K.Pressure_MPa_ == 1.5)), ...
    DataFluidProperties288K.Density_kg_m3_(find(DataFluidProperties288K.Pressure_MPa_ == 1.5)); ... %1.5 MPa at 4°C and 15°C
    DataFluidProperties277K.Density_kg_m3_(find(DataFluidProperties277K.Pressure_MPa_ == 8.0)), ...
    DataFluidProperties288K.Density_kg_m3_(find(DataFluidProperties288K.Pressure_MPa_ == 8.0))]; %8.0 MPa at 4°C and 15°C
mu_Pas = [DataFluidProperties277K.Viscosity_Pa_s_(find(DataFluidProperties277K.Pressure_MPa_ == 1.5)), ...
    DataFluidProperties288K.Viscosity_Pa_s_(find(DataFluidProperties288K.Pressure_MPa_ == 1.5)); ... %1.5 MPa at 4°C and 15°C
    DataFluidProperties277K.Viscosity_Pa_s_(find(DataFluidProperties277K.Pressure_MPa_ == 8.0)), ...
    DataFluidProperties288K.Viscosity_Pa_s_(find(DataFluidProperties288K.Pressure_MPa_ == 8.0))]; %8.0 MPa at 4°C and 15°C
g_m_per_s2 = 9.81;
R_J_per_mol_per_K = 8.31;
T_degC = [4 15]; %offshore and onshore temperature
M_kg_per_mol = 0.04401;
steelTimeFrame = [{1:5}, {1:7}, {1:8}];
S_MPa = [275 355 460 500 550 620 690 890]; %X42 X52 X65 X70 X80 X90 X100 X120
E = 1;
CA_m = 0.001;
dtRatio = [0.025 0.01]; %offshore and onshore ratio t/OD
rhoSteel_kg_per_m3 = 7900;
Csteel_EUR_per_kg = [1.17 1.20 1.37 1.49 1.51 1.53 1.54 1.79]; %X42 X52 X65 X70 X80 X90 X100 X120
SteelGrade = {'X42', 'X52', 'X65', 'X70', 'X80', 'X90', 'X100', 'X120'};
mu_misc = 0.25;
muOMpipe = 1.5/100;
muOMpumpcomp = 4/100;
vRange = [5 20; ...%gas
    0.5 6]; %dense
etaPump = 0.75;
H_h = 8760; %[h/y]
z_pipe = 50;
z_pumpcomp = 25;
PinletMAX_MPa = [3 3; ... %offshore and onshore gas
    35 24]; %offshore and onshore liquid
Pcapture_Pa = 0.11e6;
NpumpMAX = [0 L_km./40]; %offshore and onshore

%% Code

%% Initiating

disp('Initiating...')

ODNPS_terrain = ODNPS_possible{terrain(1)};
Npump_max = NpumpMAX(terrain(1));

configuration = struct;
config = struct;

%% Gas

disp('Gaseous configuration')

disp('Initiating...')
phase = 1;

Pinlet_MPa = 1.6; %[MPa]
Poutlet_MPa = 1.5; %[MPa]
IDcalc_m = 0.5; %[m]
LClow = 1e6;
Npump = 0;

v_m_per_s = Velocity(m_kg_per_s, IDcalc_m, rho_kg_per_m3(phase,terrain(1)));
Re = ReynoldsNumber(rho_kg_per_m3(phase,terrain(1)), IDcalc_m, v_m_per_s, mu_Pas(phase,terrain(1)));
f = DarcyWeisbach(epsilon_m, IDcalc_m, Re);

% DeltaPact_Pa_per_m = ActualPressureDrop(f, m_kg_per_s, rho_kg_per_m3(1,terrain(1)), IDcalc_m);
idxPinlet = 0;
disp('Iterating through the possible pressure levels and number of pumps')
while Pinlet_MPa <= PinletMAX_MPa(phase,terrain(1))
    Npump = 0;
    while Npump <= Npump_max

        % Lpump_m = MaximumDistanceBetweenPumpingStations(gas.Pinlet_MPa.*1e6, gas.Poutlet_MPa.*1e6, DeltaPact_Pa_per_m);
        % Npump = NumberPumpingStations(L_km.*1e3, Lpump_m);
        DeltaPdesign_Pa_per_m = DesignPressureDrop(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, Npump, g_m_per_s2, rho_kg_per_m3(phase,terrain(1)), z_m, L_km.*1e3);
        Lpump_m = MaximumDistanceBetweenPumpingStations(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, DeltaPdesign_Pa_per_m);
%         IDcalc_m = InnerDiameterGas(Poutlet_MPa.*1e6, Pinlet_MPa.*1e6, R_J_per_mol_per_K, ...
%             T_degC(terrain(1))+273.15, m_kg_per_s, f, L_km.*1e3, M_kg_per_mol, g_m_per_s2, ...
%             z_m, DataFluidProperties277K, DataFluidProperties288K);
        IDcalc_m = InnerDiameterGas(Poutlet_MPa.*1e6, Pinlet_MPa.*1e6, R_J_per_mol_per_K, ...
            T_degC(terrain(1))+273.15, m_kg_per_s, f, Lpump_m, M_kg_per_mol, g_m_per_s2, ...
            z_m, DataFluidProperties277K, DataFluidProperties288K);

        %    a = (max(ODNPS_terrain -2.*dtRatio(terrain(1)).*ODNPS_terrain - IDcalc_m) < 0)
        switch max(ODNPS_terrain -2.*dtRatio(terrain(1)).*ODNPS_terrain - IDcalc_m) < 0
            case 0 %there exist a ODNPS larger than IDcalc
                idxPinlet = idxPinlet + 1;

                C0 = 100^10; %re-initiating C0
                [configuration, config] = FindBestSteelGrade(configuration, config, ...
                    m_kg_per_s, timeFrame, terrain, z_m, L_km, ODNPS_terrain, idxPinlet, ...
                    Pinlet_MPa, Poutlet_MPa, phase, IDcalc_m, C0, T_degC, v_m_per_s, ...
                    rhoSteel_kg_per_m3, Csteel_EUR_per_kg, SteelFactor, steelTimeFrame, SteelGrade, ...
                    S_MPa, F, E, CA_m, dtRatio, rho_kg_per_m3, mu_Pas, ...
                    Clab_EUR_per_m2, CROW_EUR_per_m, mu_misc, muOMpipe, ...
                    epsilon_m, R_J_per_mol_per_K, M_kg_per_mol, g_m_per_s2, ...
                    DataFluidProperties277K, DataFluidProperties288K);

                [configuration, config, LClow] = FindLowestLevelizedCosts(configuration, config, ...
                    m_kg_per_s, terrain, z_m, L_km, idxPinlet, phase, Npump, ...
                    Pinlet_MPa, Poutlet_MPa, T_degC, rho_kg_per_m3, mu_Pas, ...
                    etaPump, muOMpumpcomp, H_h, C_el_EUR_per_MWh, ...
                    epsilon_m, R_J_per_mol_per_K, M_kg_per_mol, g_m_per_s2, ...
                    vRange, r, z_pipe, z_pumpcomp, LClow, Pcapture_Pa, ...
                    DataFluidProperties277K, DataFluidProperties288K);

            case 1%there is no ODNPS larger than IDcalc --> break
        end
        Npump = Npump + 1;
    end
    Pinlet_MPa = Pinlet_MPa + 0.1;
end

%% Liquid

disp('Liquid configuration')

phase = 2;
Pinlet_MPa = 9; %[MPa]
Poutlet_MPa = 8; %[MPa]
IDcalc_m = 0.5; %[m]
LClow = 1000;

v_m_per_s = Velocity(m_kg_per_s, IDcalc_m, rho_kg_per_m3(phase,terrain(1)));
Re = ReynoldsNumber(rho_kg_per_m3(phase,terrain(1)), IDcalc_m, v_m_per_s, mu_Pas(phase,terrain(1)));
f = DarcyWeisbach(epsilon_m, IDcalc_m, Re);


while Pinlet_MPa <= PinletMAX_MPa(phase,terrain(1))
    Npump = 0;
    while Npump <= Npump_max

        DeltaPdesign_Pa_per_m = DesignPressureDrop(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, Npump, g_m_per_s2, rho_kg_per_m3(phase,terrain(1)), z_m, L_km.*1e3);
        IDcalc_m = InnerDiameterLiquid(f, m_kg_per_s, rho_kg_per_m3(phase,terrain(1)), DeltaPdesign_Pa_per_m);


        switch max(ODNPS_terrain -2.*dtRatio(terrain(1)).*ODNPS_terrain - IDcalc_m) < 0
            case 0 %there exist a ODNPS larger than IDcalc
                idxPinlet = idxPinlet + 1;

                C0 = 100^10;
                [configuration, config] = FindBestSteelGrade(configuration, config, ...
                    m_kg_per_s, timeFrame, terrain, z_m, L_km, ODNPS_terrain, idxPinlet, ...
                    Pinlet_MPa, Poutlet_MPa, phase, IDcalc_m, C0, T_degC, v_m_per_s, ...
                    rhoSteel_kg_per_m3, Csteel_EUR_per_kg, SteelFactor, steelTimeFrame, SteelGrade, ...
                    S_MPa, F, E, CA_m, dtRatio, rho_kg_per_m3, mu_Pas, ...
                    Clab_EUR_per_m2, CROW_EUR_per_m, mu_misc, muOMpipe, ...
                    epsilon_m, R_J_per_mol_per_K, M_kg_per_mol, g_m_per_s2, ...
                    DataFluidProperties277K, DataFluidProperties288K);

                [configuration, config, LClow] = FindLowestLevelizedCosts(configuration, config, ...
                    m_kg_per_s, terrain, z_m, L_km, idxPinlet, phase, Npump, ...
                    Pinlet_MPa, Poutlet_MPa, T_degC, rho_kg_per_m3, mu_Pas, ...
                    etaPump, muOMpumpcomp, H_h, C_el_EUR_per_MWh, ...
                    epsilon_m, R_J_per_mol_per_K, M_kg_per_mol, g_m_per_s2, ...
                    vRange, r, z_pipe, z_pumpcomp, LClow, Pcapture_Pa, ...
                    DataFluidProperties277K, DataFluidProperties288K);
        end

        Npump = Npump + 1;
    end
    Pinlet_MPa = Pinlet_MPa + 1;

end

%% Save configuration

% nameTerrain = {'offshore','onshore'};

% filename = strcat(nameTerrain{terrain(1)},'_',sprintf('%.e',m_kg_per_s*3600*24*365/1000),'_t_per_y_',sprintf('%.0f',L_km),'_km.mat');
% save(filename,"configuration")

end