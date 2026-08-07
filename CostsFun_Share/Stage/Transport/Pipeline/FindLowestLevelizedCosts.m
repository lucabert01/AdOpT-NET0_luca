function [configuration, config, LClow] = FindLowestLevelizedCosts(configuration, config, ...
    m_kg_per_s, terrain, z_m, L_km, idxPinlet, phase, Npump, ...
    Pinlet_MPa, Poutlet_MPa, T_degC, rho_kg_per_m3, mu_Pas, ...
    etaPump, muOMpumpcomp, H_h, C_el_EUR_per_MWh, ...
    epsilon_m, R_J_per_mol_per_K, M_kg_per_mol, g_m_per_s2, ...
    vRange, r, z_pipe, z_pumpcomp, LClow, Pcapture_Pa, ...
    DataFluidProperties277K, DataFluidProperties288K)

%This function computes the lwerst levelized costs for a certain inlet
%pressure
%INPUT: config = configuration chosen for each inlet pressure and steel grade
%       configuration = configuration chosen for each inlet pressure
%       m_kg_per_s= CO2 mass flow [kg.s-1]
%       terrain
%       z_m = difference in altitude between start and end of the pipe [m]
%       L_km = length of the pipe [km]
%       idxPinlet
%       phase
%       Npump
%       Pinlet_MPa = inlet pressure [MPa]
%       Poutlet_MPa = outlet pressure [MPa]
%       T_degC
%       rhoSteel_kg_per_m3
%       mu_Pas
%       etaPump
%       muOMpumpcomp
%       H_h
%       C_el_EUR_per_MWh
%       epsilon_m = roughness height [m]
%       R_J_per_mol_per_K = gas constant [J.mol-1.K-1]
%       M_kg_per_mol = molar mass of CO2 [kg.mol-1]
%       g_m_per_s2 = gravitation constant [m.s-2]
%       vRange = velocity range
%       r
%       z_pipe
%       z_pumpcomp
%       LC_low
%       P_capture_Pa = pressure after capture
%       DataFluidProperties277K = Table with fluid properties at 277 K
%       DataFluidProperties288K = Table with fluid properties at 288 K
%OUTPUT: config = configuration chosen for each inlet pressure
%        configuration = configuration chosen for a specific phase
%        LClow = updated value of lowest levelized costs

%% Data 
optSteelGrade = configuration.inletPressure(idxPinlet).optSteelGrade;
IDNPS_m = configuration.inletPressure(idxPinlet).IDNPS_m;
ODNPS_m = configuration.inletPressure(idxPinlet).ODNPS_m;
Ipipe_EUR = configuration.inletPressure(idxPinlet).Ipipe_EUR;
OCpipe_EUR_per_y = configuration.inletPressure(idxPinlet).OCpipe_EUR_per_y;

%% Calculations

v_m_per_s = Velocity(m_kg_per_s, IDNPS_m, rho_kg_per_m3(phase,terrain(1)));

switch (v_m_per_s >= vRange(phase,1)) & (v_m_per_s <= vRange(phase,2))
    case 0 %the velocity is not within the given range --> do not keep this case
        configuration.inletPressure(idxPinlet).v_m_per_s = v_m_per_s;
        configuration.inletPressure(idxPinlet).Ecomp_kJ_per_kg = NaN;
        configuration.inletPressure(idxPinlet).Wcomp_MW = NaN;
        configuration.inletPressure(idxPinlet).Epump_MJ_per_kg = NaN;
        configuration.inletPressure(idxPinlet).Wpump_MW = NaN;
        configuration.inletPressure(idxPinlet).Icomp_EUR = NaN;
        configuration.inletPressure(idxPinlet).OCcomp_EUR_per_y = NaN;
        configuration.inletPressure(idxPinlet).ECcomp_EUR_per_y = NaN;
        configuration.inletPressure(idxPinlet).Ipump_EUR = NaN;
        configuration.inletPressure(idxPinlet).OCpump_EUR_per_y = NaN;
        configuration.inletPressure(idxPinlet).ECpump_EUR_per_y = NaN;
        configuration.inletPressure(idxPinlet).LC_EUR_per_t = NaN;
        configuration.inletPressure(idxPinlet).LCtrans_EUR_per_t = NaN;
        configuration.inletPressure(idxPinlet).LCcomp_EUR_per_t = NaN;
        configuration.inletPressure(idxPinlet).Npump = NaN;
        configuration.inletPressure(idxPinlet).Lpump_km = NaN;
        configuration.inletPressure(idxPinlet).DeltaPact_Pa_per_m = NaN;
    case 1
        Re = ReynoldsNumber(rho_kg_per_m3(phase,terrain(1)), IDNPS_m, v_m_per_s, mu_Pas(phase,terrain(1)));
        f = DarcyWeisbach(epsilon_m, IDNPS_m, Re);
        DeltaPact_Pa_per_m = ActualPressureDrop(f, m_kg_per_s, rho_kg_per_m3(phase,terrain(1)), IDNPS_m);
        switch phase
            case 1 %gas
                %Not yet the density of the gas/liquid at outlet pressure for the compressor energy!
                switch terrain(1)
                    case 1 %offshore
                        Lpump_m = NaN;
                        Lpump_km = NaN;
                        [Poutlet_adapted_gas_Pa, Poutlet_adapted_gas_Pa_wrong] = CompressorOutletPressure(R_J_per_mol_per_K, ...
                            T_degC(terrain(1))+273.15, m_kg_per_s, f, L_km.*1e3, g_m_per_s2, Poutlet_MPa.*1e6, ...
                            Pinlet_MPa.*1e6, M_kg_per_mol, z_m, IDNPS_m, DataFluidProperties277K, DataFluidProperties288K);
                        P2_Pa = Poutlet_adapted_gas_Pa;
                        P2_Pa_wrong = Poutlet_adapted_gas_Pa_wrong;

                        [Epump_MJ_per_kg, Wpump_MW] = PumpingEnergy(Pinlet_MPa, Poutlet_MPa, etaPump, rho_kg_per_m3(phase,terrain(1)), m_kg_per_s, phase, terrain(1));
                        Ipump_EUR = PumpInvestment(Wpump_MW, Npump, phase);
                        ECpump_EUR_per_y = PumpEnergyCosts(Wpump_MW, Npump, H_h, C_el_EUR_per_MWh);
                    case 2 %onshore
                        Lpump_m = MaximumDistanceBetweenPumpingStations(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, DeltaPact_Pa_per_m);
                        Lpump_km = Lpump_m./1000;
                        [Poutlet_adapted_gas_Pa, Poutlet_adapted_gas_Pa_wrong] = CompressorOutletPressure(R_J_per_mol_per_K, ...
                            T_degC(terrain(1))+273.15, m_kg_per_s, f, Lpump_m, g_m_per_s2, Poutlet_MPa.*1e6, ...
                            Pinlet_MPa.*1e6, M_kg_per_mol, z_m, IDNPS_m, DataFluidProperties277K, DataFluidProperties288K);
                        Npump = NumberPumpingStations(L_km.*1e3, Lpump_m);
                        Poutlet_last_pump_Pa = OutletPressureLastPumpingStation(Poutlet_MPa.*1e6, L_km.*1e3, Lpump_m, Npump, DeltaPact_Pa_per_m);
                        P2_Pa = Poutlet_adapted_gas_Pa;
                        P2_Pa_wrong = Pinlet_MPa.*1e6;

                        %In the case of gas transport, pumping stations are
                        %recompression stations
                        [Epump_MJ_per_kg, Wpump_MW, E_lastpump_MJ_per_kg, W_lastpump_MW] = PumpingEnergy(Pinlet_MPa, Poutlet_MPa, etaPump, rho_kg_per_m3(phase,terrain(1)), m_kg_per_s, phase, terrain(1), Poutlet_last_pump_Pa, R_J_per_mol_per_K, M_kg_per_mol);
                        Ipump_EUR = PumpInvestment(Wpump_MW, Npump, phase, W_lastpump_MW);
                        ECpump_EUR_per_y = PumpEnergyCosts(Wpump_MW, Npump, H_h, C_el_EUR_per_MWh, W_lastpump_MW);
                end

            case 2 %liquid
                switch terrain(1)
                    case 1 %offshore
                        Lpump_m = NaN;
                        Lpump_km = NaN;
                        Poutlet_last_pump_Pa = OutletPressureLastPumpingStation(Poutlet_MPa.*1e6, L_km.*1e3, 1, Npump, DeltaPact_Pa_per_m);
                        P2_Pa = Poutlet_last_pump_Pa;
                        P2_Pa_wrong = NaN;

                        [Epump_MJ_per_kg, Wpump_MW] = PumpingEnergy(Pinlet_MPa, Poutlet_MPa, etaPump, rho_kg_per_m3(phase,terrain(1)), m_kg_per_s, phase, terrain(1));
                        Ipump_EUR = PumpInvestment(Wpump_MW, Npump, phase);
                        ECpump_EUR_per_y = PumpEnergyCosts(Wpump_MW, Npump, H_h, C_el_EUR_per_MWh);
                    case 2 %onshore
                        Lpump_m = MaximumDistanceBetweenPumpingStations(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, DeltaPact_Pa_per_m);
                        Lpump_km = Lpump_m./1000;
                        Npump = NumberPumpingStations(L_km.*1e3, Lpump_m);
                        Poutlet_last_pump_Pa = OutletPressureLastPumpingStation(Poutlet_MPa.*1e6, L_km.*1e3, Lpump_m, Npump, DeltaPact_Pa_per_m);
                        P2_Pa = Pinlet_MPa.*1e6;
                        P2_Pa_wrong = NaN;

                        [Epump_MJ_per_kg, Wpump_MW, E_lastpump_MJ_per_kg, W_lastpump_MW] = PumpingEnergy(Pinlet_MPa, Poutlet_MPa, etaPump, rho_kg_per_m3(phase,terrain(1)), m_kg_per_s, phase, terrain(1), Poutlet_last_pump_Pa);
                        Ipump_EUR = PumpInvestment(Wpump_MW, Npump, phase, W_lastpump_MW);
                        ECpump_EUR_per_y = PumpEnergyCosts(Wpump_MW, Npump, H_h, C_el_EUR_per_MWh, W_lastpump_MW);
                end
        end
        [Ecomp_kJ_per_kg, Wcomp_MW] = CompressorEnergy(P2_Pa, Pcapture_Pa, m_kg_per_s, 303.15, R_J_per_mol_per_K, M_kg_per_mol, rho_kg_per_m3(phase,terrain(1)));                     
        Icomp_EUR = CompressorInvestment(Wcomp_MW);
        OCcomp_EUR_per_y = OpAndM(Icomp_EUR, muOMpumpcomp);
        ECcomp_EUR_per_y = CompressorEnergyCosts(Wcomp_MW, H_h, C_el_EUR_per_MWh);     
        OCpump_EUR_per_y = OpAndM(Ipump_EUR, muOMpumpcomp);   
        [LC_EUR_per_t, LCtrans_EUR_per_t, LCcomp_EUR_per_t] = LevelizedCosts(Ipipe_EUR, OCpipe_EUR_per_y, ...
            Icomp_EUR, OCcomp_EUR_per_y, ECcomp_EUR_per_y, ...
            Ipump_EUR, OCpump_EUR_per_y, ECpump_EUR_per_y, ...
            r, z_pipe, z_pumpcomp, m_kg_per_s, H_h);

        %Calculate for wrong numbers
        [Ecomp_kJ_per_kg_wrong, Wcomp_MW_wrong] = CompressorEnergy(P2_Pa_wrong, Pcapture_Pa, m_kg_per_s, 303.15, R_J_per_mol_per_K, M_kg_per_mol, rho_kg_per_m3(phase,terrain(1)));                    
        Icomp_EUR_wrong = CompressorInvestment(Wcomp_MW_wrong);
        OCcomp_EUR_per_y_wrong = OpAndM(Icomp_EUR_wrong, muOMpumpcomp);
        ECcomp_EUR_per_y_wrong = CompressorEnergyCosts(Wcomp_MW_wrong, H_h, C_el_EUR_per_MWh);       
        [LC_EUR_per_t_wrong, LCtrans_EUR_per_t_wrong, LCcomp_EUR_per_t_wrong] = LevelizedCosts(Ipipe_EUR, OCpipe_EUR_per_y, ...
            Icomp_EUR_wrong, OCcomp_EUR_per_y_wrong, ECcomp_EUR_per_y_wrong, ...
            Ipump_EUR, OCpump_EUR_per_y, ECpump_EUR_per_y, ...
            r, z_pipe, z_pumpcomp, m_kg_per_s, H_h);

        disp(sprintf('New optimal configuration found in phase %d for inlet pressure of %d MPa',phase,Pinlet_MPa))

        configuration.inletPressure(idxPinlet).v_m_per_s = v_m_per_s;
        configuration.inletPressure(idxPinlet).Ecomp_kJ_per_kg = Ecomp_kJ_per_kg;
        configuration.inletPressure(idxPinlet).Wcomp_MW = Wcomp_MW;
        configuration.inletPressure(idxPinlet).Epump_MJ_per_kg = Epump_MJ_per_kg;
        configuration.inletPressure(idxPinlet).Wpump_MW = Wpump_MW;
        configuration.inletPressure(idxPinlet).Icomp_EUR = Icomp_EUR;
        configuration.inletPressure(idxPinlet).OCcomp_EUR_per_y = OCcomp_EUR_per_y;
        configuration.inletPressure(idxPinlet).ECcomp_EUR_per_y = ECcomp_EUR_per_y;
        configuration.inletPressure(idxPinlet).Ipump_EUR = Ipump_EUR;
        configuration.inletPressure(idxPinlet).OCpump_EUR_per_y = OCpump_EUR_per_y;
        configuration.inletPressure(idxPinlet).ECpump_EUR_per_y = ECpump_EUR_per_y;
        configuration.inletPressure(idxPinlet).LC_EUR_per_t = LC_EUR_per_t;
        configuration.inletPressure(idxPinlet).LCtrans_EUR_per_t = LCtrans_EUR_per_t;
        configuration.inletPressure(idxPinlet).LCcomp_EUR_per_t = LCcomp_EUR_per_t;
        configuration.inletPressure(idxPinlet).Npump = Npump;
        configuration.inletPressure(idxPinlet).Lpump_km = Lpump_km;
        configuration.inletPressure(idxPinlet).DeltaPact_Pa_per_m = DeltaPact_Pa_per_m;

        switch LC_EUR_per_t < LClow
            case 1
                disp(sprintf('New optimal configuration found in phase %d',phase))

                LClow = LC_EUR_per_t;
                configuration.config(phase).LC_EUR_per_t = LC_EUR_per_t;
                configuration.config(phase).LCtrans_EUR_per_t = LCtrans_EUR_per_t;
                configuration.config(phase).ODNPS_m = ODNPS_m;
                configuration.config(phase).IDNPS_m = IDNPS_m;
                configuration.config(phase).Pinlet_MPa = Pinlet_MPa;
                configuration.config(phase).Npump = Npump;
                configuration.config(phase).optSteelGrade = optSteelGrade;
                configuration.config(phase).DeltaPact_Pa_per_m = DeltaPact_Pa_per_m;

                configuration.config(phase).t_m = configuration.inletPressure(idxPinlet).t_m;
                configuration.config(phase).Lpump_km = Lpump_km;
                configuration.config(phase).v_m_per_s = v_m_per_s;
                configuration.config(phase).Ecomp_kJ_per_kg = Ecomp_kJ_per_kg;
                configuration.config(phase).Wcomp_MW = Wcomp_MW;
                configuration.config(phase).Epump_MJ_per_kg = Epump_MJ_per_kg;
                configuration.config(phase).Wpump_MW = Wpump_MW;
                configuration.config(phase).Cmaterial_EUR = configuration.inletPressure(idxPinlet).Cmaterial_EUR;
                configuration.config(phase).Clab_EUR = configuration.inletPressure(idxPinlet).Clab_EUR;
                configuration.config(phase).CROW_EUR = configuration.inletPressure(idxPinlet).CROW_EUR;
                configuration.config(phase).Cmisc_EUR = configuration.inletPressure(idxPinlet).Cmisc_EUR;
                configuration.config(phase).Ipipe_EUR = Ipipe_EUR;
                configuration.config(phase).OCpipe_EUR_per_y = OCpipe_EUR_per_y;
                configuration.config(phase).Icomp_EUR = Icomp_EUR;
                configuration.config(phase).OCcomp_EUR_per_y = OCcomp_EUR_per_y;
                configuration.config(phase).ECcomp_EUR_per_y = ECcomp_EUR_per_y;
                configuration.config(phase).Ipump_EUR = Ipump_EUR;
                configuration.config(phase).OCpump_EUR_per_y = OCpump_EUR_per_y;
                configuration.config(phase).ECpump_EUR_per_y = ECpump_EUR_per_y;
                configuration.config(phase).LCcomp_EUR_per_t = LCcomp_EUR_per_t;
               
                switch phase
                    case 1 %gas
                        configuration.config(phase).Poutlet_adapted_gas_Pa = Poutlet_adapted_gas_Pa;
                        configuration.config(phase).Poutlet_adapted_gas_Pa_wrong = Poutlet_adapted_gas_Pa_wrong;
                        
                        configuration.config(phase).Ecomp_kJ_per_kg_wrong = Ecomp_kJ_per_kg_wrong;
                        configuration.config(phase).Wcomp_MW_wrong = Wcomp_MW_wrong;
                        configuration.config(phase).Icomp_EUR_wrong = Icomp_EUR_wrong;
                        configuration.config(phase).OCcomp_EUR_per_y_wrong = OCcomp_EUR_per_y_wrong;
                        configuration.config(phase).ECcomp_EUR_per_y_wrong = ECcomp_EUR_per_y_wrong;
                        configuration.config(phase).LC_EUR_per_t_wrong = LC_EUR_per_t_wrong;
                        configuration.config(phase).LCtrans_EUR_per_t_wrong = LCtrans_EUR_per_t_wrong;
                        configuration.config(phase).LCcomp_EUR_per_t_wrong = LCcomp_EUR_per_t_wrong;
                end
        end

end



end