function [configuration, config] = FindBestSteelGrade(configuration, config, ...
    m_kg_per_s, timeFrame, terrain, z_m, L_km, ODNPS_terrain, idxPinlet, ...
    Pinlet_MPa, Poutlet_MPa, phase, IDcalc_m, C0, T_degC, v_m_per_s, ...
    rhoSteel_kg_per_m3, Csteel_EUR_per_kg, SteelFactor, steelTimeFrame, SteelGrade, ...
    S_MPa, F, E, CA_m, dtRatio, rho_kg_per_m3, mu_Pas, ...
    Clab_EUR_per_m2, CROW_EUR_per_m, mu_misc, muOMpipe, ...
    epsilon_m, R_J_per_mol_per_K, M_kg_per_mol, g_m_per_s2, ...
    DataFluidProperties277K, DataFluidProperties288K)
%This function computes the best configuration of pipeline for a certain
%inlet pressure
%INPUT: configuration
%       config
%       m_kg_per_s= CO2 mass flow [kg.s-1]
%       timeFrame
%       terrain
%       z_m = difference in altitude between start and end of the pipe [m]
%       L_km = length of the pipe [km]
%       ODNPS_terrain
%       idxPinlet
%       Pinlet_MPa = inlet pressure [MPa]
%       Poutlet_MPa = outlet pressure [MPa]
%       phase
%       IDcalc_m
%       C0
%       T_degC
%       v_m_per_s
%       rhoSteel_kg_per_m3
%       Csteel_EUR_per_kg
%       SteelFactor
%       steelTimeFrame
%       SteelGrade
%       S_MPa
%       F
%       E
%       CA_m
%       dtRatio
%       rho_kg_per_m3 = CO2 density [kg.m-3]
%       mu_Pas
%       Clab_EUR_per_m2
%       CROW_EUR_per_m
%       mu_misc
%       muOMpipe
%       epsilon_m = roughness height [m]
%       R_J_per_mol_per_K = gas constant [J.mol-1.K-1]
%       M_kg_per_mol = molar mass of CO2 [kg.mol-1]
%       g_m_per_s2 = gravitation constant [m.s-2]
%       DataFluidProperties277K = Table with fluid properties at 277 K
%       DataFluidProperties288K = Table with fluid properties at 288 K
%OUTPUT:    config = configuration chosen for each inlet pressure and steel grade
%           configuration = configuration chosen for each inlet pressure

config.inletPressure(idxPinlet).Pinlet_MPa = Pinlet_MPa;
MAOP_MPa = MaximumAllowableOperationPressure(Pinlet_MPa);

for steelGradeIdx = steelTimeFrame{timeFrame}

    IDNPS_m = 0;

    disp(sprintf('Calculating the inner and outer diameters for inlet pressure of %d MPa and steel grade %s',Pinlet_MPa,SteelGrade{steelGradeIdx}))

    while IDcalc_m > IDNPS_m

        ODNPSidx = find(ODNPS_terrain - IDcalc_m > 0,1,'first');
        while (IDNPS_m < IDcalc_m) 
            switch isempty(ODNPSidx)
                case 0
                    switch (ODNPSidx > length(ODNPS_terrain))
                        case 1
                            break
                    end
                case 1
                    break
            end

            ODNPS_m = ODNPS_terrain(ODNPSidx);

            t_m = PipeThickness(ODNPS_m, MAOP_MPa, S_MPa(steelGradeIdx), F(terrain(1)), E, CA_m, dtRatio(terrain(1)));

            IDNPS_m = InnerDiameter(ODNPS_m, t_m);

            ODNPSidx = ODNPSidx + 1;
        end
        %     ODNPSidx = ODNPSidx - 1;

        switch isempty(ODNPSidx)
            case 0
                switch (ODNPSidx > length(ODNPS_terrain))
                    case 1
                        break
                end
            case 1
                break
        end

        v_m_per_s = Velocity(m_kg_per_s, IDNPS_m, rho_kg_per_m3(phase,terrain(1)));
        Re = ReynoldsNumber(rho_kg_per_m3(phase,terrain(1)), IDNPS_m, v_m_per_s, mu_Pas(phase,terrain(1)));
        f = DarcyWeisbach(epsilon_m, IDNPS_m, Re);
        DeltaPact_Pa_per_m = ActualPressureDrop(f,m_kg_per_s, rho_kg_per_m3(phase,terrain(1)),IDNPS_m);
        Lpump_m = MaximumDistanceBetweenPumpingStations(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, DeltaPact_Pa_per_m);
        Npump = NumberPumpingStations(L_km.*1e3,Lpump_m);
        DeltaPdesign_Pa_per_m = DesignPressureDrop(Pinlet_MPa.*1e6, Poutlet_MPa.*1e6, Npump, g_m_per_s2, rho_kg_per_m3(phase,terrain(1)), z_m, L_km.*1e3);

        switch phase
            case 1 %gas
%                 IDcalc_m = InnerDiameterGas(Poutlet_MPa.*1e6, Pinlet_MPa.*1e6, R_J_per_mol_per_K, ...
%                     T_degC(terrain(1))+273.15, m_kg_per_s, f, L_km.*1e3, M_kg_per_mol, g_m_per_s2, z_m, ...
%                     DataFluidProperties277K, DataFluidProperties288K);
                IDcalc_m = InnerDiameterGas(Poutlet_MPa.*1e6, Pinlet_MPa.*1e6, R_J_per_mol_per_K, ...
                    T_degC(terrain(1))+273.15, m_kg_per_s, f, Lpump_m, M_kg_per_mol, g_m_per_s2, z_m, ...
                    DataFluidProperties277K, DataFluidProperties288K);
            case 2 %liquid
                IDcalc_m = InnerDiameterLiquid(f, m_kg_per_s, rho_kg_per_m3(phase,terrain(1)), DeltaPdesign_Pa_per_m);
        end

    end

    Imaterial_EUR = PipeMaterialCost(t_m, ODNPS_m, L_km.*1e3, rhoSteel_kg_per_m3, ...
        Csteel_EUR_per_kg(steelGradeIdx), SteelFactor);
    Ilab_EUR = PipeLaborCost(ODNPS_m, L_km.*1e3, Clab_EUR_per_m2);
    IROW_EUR = PipeROWCost(L_km.*1e3, CROW_EUR_per_m(terrain(1)));
    Imisc_EUR = PipeMiscellaneous(Imaterial_EUR, Ilab_EUR, mu_misc);
    Ipipe_EUR = PipeInvestment(Imaterial_EUR, Ilab_EUR, IROW_EUR, Imisc_EUR);
    OCpipe_EUR_per_y = OpAndM(Ipipe_EUR, muOMpipe);

    %Save run
    config.inletPressure(idxPinlet).run(steelGradeIdx).IDcalc_m = IDcalc_m;
    config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m = IDNPS_m;
    %     config.inletPressure(idxPinlet).run(steelGradeIdx).ODNPSidx = ODNPSidx;
    config.inletPressure(idxPinlet).run(steelGradeIdx).MAOP_MPa = MAOP_MPa;
    config.inletPressure(idxPinlet).run(steelGradeIdx).ODNPS_m = ODNPS_m;
    config.inletPressure(idxPinlet).run(steelGradeIdx).t_m = t_m;
    config.inletPressure(idxPinlet).run(steelGradeIdx).Cmaterial_EUR = Imaterial_EUR;
    config.inletPressure(idxPinlet).run(steelGradeIdx).Clab_EUR = Ilab_EUR;
    config.inletPressure(idxPinlet).run(steelGradeIdx).CROW_EUR = IROW_EUR;
    config.inletPressure(idxPinlet).run(steelGradeIdx).Cmisc_EUR = Imisc_EUR;
    config.inletPressure(idxPinlet).run(steelGradeIdx).Ipipe_EUR = Ipipe_EUR;
    config.inletPressure(idxPinlet).run(steelGradeIdx).OCpipe_EUR_per_y = OCpipe_EUR_per_y;

    switch Ipipe_EUR < C0
        case 1
            C0 = Ipipe_EUR;
            configuration.inletPressure(idxPinlet).Pinlet_MPa = Pinlet_MPa;
            configuration.inletPressure(idxPinlet).optSteelGrade = SteelGrade(steelGradeIdx);
            configuration.inletPressure(idxPinlet).optSteelGradeIdx = steelGradeIdx;
            configuration.inletPressure(idxPinlet).IDNPS_m = IDNPS_m;
            configuration.inletPressure(idxPinlet).ODNPS_m = ODNPS_m;
            configuration.inletPressure(idxPinlet).t_m = t_m;
            configuration.inletPressure(idxPinlet).Cmaterial_EUR = Imaterial_EUR;
            configuration.inletPressure(idxPinlet).Clab_EUR = Ilab_EUR;
            configuration.inletPressure(idxPinlet).CROW_EUR = IROW_EUR;
            configuration.inletPressure(idxPinlet).Cmisc_EUR = Imisc_EUR;
            configuration.inletPressure(idxPinlet).Ipipe_EUR = Ipipe_EUR;
            configuration.inletPressure(idxPinlet).OCpipe_EUR_per_y = OCpipe_EUR_per_y;
    end

    %     switch steelGradeIdx
    %         case 2
    %             switch config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m - config.inletPressure(idxPinlet).run(steelGradeIdx-1).IDNPS_m == 0
    %                 case 0
    %                     Re = ReynoldsNumber(rho_kg_per_m3(phase,terrain(1)), config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m, v_m_per_s, mu_Pas(phase,terrain(1)));
    %                     f = DarcyWeisbach(epsilon_m, config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m, Re);
    %                     IDcalc_m = InnerDiameterGas(Poutlet_MPa.*1e6, Pinlet_MPa.*1e6, R_J_per_mol_per_K, ...
    %                         T_degC(terrain(1))+273.15, m_kg_per_s, f, L_km.*1e3, M_kg_per_mol, g_m_per_s2, z_m);
    %             end
    %         case {3:8}
    %             switch (config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m - config.inletPressure(idxPinlet).run(steelGradeIdx-1).IDNPS_m == 0) & (config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m - config.inletPressure(idxPinlet).run(steelGradeIdx-2).IDNPS_m == 0)
    %                 case 0
    %                     Re = ReynoldsNumber(rho_kg_per_m3(phase,terrain(1)), config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m, v_m_per_s, mu_Pas(phase,terrain(1)));
    %                     f = DarcyWeisbach(epsilon_m, config.inletPressure(idxPinlet).run(steelGradeIdx).IDNPS_m, Re);
    %                     IDcalc_m = InnerDiameterGas(config.Poutlet_MPa.*1e6, config.Pinlet_MPa.*1e6, R_J_per_mol_per_K, ...
    %                         T_degC(terrain(1))+273.15, m_kg_per_s, f, L_km.*1e3, M_kg_per_mol, g_m_per_s2, z_m);
    %             end
    %     end

end

end