function [Terrain] = PipelineCostsforGrid(m_t_per_y_lin, L_km_lin, z_m, timeFrame, ...
    pipeline, finance, electricity, b_meshgrid)
%Computes the pipeline costs for a grid of amounts transported and
%distances
%INPUT: m_t_per_y_lin = 1xn vector with amounts transported
%       L_km_lin = 1xm vector with distances between two nodes
%       z_m = elevation difference between two nodes
%       timeFrame = time horizon that is decisive for the steel types available: 1 = short-term, 2 = mid-term, 3 = long-term
%       pipeline = pipeline struct
%       finance = finance struct
%       electricity = electricity struct
%       b_meshgrid = boolean saying if the amounts and the distances should create a grid
%OUTPUT: Terrain = struct containing information relevant to the pipelines built, for each terrain (on/offshore) and phase (gas/dense)

r = finance.r;
SteelFactor = pipeline.SteelFactor;
Clab_EUR_per_m2 = pipeline.C_lab_EUR_per_m2;
CROW_EUR_per_m = [0 pipeline.C_ROW_EUR_per_m];
C_el_EUR_per_MWh = electricity.T_electricity.EU27(1).*1e3;
F = [0.72 0.61];
ODNPS_possible = [{[0.03 0.04 0.05 0.06 0.07 0.09 0.1 0.11 0.14 0.17 0.22 ...
    0.27 0.32 0.36 0.41 0.51 0.61 0.76 0.91 1.07 1.22 1.32 1.42]}, ...
    {[0.03 0.04 0.05 0.06 0.07 0.09 0.1 0.11 0.14 0.17 0.22 ...
    0.27 0.32 0.36 0.41 0.51 0.61 0.76 0.91 1.07 1.22 1.32 1.42]}];

m_kg_per_s_lin = m_t_per_y_lin .*1000./(365*24*3600);

if b_meshgrid
    [M, L] = meshgrid(m_kg_per_s_lin, L_km_lin);
    
    M_lin = reshape(M,[numel(M),1,1]);
    L_lin = reshape(L,[numel(L),1,1]);
    % T_lin = reshape(T,[numel(T),1,1]);
    
    Data = table(M_lin.*365.*24.*3600./1000, M_lin, L_lin, 'VariableNames',{'m_t_per_y','m_kg_per_s','L_km'});
else
    Data = table(m_t_per_y_lin, m_kg_per_s_lin, L_km_lin, 'VariableNames',{'m_t_per_y','m_kg_per_s','L_km'});
end

DataGas = Data;
DataLiq = Data;

Terrain = struct;

%% Algorithm

for terrain = 1:2 %1 = offshore, 2 = onshore
    for i = 1:height(Data)

        %% Case data

        m_kg_per_s = Data.m_kg_per_s(i); %[kg.s-1] mass flow

        L_km = Data.L_km(i); %[km] length

        %% Algorithm

        [configuration, config] = AdaptedKnoopesConfiguration(m_kg_per_s, timeFrame, ...
            terrain, z_m, L_km, r, F, SteelFactor, Clab_EUR_per_m2, CROW_EUR_per_m, ...
            C_el_EUR_per_MWh, ODNPS_possible);

%         folder = fileparts(which('main.m'));
%         mkdir(strcat(folder,'/CostsFun/Stage/Transport/Pipeline/Configurations/','Terrain_',num2str(terrain),'_',num2str(L_km),'_km_',num2str(m_kg_per_s),'kg_per_s'))
%         foldername = strcat(folder,'/CostsFun/Stage/Transport/Pipeline/Configurations/','Terrain_',num2str(terrain),'_',num2str(L_km),'_km_',num2str(m_kg_per_s),'kg_per_s/');
%         filenameconfiguration = strcat(foldername,'Configuration');
%         save(filenameconfiguration,'configuration')

        %% Check

        switch numel(fieldnames(configuration))
            case 2
                switch isempty(configuration.config(1).LC_EUR_per_t)
                    case 0
                        DataGas.OD_m(i) = configuration.config(1).ODNPS_m;
                        DataGas.Pinlet_MPa(i) = configuration.config(1).Pinlet_MPa;
                        DataGas.Npump(i) = configuration.config(1).Npump;
                        DataGas.LC_EUR_per_t(i) = configuration.config(1).LC_EUR_per_t;
                        DataGas.LCtrans(i) = configuration.config(1).LCtrans_EUR_per_t;
                        DataGas.optSteelGrade(i) = configuration.config(1).optSteelGrade;
                        DataGas.DeltaPact_Pa_per_m(i) = configuration.config(1).DeltaPact_Pa_per_m;

                        DataGas.t_m(i) = configuration.config(1).t_m;
                        DataGas.Lpump_km(i) = configuration.config(1).Lpump_km;
                        DataGas.v_m_per_s(i) = configuration.config(1).v_m_per_s;
                        DataGas.Ecomp_kJ_per_kg(i) = configuration.config(1).Ecomp_kJ_per_kg;
                        DataGas.Wcomp_MW(i) = configuration.config(1).Wcomp_MW;
                        DataGas.Epump_MJ_per_kg(i) = configuration.config(1).Epump_MJ_per_kg;
                        DataGas.Wpump_MW(i) = configuration.config(1).Wpump_MW;
                        DataGas.Cmaterial_EUR(i) = configuration.config(1).Cmaterial_EUR;
                        DataGas.Clab_EUR(i) = configuration.config(1).Clab_EUR;
                        DataGas.CROW_EUR(i) = configuration.config(1).CROW_EUR;
                        DataGas.Cmisc_EUR(i) = configuration.config(1).Cmisc_EUR;
                        DataGas.Ipipe_EUR(i) = configuration.config(1).Ipipe_EUR;
                        DataGas.OCpipe_EUR_per_y(i) = configuration.config(1).OCpipe_EUR_per_y;
                        DataGas.Icomp_EUR(i) = configuration.config(1).Icomp_EUR;
                        DataGas.OCcomp_EUR_per_y(i) = configuration.config(1).OCcomp_EUR_per_y;
                        DataGas.ECcomp_EUR_per_y(i) = configuration.config(1).ECcomp_EUR_per_y;
                        DataGas.Ipump_EUR(i) = configuration.config(1).Ipump_EUR;
                        DataGas.OCpump_EUR_per_y(i) = configuration.config(1).OCpump_EUR_per_y;
                        DataGas.ECpump_EUR_per_y(i) = configuration.config(1).ECpump_EUR_per_y;
                        DataGas.LCcomp_EUR_per_t(i) = configuration.config(1).LCcomp_EUR_per_t;

                        DataGas.Poutlet_adapted_gas_Pa(i) = configuration.config(1).Poutlet_adapted_gas_Pa;
                        DataGas.Poutlet_adapted_gas_Pa_wrong(i) = configuration.config(1).Poutlet_adapted_gas_Pa_wrong;
                        DataGas.Ecomp_kJ_per_kg_wrong(i) = configuration.config(1).Ecomp_kJ_per_kg_wrong;
                        DataGas.Wcomp_MW_wrong(i) = configuration.config(1).Wcomp_MW_wrong;
                        DataGas.Icomp_EUR_wrong(i) = configuration.config(1).Icomp_EUR_wrong;
                        DataGas.OCcomp_EUR_per_y_wrong(i) = configuration.config(1).OCcomp_EUR_per_y_wrong;
                        DataGas.ECcomp_EUR_per_y_wrong(i) = configuration.config(1).ECcomp_EUR_per_y_wrong;
                        DataGas.LC_EUR_per_t_wrong(i) = configuration.config(1).LC_EUR_per_t_wrong;
                        DataGas.LCtrans_EUR_per_t_wrong(i) = configuration.config(1).LCtrans_EUR_per_t_wrong;
                        DataGas.LCcomp_EUR_per_t_wrong(i) = configuration.config(1).LCcomp_EUR_per_t_wrong;
                      
                        DataGas.error_Poutlet_adapted_gas_perc(i) = (DataGas.Poutlet_adapted_gas_Pa_wrong(i) - DataGas.Poutlet_adapted_gas_Pa(i)).*100./ DataGas.Poutlet_adapted_gas_Pa(i);
                        DataGas.error_Ecomp_perc(i) = (DataGas.Ecomp_kJ_per_kg_wrong(i) - DataGas.Ecomp_kJ_per_kg(i)).*100./ DataGas.Ecomp_kJ_per_kg(i);
                        DataGas.error_Wcomp_perc(i) = (DataGas.Wcomp_MW_wrong(i) - DataGas.Wcomp_MW(i)).*100./ DataGas.Wcomp_MW(i);
                        DataGas.error_Icomp_perc(i) = (DataGas.Icomp_EUR_wrong(i) - DataGas.Icomp_EUR(i)).*100./ DataGas.Icomp_EUR(i);
                        DataGas.error_OCcomp_perc(i) = (DataGas.OCcomp_EUR_per_y_wrong(i) - DataGas.OCcomp_EUR_per_y(i)).*100./ DataGas.OCcomp_EUR_per_y(i);
                        DataGas.error_ECcomp_perc(i) = (DataGas.ECcomp_EUR_per_y_wrong(i) - DataGas.ECcomp_EUR_per_y(i)).*100./ DataGas.ECcomp_EUR_per_y(i);
                        DataGas.error_LC_perc(i) = (DataGas.LC_EUR_per_t_wrong(i) - DataGas.LC_EUR_per_t(i)).*100./ DataGas.LC_EUR_per_t(i);
                        DataGas.error_LCtrans_perc(i) = (DataGas.LCtrans_EUR_per_t_wrong(i) - DataGas.LCtrans(i)).*100./ DataGas.LCtrans(i);
                        DataGas.error_LCcomp_perc(i) = (DataGas.LCcomp_EUR_per_t_wrong(i) - DataGas.LCcomp_EUR_per_t(i)).*100./ DataGas.LCcomp_EUR_per_t(i);
                end
        
                switch isempty(configuration.config(2).LC_EUR_per_t)
                    case 0
                        DataLiq.OD_m(i) = configuration.config(2).ODNPS_m;
                        DataLiq.Pinlet_MPa(i) = configuration.config(2).Pinlet_MPa;
                        DataLiq.Npump(i) = configuration.config(2).Npump;
                        DataLiq.LC_EUR_per_t(i) = configuration.config(2).LC_EUR_per_t;
                        DataLiq.LCtrans(i) = configuration.config(2).LCtrans_EUR_per_t;
                        DataLiq.optSteelGrade(i) = configuration.config(2).optSteelGrade;
                        DataLiq.DeltaPact_Pa_per_m(i) = configuration.config(2).DeltaPact_Pa_per_m;

                        DataLiq.t_m(i) = configuration.config(2).t_m;
                        DataLiq.Lpump_km(i) = configuration.config(2).Lpump_km;
                        DataLiq.v_m_per_s(i) = configuration.config(2).v_m_per_s;
                        DataLiq.Ecomp_kJ_per_kg(i) = configuration.config(2).Ecomp_kJ_per_kg;
                        DataLiq.Wcomp_MW(i) = configuration.config(2).Wcomp_MW;
                        DataLiq.Epump_MJ_per_kg(i) = configuration.config(2).Epump_MJ_per_kg;
                        DataLiq.Wpump_MW(i) = configuration.config(2).Wpump_MW;
                        DataLiq.Cmaterial_EUR(i) = configuration.config(2).Cmaterial_EUR;
                        DataLiq.Clab_EUR(i) = configuration.config(2).Clab_EUR;
                        DataLiq.CROW_EUR(i) = configuration.config(2).CROW_EUR;
                        DataLiq.Cmisc_EUR(i) = configuration.config(2).Cmisc_EUR;
                        DataLiq.Ipipe_EUR(i) = configuration.config(2).Ipipe_EUR;
                        DataLiq.OCpipe_EUR_per_y(i) = configuration.config(2).OCpipe_EUR_per_y;
                        DataLiq.Icomp_EUR(i) = configuration.config(2).Icomp_EUR;
                        DataLiq.OCcomp_EUR_per_y(i) = configuration.config(2).OCcomp_EUR_per_y;
                        DataLiq.ECcomp_EUR_per_y(i) = configuration.config(2).ECcomp_EUR_per_y;
                        DataLiq.Ipump_EUR(i) = configuration.config(2).Ipump_EUR;
                        DataLiq.OCpump_EUR_per_y(i) = configuration.config(2).OCpump_EUR_per_y;
                        DataLiq.ECpump_EUR_per_y(i) = configuration.config(2).ECpump_EUR_per_y;
                        DataLiq.LCcomp_EUR_per_t(i) = configuration.config(2).LCcomp_EUR_per_t;
                end
        
        end


    end

    Terrain(terrain).Phase(1).Data = DataGas;
    Terrain(terrain).Phase(2).Data = DataLiq;
end

end