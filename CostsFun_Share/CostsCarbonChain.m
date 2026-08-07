function [CostsCarbonStored, CostsCarbonAvoided, CO2AvoidanceEfficiency, ...
    Conditioning] = CostsCarbonChain(plant, conditioning, temporaryStorage, ...
    isotainer, storage, Emissions, Leakages, TransportCosts, i, st, b_tempStorage, ...
    b_isotainer, idx_trspt, idx_pipelineGas, idx_buildNetwork, n_work, n_scen)
%This function calculates the levelized costs of carbon avoided and the CO2
%avoidance efficiency from the chain
%INPUT: plant = struct containing the plant information
%       conditioning = struct containing the conditioning information
%       temporaryStorage = struct containing the temporary storage information
%       isotainer = struct containing the isotainer information
%       storage = struct containing the storage information
%       Emissions = transport emissions of the chain [tCO2-eq/tCO2transp]
%       TransportCosts = transport costs of the chain, each column standing for one connection [CHF/t]
%       i = plant index
%       st = storage site index
%       b_tempStorage c
%       b_isotainer = booolean expressing if there is/are isotainers in the chain
%       idx_trspt = array with indices of transport modes of a chain
%       idx_pipelineGas = index at which there is the gaseous phase pipeline struct within the Network struct
%       idx_buildNetwork = index at which there is the buildNetwork within the Network struct
%       n_work = number of working days scenarios
%       n_scen = number of scenarios
%OUTPUT: CostsCarbonStored = levelized costs of stored carbon [CHF/tCO2stored]
%        CostsCarbonAvoided = levelized costs of avoided carbon [CHF/tCO2avoided]
%        CO2AvoidanceEfficiency = avoidance efficiency of CO2 from the conditioning to the storage [-]

%% Efficiency and emissions

%Conditioning needed: compression or liquefaction
not_only_gas_pipeline = idx_trspt ~= 10 & idx_trspt ~= 12;

switch any(not_only_gas_pipeline)
    case 0
        Conditioning = 'Compression';
        eta_conditioning = conditioning.eta_CO2_compression;
        gamma_conditioning = conditioning.capture(i).compression_GasOn.gamma_comp_t_per_t;
        C_conditioning_CHF_per_y = conditioning.capture(i).compression_GasOn.costs.TAC_CHF_per_y; %[CHF/y]
    case 1
        only_low_pressure_transport = all(idx_trspt(not_only_gas_pipeline) == 9);
        switch only_low_pressure_transport
            case 0
                Conditioning = 'Liquefaction 16 bar';
                eta_conditioning = conditioning.eta_CO2_liquefaction_15barg(2);
                gamma_conditioning = conditioning.capture(i).liquefaction15barg.gamma_liq_t_per_t;
                C_conditioning_CHF_per_y = conditioning.capture(i).liquefaction15barg.costs.TAC_CHF_per_y; %[CHF/y]
            case 1
                Conditioning = 'Liquefaction 8 bar';
                eta_conditioning = conditioning.eta_CO2_liquefaction_7barg(2);
                gamma_conditioning = conditioning.capture(i).liquefaction7barg.gamma_liq_t_per_t;
                C_conditioning_CHF_per_y = conditioning.capture(i).liquefaction7barg.costs.TAC_CHF_per_y; %[CHF/y]
        end
end

%Emptying isotainer or not
% switch b_isotainer
%     case 0 
%         eta_empty = 1; %[-]
%     case 1
%         eta_empty = isotainer.eta_CO2_emptyIsotainer; %[-]
% end

%We assume steady-state --> the fact that we do not empty the isotainer
%only leads to a reduced capacity, but not a loss of CO2
eta_empty = 1; %[-]

eta_tempStorage = temporaryStorage.eta_tempStorage; %[-]
eta_storage = storage.eta_storage; %[%]
% Emissions = Emissions; %[tCO2/ttransported]
% leakages_transport = Leakages; %[tCO2/ttransported]

%% Netto mass avoided

m_capt_t_per_y = plant.capture(i).m_capt_t_per_y; %[t/y]
m_cond_t_per_y = m_capt_t_per_y.*eta_conditioning;

switch b_tempStorage
    case 0
        CostsCarbonChain = zeros(n_scen,length(idx_trspt)+3,n_work);
        CostsCarbonStored = zeros(n_scen,length(idx_trspt)+3,n_work);
        CostsCarbonAvoided = zeros(n_scen,length(idx_trspt)+3,n_work);
        m_transport_t_per_y = m_cond_t_per_y;
%         CO2AvoidanceEfficiency = eta_conditioning*(1-Emissions-Leakages).*eta_empty.*eta_storage;
        CO2AvoidanceEfficiency = eta_conditioning*(1-Emissions).*eta_empty.*eta_storage; %Leakages are now included in Emissions
    case 1
        CostsCarbonChain = zeros(n_scen,length(idx_trspt)+4,n_work);
        CostsCarbonStored = zeros(n_scen,length(idx_trspt)+4,n_work);
        CostsCarbonAvoided = zeros(n_scen,length(idx_trspt)+4,n_work);
        m_transport_t_per_y = m_cond_t_per_y.*eta_tempStorage;
%         CO2AvoidanceEfficiency = eta_conditioning*(1-Emissions-Leakages).*eta_empty.*eta_tempStorage.*eta_storage;
        CO2AvoidanceEfficiency = eta_conditioning*(1-Emissions).*eta_empty.*eta_tempStorage.*eta_storage; %Leakages are now included in Emissions
end

m_stored_t_per_y = m_transport_t_per_y.*(1-Leakages).*eta_empty.*eta_storage;
m_emitted_t_per_y = m_cond_t_per_y.*gamma_conditioning + ...
    m_transport_t_per_y.*Emissions;

m_avoided_t_per_y = m_stored_t_per_y - m_emitted_t_per_y;

%% Total costs

for m = 1:n_work
    C_capture_CHF_per_y = plant.capture(i).costs.TAC_CHF_per_y; %[CHF/y]
    C_tempStorage_CHF_per_y = temporaryStorage.capture(i).costs.TAC_CHF_per_y; %[CHF/y]
    C_transport_CHF_per_y = m_transport_t_per_y.*TransportCosts(:,:,m); %[CHF/y]
    C_storage_CHF_per_y = m_stored_t_per_y.*storage.capture(i).start(st).goal.costs.LC_CHF_per_t; %[CHF/y]
    
    %% Levelized costs of carbon avoided
    
    switch b_tempStorage
        case 0
            CostsCarbonChain(:,:,m) = [C_capture_CHF_per_y.*ones(1,3)', ...
                C_conditioning_CHF_per_y', ...
                C_transport_CHF_per_y, ...
                C_storage_CHF_per_y']; %[CHF/y]
        case 1
            CostsCarbonChain(:,:,m) = [C_capture_CHF_per_y.*ones(1,3)', ...
                C_conditioning_CHF_per_y', ...
                C_tempStorage_CHF_per_y', ...
                C_transport_CHF_per_y, ...
                C_storage_CHF_per_y']; %[CHF/y]
    end
    
    CostsCarbonStored(:,:,m) = CostsCarbonChain(:,:,m)./m_stored_t_per_y'; %[CHF/t]
    CostsCarbonAvoided(:,:,m) = CostsCarbonChain(:,:,m)./m_avoided_t_per_y'; %[CHF/t]
    
end

end

