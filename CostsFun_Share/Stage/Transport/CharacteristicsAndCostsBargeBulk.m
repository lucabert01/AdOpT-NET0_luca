function [barge, finance] = CharacteristicsAndCostsBargeBulk(barge, conditioning, temporaryStorage, vesselLoadingStation, finance, i, n_scen, n_work)
%This function calculates the costs for the barge bulk transport
%INPUT: barge = struct containing the barge information
%       conditioning = struct containing the conditioning information
%       temporaryStorage = struct containing the information about temporary storage
%       vesselLoadingStation = struct containing the informaiton about the vessels loading stations
%       finance = struct containing the finance information
%       i = plant index
%       n_scen = number of scenarios
%       n_work = number of working days scenarios
%OUTPUT: barge = struct containing the barge information
%        Vbarge_m3 = selected barge volume [m3]
%        mLoadAvg_t = average load of the barge over the year [t]
%        mBarge_t = weight of the empty barge [t]
%        n_shipment = number of shipments per year
%        f_per_y = frequency of transportation per year
%        nBarge = number of barges
%        CAPEX_barge_EUR_per_y = yearly CAPEX costs for the barge [EUR.y-1]
%        OPEX_harbour_EUR_per_y = harbour costs per year [EUR.y-1]
%        OPEX_gasoil_EUR_per_y = oil costs per year [EUR.y-1]
%        AIC_CHF_per_y = annualized investment costs [CHF/y]
%        AOC_CHF_per_y = annualized operating costs [CHF/y]
%        TAC_CHF_per_y = total annualized costs [CHF/y]
%        LC_CHF_per_t = levelized costs [CHF/t]
%        CostMatrix = cost matrix
%        TextLegend = string containing the categories of costs

%% Data

m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2); %[t/y]
r = finance.r;

for st = 1:size(barge.start,2)
    for go = 1:size(barge.start(st).goal,2)

        d_km = barge.start(st).goal(go).d_km; %[km]
        Vmax_m3 = barge.start(st).goal(go).Vmax_m3; %[m3]

        t_transport_h = d_km./barge.v_km_per_h; %[h]
        t_roundtrip_h = barge.t_load_h + 2.*t_transport_h + barge.t_unload_h; %[h]
        
        emissions_t_per_t = d_km.*barge.gamma_kg_per_t_per_km./1000; %[tCO2/tCO2transp]
        leakage_t = d_km.*barge.lambda_t_per_km; %[tCO2]
        leakage_t_per_t = leakage_t./m_liq_t_per_y; %[tCO2/tCO2transp]
        
        %Write the outcomes in the right place
        barge.start(st).goal(go).t_transport_h = t_transport_h;
        barge.start(st).goal(go).t_roundtrip_h = t_roundtrip_h; %[h]
        barge.capture(i).start(st).goal(go).t_transport_h = t_transport_h;
        barge.capture(i).start(st).goal(go).t_roundtrip_h = t_roundtrip_h; %[h]
        barge.capture(i).start(st).goal(go).emissions_t_per_t = emissions_t_per_t;
        barge.capture(i).start(st).goal(go).leakage_t = leakage_t;
        barge.capture(i).start(st).goal(go).leakage_t_per_t = leakage_t_per_t;
        barge.capture(i).start(st).goal(go).n_isotainer = 0;

        [Vbarge_m3, mLoadAvg_t, mBarge_t, mTot_t, n_shipments, f_per_y, ...
    nBarge, CAPEX_barge_EUR_per_y, OPEX_harbour_EUR_per_y, OPEX_gasoil_EUR_per_y, ...
    TYC_EUR_per_y, LC_EUR_per_t] = BargeBulkCosts(barge, m_liq_t_per_y, d_km, Vmax_m3, t_roundtrip_h);
       
        n_loadingStations = ceil(n_shipments/vesselLoadingStation.n_loadings);
        
        barge.capture(i).start(st).goal(go).n_shipment = n_shipments;
        barge.capture(i).start(st).goal(go).n_journey = n_shipments;
        barge.capture(i).start(st).goal(go).n_loadingStations = n_loadingStations;       
        barge.capture(i).start(st).goal(go).f_per_y = f_per_y;
        barge.capture(i).start(st).goal(go).mMax_t = mLoadAvg_t;
        barge.capture(i).start(st).goal(go).nBarge = nBarge;
        barge.capture(i).start(st).goal(go).Vbarge_m3 = Vbarge_m3;
        barge.capture(i).start(st).goal(go).mBarge_t = mBarge_t;      
        
        %% CAPEX

        finance.vesselLoadingStation.a_CAPEX = r./(1-(1+r).^-vesselLoadingStation.lifetime);
        n_loadingStations = barge.capture(i).start(st).goal(go).n_loadingStations;
        
        C_temporaryStorage_CAPEX_CHF_per_y = temporaryStorage.capture(i).costs.AIC_CHF_per_y; %[CHF/y]
        
        C_vesselLoadingStation_CAPEX_CHF_per_y = vesselLoadingStation.CAPEX_0_CHF.*...
            finance.vesselLoadingStation.a_CAPEX.*n_loadingStations; %[CHF/y]
        
%         CAPEX_tot = temporaryStorage.capture(i).costs.CAPEX_CHF + vesselLoadingStation.CAPEX_0_CHF; %[CHF]
        AIC_CHF_per_y =  CAPEX_barge_EUR_per_y.*finance.x_EURCHF + C_temporaryStorage_CAPEX_CHF_per_y + C_vesselLoadingStation_CAPEX_CHF_per_y; %[CHF/y]

        %% OPEX

        C_vesselLoadingStation_OPEX = vesselLoadingStation.n_personnel.*...
            vesselLoadingStation.C_salary_CHF_per_y.*n_loadingStations; %[CHF/y] missing: energy

        C_temporaryStorage_OPEX = temporaryStorage.capture(i).costs.AOC_CHF_per_y; %[CHF/y]
        
        AOC_CHF_per_y = C_vesselLoadingStation_OPEX + C_temporaryStorage_OPEX + (OPEX_harbour_EUR_per_y + OPEX_gasoil_EUR_per_y).*finance.x_EURCHF; %[CHF/y]

         %% Yearly costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]
        
        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t]

        %% Rename and resize matrices

%         barge.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
        barge.capture(i).start(st).goal(go).costs.AIC_CHF_per_y = AIC_CHF_per_y; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t]

        fn = fieldnames(barge.capture(i).start(st).goal(go).costs);

        for j = 1:numel(fn)
            switch isa(barge.capture(i).start(st).goal(go).costs.(fn{j}),'double')
                case 1
                    if size(barge.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(barge.capture(i).start(st).goal(go).costs.(fn{j}),2) == 3
                        barge.capture(i).start(st).goal(go).costs.(fn{j}) = barge.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,1);
                    elseif size(barge.capture(i).start(st).goal(go).costs.(fn{j}),1) == 3 && size(barge.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                        barge.capture(i).start(st).goal(go).costs.(fn{j}) = barge.capture(i).start(st).goal(go).costs.(fn{j}).*ones(1,3);
                    elseif size(barge.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(barge.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                        barge.capture(i).start(st).goal(go).costs.(fn{j}) = barge.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,3);
                    end
            end
        end
        
        % Prepare matrix for plots
        barge.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,1,n_work);
        for l = 1:n_work
            barge.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                barge.capture(i).start(st).goal(go).costs.LC_CHF_per_t(l,:)';
        end
        barge.capture(i).start(st).goal(go).costs.TextLegend = {'Total'};
    end
end

end

