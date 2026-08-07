function [ship, finance] = CostsShipBulk_old(ship, conditioning, temporaryStorage, vesselLoadingStation, finance, i, n_scen, n_work)
%This function calculates the costs for the ship batch transport (isotainer)
%INPUT: ship = struct containing the ship information
%       conditioning = struct containing the conditioning information
%       finance = struct containing the finance information
%       i = plant index
%       n_scen = number of scenarios
%       n_work = number of working days scenarios
%OUTPUT: ship = struct containing the ship information
%           C_isotainer_buy_CHF_per_y = yearly cost for isotainers [CHF/y]
%           CAPEX_tot_CHF = total CAPEX [CHF]
%           C_isotainer_rent_CHF_per_y = yearly cost for renting isotainers [CHF/y]
%           C_transport_ship_CHF_per_y = yearly cost for transport by barge [CHF/y]
%           C_customs_CHF_per_y = yearly cost for customs per year [CHF/y]
%           C_arr_not_CHF_per_y = yearly cost for arrival notification [CHF/y]
%           C_dangerous_goods_CHF_per_y = yearly cost for dangerous goods supplement [CHF/y]
%           OPEX_0_CHF_per_y = OPEX [CHF/y]
%           year_costs_0_CHF_per_y = yearly costs [CHF/y]
%           levelized_costs_0_CHF_per_t = levelized_costs_0; %[CHF/t]
%           CostMatrix = cost matrix
%           TextLegend = string containing the categories of costs

%% Data

m_liq_t_per_y = conditioning.capture(i).m_liq_t_per_y; %[t/y]

r = finance.r;

for st = 1:size(ship.start,2)
    for go = 1:size(ship.start(st).goal,2)

%         n_shipment = ship.capture(i).start(st).goal(go).n_shipment;
%         n_journey = ship.capture(i).start(st).goal(go).n_journey;
        d_transport_km = ship.start(st).goal(go).d_transport_km; %[km]
        C_transport_CHF_per_t_per_km = ship.C_transport_CHF_per_t_per_km; %[CHF/t/km]
%         C_transshipment_start_CHF_per_isotainer = ship.start(st).C_transshipment_CHF_per_isotainer; %[CHF]
%         C_transshipment_goal_CHF_per_isotainer = ship.start(st).goal(go).C_transshipment_CHF_per_isotainer; %[CHF]
        
        %% CAPEX
        
        finance.vesselLoadingStation.a_CAPEX = r./(1-(1+r).^-vesselLoadingStation.lifetime);
        n_loadingStations = ship.capture(i).start(st).goal(go).n_loadingStations;
        
        C_temporaryStorage_CAPEX_CHF_per_y = temporaryStorage.capture(i).costs.CAPEX_CHF_per_y; %[CHF/y]
        
        C_vesselLoadingStation_CAPEX_CHF_per_y = vesselLoadingStation.CAPEX_0_CHF.*...
            finance.vesselLoadingStation.a_CAPEX.*n_loadingStations; %[CHF/y]
        
        CAPEX_tot = temporaryStorage.capture(i).costs.CAPEX_CHF + vesselLoadingStation.CAPEX_0_CHF; %[CHF]
        CAPEX_CHF_per_y =  C_temporaryStorage_CAPEX_CHF_per_y + C_vesselLoadingStation_CAPEX_CHF_per_y; %[CHF/y]

        %% OPEX
        
        C_vesselLoadingStation_OPEX = vesselLoadingStation.n_personnel.*...
            vesselLoadingStation.C_salary_CHF_per_y.*n_loadingStations; %[CHF/y] missing: energy
        
        C_transport_ship = d_transport_km.*m_liq_t_per_y.*C_transport_CHF_per_t_per_km; %[CHF/y]
        
        OPEX_0 = C_transport_ship + C_vesselLoadingStation_OPEX; %[CHF/y]
            

         %% Yearly costs

        year_costs_0 = CAPEX_CHF_per_y + OPEX_0; %[CHF/y]

        levelized_costs_0 = year_costs_0./m_liq_t_per_y; %[CHF/t]

        %% Rename and resize matrices

        ship.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
        ship.capture(i).start(st).goal(go).costs.CAPEX_CHF_per_y = CAPEX_CHF_per_y; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_temporaryStorage_CHF_per_y = C_temporaryStorage_CAPEX_CHF_per_y; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_vesselLoadingStation_CHF_per_y = C_vesselLoadingStation_CAPEX_CHF_per_y + C_vesselLoadingStation_OPEX; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_transport_ship_CHF_per_y = C_transport_ship; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.OPEX_0_CHF_per_y = OPEX_0; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.year_costs_0_CHF_per_y = year_costs_0; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.levelized_costs_0_CHF_per_t = levelized_costs_0; %[CHF/t]

        fn = fieldnames(ship.capture(i).start(st).goal(go).costs);

        for j = 1:numel(fn)
            if size(ship.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(ship.capture(i).start(st).goal(go).costs.(fn{j}),2) == 3
                ship.capture(i).start(st).goal(go).costs.(fn{j}) = ship.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,1);
            elseif size(ship.capture(i).start(st).goal(go).costs.(fn{j}),1) == 3 && size(ship.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                ship.capture(i).start(st).goal(go).costs.(fn{j}) = ship.capture(i).start(st).goal(go).costs.(fn{j}).*ones(1,3);
            elseif size(ship.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(ship.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                ship.capture(i).start(st).goal(go).costs.(fn{j}) = ship.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,3);
            end
        end
        
        % Prepare matrix for plots
        ship.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,3,n_work);
        for l = 1:n_work
            %sprintf('%5d',[i j st go l])
            ship.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [ship.capture(i).start(st).goal(go).costs.C_transport_ship_CHF_per_y(l,:)', ...
                ship.capture(i).start(st).goal(go).costs.C_temporaryStorage_CHF_per_y(l,:)',...
                ship.capture(i).start(st).goal(go).costs.C_vesselLoadingStation_CHF_per_y(l,:)'];
        end
        ship.capture(i).start(st).goal(go).costs.TextLegend = {'Transport','Intermediary storage','Loading station'};
        %ship.capture(i).start(st).goal(go).costs.TextXTickLabel = {'Optimistic','Average','Conservative'};

    end
end

end

