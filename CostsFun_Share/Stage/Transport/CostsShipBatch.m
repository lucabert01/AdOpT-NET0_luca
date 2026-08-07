function [ship, finance] = CostsShipBatch(ship, conditioning, isotainer, finance, i, n_scen, n_work)
%This function calculates the costs for the ship batch transport (isotainer)
%INPUT: ship = struct containing the ship information
%       conditioning = struct containing the conditioning information
%       isotainer = struct containing the isotainer information
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
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]
%           CostMatrix = cost matrix
%           TextLegend = string containing the categories of costs

%% Data

m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2); %[t/y]
r = finance.r;

for st = 1:size(ship.start,2)
    for go = 1:size(ship.start(st).goal,2)

        n_isotainer = ship.capture(i).start(st).goal(go).n_isotainer;
        n_shipment = ship.capture(i).start(st).goal(go).n_shipment;
        n_journey = ship.capture(i).start(st).goal(go).n_journey;
%         d_transport_km = ship.start(st).goal(go).d_transport_km; %[km]
%         C_transshipment_start_CHF_per_isotainer = ship.start(st).C_transshipment_CHF_per_isotainer; %[CHF]
%         C_transshipment_goal_CHF_per_isotainer = ship.start(st).goal(go).C_transshipment_CHF_per_isotainer; %[CHF]
        
        %% CAPEX

finance.isotainer.a_isotainer = r./(1-(1+r).^-isotainer.lifetime);

        C_isotainer_buy = isotainer.C_1isotainer_CHF.*n_isotainer.*finance.isotainer.a_isotainer; %[CHF/y]

        switch isotainer.choice
            case 'rent'
                CAPEX_tot = 0; %[CHF]
                AIC_CHF_per_y = 0; %[CHF/y]
            case 'buy'
                CAPEX_tot = isotainer.C_1isotainer_CHF.*n_isotainer; %[CHF]
                AIC_CHF_per_y = C_isotainer_buy; %[CHF/y]
        end

        %% OPEX

%         C_transshipment = n_transshipment.*(C_transshipment_start_CHF_per_isotainer + C_transshipment_goal_CHF_per_isotainer); %[CHF/y]
%         C_isotainer_rent = isotainer.C_isotainer_CHF_per_y.*n_isotainer; %[CHF/y]
        C_transport_ship = n_shipment.*ship.start(st).goal(go).C_transport_CHF_per_isotainer; %[CHF/y]
        C_customs = n_journey.*(ship.C_import_customs_CHF_per_shipment + ship.C_export_customs_CHF_per_shipment); %[CHF/y]
        C_arr_not = n_journey.*ship.C_arrival_notification_CHF_per_shipment; %[CHF/y]
        C_dangerous_goods = n_shipment.*ship.C_dangerous_goods_CHF_per_isotainer; %[CHF/y]

        %Total costs
        switch isotainer.choice
            case 'rent'
                AOC_CHF_per_y = C_transport_ship + C_customs + C_arr_not + C_dangerous_goods + ...
                    C_isotainer_rent; %[CHF/y]
            case 'buy'
                AOC_CHF_per_y = C_transport_ship + C_customs + C_arr_not + C_dangerous_goods; %[CHF/y]
        end

         %% Yearly costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]

        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t]

        %% Rename and resize matrices

        ship.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y = C_isotainer_buy;
        ship.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
%         ship.capture(i).start(st).goal(go).costs.C_isotainer_rent_CHF_per_y = C_isotainer_rent; %[CHF/y]
%         ship.capture(i).start(st).goal(go).costs.C_transshipment_CHF_per_y = C_transshipment; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_transport_ship_CHF_per_y = C_transport_ship; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y = C_customs; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_arr_not_CHF_per_y = C_arr_not; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_dangerous_goods_CHF_per_y = C_dangerous_goods; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t]

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
        ship.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,5,n_work);
        for l = 1:n_work
            %sprintf('%5d',[i j st go l])
            switch isotainer.choice
                case 'rent'
                    C_isotainer_CHF_per_y = ship.capture(i).start(st).goal(go).costs.C_isotainer_rent_CHF_per_y(l,:)';
                case 'buy'
                    C_isotainer_CHF_per_y = ship.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y(l,:)';
            end
            ship.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [ship.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y(l,:)', ...
                ship.capture(i).start(st).goal(go).costs.C_arr_not_CHF_per_y(l,:)',...
                ship.capture(i).start(st).goal(go).costs.C_dangerous_goods_CHF_per_y(l,:)', ...
                ship.capture(i).start(st).goal(go).costs.C_transport_ship_CHF_per_y(l,:)', ...
                C_isotainer_CHF_per_y];
        end
        ship.capture(i).start(st).goal(go).costs.TextLegend = {'Customs','Arrival notification','Dangerous goods','Ship transport','Isotainer'};
        %ship.capture(i).start(st).goal(go).costs.TextXTickLabel = {'Optimistic','Average','Conservative'};

    end
end

end

