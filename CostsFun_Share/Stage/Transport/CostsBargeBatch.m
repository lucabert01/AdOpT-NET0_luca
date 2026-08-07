function [barge, finance] = CostsBargeBatch(barge, conditioning, isotainer, finance, i, n_scen, n_work)
%This function calculates the costs for the barge batch transport (isotainer)
%INPUT: barge = struct containing the barge information
%       conditioning = struct containing the conditioning information
%       isotainer = struct containing the isotainer information
%       finance = struct containing the finance information
%       i = plant index
%       n_scen = number of scenarios
%       n_work = number of working days scenarios
%OUTPUT: barge = struct containing the barge information
%           C_isotainer_buy_CHF_per_y = yearly cost for isotainers [CHF/y]
%           CAPEX_tot_CHF = total CAPEX [CHF]
%           C_isotainer_rent_CHF_per_y = yearly cost for renting isotainers [CHF/y]
%           C_transshipment_CHF_per_y = yearly cost for transshipment [CHF/y]
%           C_transport_barge_CHF_per_y = yearly cost for transport by barge [CHF/y]
%           C_lowWater_CHF_per_y = yearly cost for low water supplement [CHF/y]
%           C_danger_CHF_per_y = yearly cost for dangerous goods supplement [CHF/y]
%           C_customs_CHF_per_y = yearly cost for customs per year [CHF/y]
%           C_congestion_CHF_per_y = yearly cost for congestion [CHF/y]
%           C_weight_CHF_per_y = yearly cost for weighing containers per year [CHF/y]
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]
%           CostMatrix = cost matrix
%           TextLegend = string containing the categories of costs

%% Data

m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2); %[t/y]
r = finance.r;

for st = 1:size(barge.start,2)
    for go = 1:size(barge.start(st).goal,2)

        n_isotainer = barge.capture(i).start(st).goal(go).n_isotainer;
        n_shipment = barge.capture(i).start(st).goal(go).n_shipment;
        n_journey = barge.capture(i).start(st).goal(go).n_journey;
        %d_transport_km = barge.start(st).goal(go).d_transport_km; %[km]
        C_transport_CHF_per_isotainer = barge.start(st).goal(go).C_transport_CHF_per_isotainer;
        C_lowWater_CHF_per_isotainer = barge.start(st).C_lowWater_CHF_per_isotainer;

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

        C_transshipment = 4.*n_shipment.*barge.C_transshipment_CHF_per_isotainer; %[CHF/y]
%         C_isotainer_rent = isotainer.C_isotainer_CHF_per_y.*n_isotainer; %[CHF/y]
%         C_transport_barge = barge.C_barge_CHF_per_t_per_km.*n_transshipment.*isotainer.m_isotainer_t.*d_transport_km; %[CHF/y]
        C_transport_barge = n_shipment.*C_transport_CHF_per_isotainer; %[CHF/y]
        C_lowWaterSupplement = n_shipment.*C_lowWater_CHF_per_isotainer; %[CHF/y]
        C_customs = n_journey.*barge.C_customs_CHF_per_shipment; %[CHF/y]
        C_dangerous_goods = n_shipment.*barge.C_dangerous_goods_CHF_per_isotainer; %[CHF/y]
        C_congestion = n_shipment.*2.*barge.C_congestion_CHF_per_direction; %[CHF/y]
        C_weight = n_shipment.*barge.C_weight_CHF_per_container; %[CHF/y]

        %Total costs
        switch isotainer.choice
            case 'rent'
                AOC_CHF_per_y = C_transshipment + C_transport_barge + C_lowWaterSupplement + ...
                    C_dangerous_goods + C_customs + C_congestion + C_weight +...
                    C_isotainer_rent; %[CHF/y]
            case 'buy'
                AOC_CHF_per_y = C_transshipment + C_transport_barge + C_lowWaterSupplement + ...
                    C_dangerous_goods + C_customs + C_congestion + C_weight; %[CHF/y]
        end

         %% Yearly costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]
        
        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t]

        %% Rename and resize matrices

        barge.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y = C_isotainer_buy;
        barge.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
%        barge.capture(i).start(st).goal(go).costs.C_isotainer_rent_CHF_per_y = C_isotainer_rent; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.C_transshipment_CHF_per_y = C_transshipment; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.C_transport_barge_CHF_per_y = C_transport_barge; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.C_lowWater_CHF_per_y = C_lowWaterSupplement ; %[CHF/y] 
        barge.capture(i).start(st).goal(go).costs.C_danger_CHF_per_y = C_dangerous_goods; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y = C_customs; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.C_congestion_CHF_per_y = C_congestion; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.C_weight_CHF_per_y = C_weight; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        barge.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t]

        fn = fieldnames(barge.capture(i).start(st).goal(go).costs);

        for j = 1:numel(fn)
            if size(barge.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(barge.capture(i).start(st).goal(go).costs.(fn{j}),2) == 3
                barge.capture(i).start(st).goal(go).costs.(fn{j}) = barge.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,1);
            elseif size(barge.capture(i).start(st).goal(go).costs.(fn{j}),1) == 3 && size(barge.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                barge.capture(i).start(st).goal(go).costs.(fn{j}) = barge.capture(i).start(st).goal(go).costs.(fn{j}).*ones(1,3);
            elseif size(barge.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(barge.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                barge.capture(i).start(st).goal(go).costs.(fn{j}) = barge.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,3);
            end
        end
        
        % Prepare matrix for plots
        barge.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,8,n_work);
        for l = 1:n_work
            switch isotainer.choice
                case 'rent'
                    C_isotainer_CHF_per_y = barge.capture(i).start(st).goal(go).costs.C_isotainer_rent_CHF_per_y(l,:)';
                case 'buy'
                    C_isotainer_CHF_per_y = barge.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y(l,:)';
            end
            barge.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [C_isotainer_CHF_per_y, ...
                barge.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y(l,:)', ...
                barge.capture(i).start(st).goal(go).costs.C_congestion_CHF_per_y(l,:)', ...
                barge.capture(i).start(st).goal(go).costs.C_weight_CHF_per_y(l,:)', ...
                barge.capture(i).start(st).goal(go).costs.C_transshipment_CHF_per_y(l,:)', ...
                barge.capture(i).start(st).goal(go).costs.C_danger_CHF_per_y(l,:)', ...
                barge.capture(i).start(st).goal(go).costs.C_lowWater_CHF_per_y(l,:)', ...
                barge.capture(i).start(st).goal(go).costs.C_transport_barge_CHF_per_y(l,:)'];
        end
        barge.capture(i).start(st).goal(go).costs.TextLegend = {'Isotainer','Customs','Congestion','Weighing','Transshipment','Dangerous goods','Low water','Barge transport'};
    end
end

end

