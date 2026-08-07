function [train, finance] = CostsTrainBatch(train, conditioning, isotainer, finance, i, n_scen, n_work)
%This function calculates the costs for the train batch transport (isotainer)
%INPUT: train = struct containing the train information
%       conditioning = struct containing the conditioning information
%       isotainer = struct containing the isotainer information
%       finance = struct containing the finance information
%       i = plant index
%       n_scen = number of scenarios
%       n_work = number of working days scenarios
%OUTPUT: train = struct containing the train information
%           C_isotainer_buy_CHF_per_y = yearly cost for isotainers [CHF/y]
%           CAPEX_tot_CHF = total CAPEX [CHF]
%           C_isotainer_rent_CHF_per_y = yearly cost for renting isotainers [CHF/y]
%           C_transshipment_CHF_per_y = yearly cost for transshipment [CHF/y]
%           C_customs_CHF_per_y = yearly cost for customs per year [CHF/y]
%           C_weigh_CHF_per_y = yearly cost for weighing containers per year [CHF/y]
%           C_transport_CHF_per_y = yearly cost for transport by train [CHF/y]
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]
%           CostMatrix = cost matrix
%           TextLegend = string containing the categories of costs

%% Data

m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2); %[t/y]
C_transshipment_CHF_per_isotainer = train.C_transshipment_CHF_per_isotainer; %[CHF]
r = finance.r;

for st = 1:size(train.start,2)
    for go = 1:size(train.start(st).goal,2)

%         C_isotainer_CHF_per_t = train.start(st).goal(go).C_isotainer_CHF_per_t; %[CHF/t]
        n_isotainer = train.capture(i).start(st).goal(go).n_isotainer;
        n_shipment = train.capture(i).start(st).goal(go).n_shipment;
%         t_roundtrip_train_h = train.start(st).goal(go).t_roundtrip_train_h; %[h]
        d_km = train.start(st).goal(go).d_km; %[km]
        b_custom = train.start(st).goal(go).b_custom;
        f_per_y = train.start(st).goal(go).f_per_y;

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

        C_transshipment = n_shipment.*C_transshipment_CHF_per_isotainer; %[CHF/y]
%         C_wagon_rent = n_transshipment.*ceil(t_roundtrip_train_h./24).*train.C_wagon_4axles; %[CHF/y]
        C_customs = b_custom.*f_per_y.*train.C_custom; %[CHF/y]
%         C_transport = n_transshipment.*isotainer.m_isotainer_t.*C_isotainer_CHF_per_t; %[CHF/y]
        C_weigh = n_shipment.*train.C_weight_CHF_per_container; %[CHF/y]
        C_transport = n_shipment.*(d_km.*train.C_transport_CHF_per_isotainer_per_km + train.C_transport_base_CHF_per_isotainer); %[CHF/y]
%        C_isotainer_rent = isotainer.C_isotainer_CHF_per_y.*n_isotainer; %[CHF/y]

        %Total costs
        switch isotainer.choice
            case 'rent'
                AOC_CHF_per_y = C_transshipment + C_customs + ...
                    C_weigh + C_transport + ...
                    C_isotainer_rent; %[CHF/y]
            case 'buy'
                AOC_CHF_per_y = C_transshipment + C_customs + ...
                    C_weigh + C_transport; %[CHF/y]
        end

        %% Yearly costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]

        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t]

        %% Rename and resize matrices

        train.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y = C_isotainer_buy;
        train.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
%        train.capture(i).start(st).goal(go).costs.C_isotainer_rent_CHF_per_y = C_isotainer_rent; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_transshipment_CHF_per_y = C_transshipment; %[CHF/y]
%         train.capture(i).start(st).goal(go).costs.C_wagon_rent_CHF_per_y = C_wagon_rent; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y = C_customs; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_weigh_CHF_per_y = C_weigh; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_transport_CHF_per_y = C_transport; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t]

        fn = fieldnames(train.capture(i).start(st).goal(go).costs);

        for j = 1:numel(fn)
            if size(train.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(train.capture(i).start(st).goal(go).costs.(fn{j}),2) == 3
                train.capture(i).start(st).goal(go).costs.(fn{j}) = train.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,1);
            elseif size(train.capture(i).start(st).goal(go).costs.(fn{j}),1) == 3 && size(train.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                train.capture(i).start(st).goal(go).costs.(fn{j}) = train.capture(i).start(st).goal(go).costs.(fn{j}).*ones(1,3);
            elseif size(train.capture(i).start(st).goal(go).costs.(fn{j}),1) == 1 && size(train.capture(i).start(st).goal(go).costs.(fn{j}),2) == 1
                train.capture(i).start(st).goal(go).costs.(fn{j}) = train.capture(i).start(st).goal(go).costs.(fn{j}).*ones(3,3);
            end
        end
        
        % Prepare matrix for plots
        train.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,5,n_work);
        for l = 1:n_work
            switch isotainer.choice
                case 'rent'
                    C_isotainer_CHF_per_y = train.capture(i).start(st).goal(go).costs.C_isotainer_rent_CHF_per_y(l,:)';
                case 'buy'
                    C_isotainer_CHF_per_y = train.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y(l,:)';
            end
            train.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [C_isotainer_CHF_per_y, ...
                train.capture(i).start(st).goal(go).costs.C_transshipment_CHF_per_y(l,:)', ...
                train.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y(l,:)', ...
                train.capture(i).start(st).goal(go).costs.C_weigh_CHF_per_y(l,:)', ...
                train.capture(i).start(st).goal(go).costs.C_transport_CHF_per_y(l,:)'];
        end
        train.capture(i).start(st).goal(go).costs.TextLegend = {'Isotainer',...
            'Transshipment', 'Customs', 'Weighing', 'Train transport'};
          

    end
end

end

