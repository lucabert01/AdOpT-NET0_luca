function [train, finance] = CostsTrainBulk(train, conditioning, temporaryStorage, filling_station, finance, i, n_scen, n_work)
%This function calculates the costs for the train bulk transport (RTC)
%INPUT: train = struct containing the train information
%       conditioning = struct containing the conditioning information
%       isotainer = struct containing the isotainer information
%       finance = struct containing the finance information
%       i = plant index
%       n_scen = number of scenarios
%       n_work = number of working days scenarios
%OUTPUT: train = struct containing the train information
%           C_filling_station_CHF_per_y = yearly costs for a filling station [CHF/y]
%           C_wagon_rent_CHF_per_y = yearly costs for renting wagons [CHF/y]
%           C_customs_CHF_per_y = yearly cost for customs per year [CHF/y]
%           C_isotainer_buy_CHF_per_y = yearly cost for isotainers [CHF/y]
%           CAPEX_tot_CHF = total CAPEX [CHF]
%           C_isotainer_rent_CHF_per_y = yearly cost for renting isotainers [CHF/y]
%           C_transshipment_CHF_per_y = yearly cost for transshipment [CHF/y]
%           C_transport_CHF_per_y = yearly cost for transport by train [CHF/y]
%           C_personnel_CHF_per_y = personnel costs for the filling station [CHF/y]
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]
%           CostMatrix = cost matrix
%           TextLegend = string containing the categories of costs

%% Data

m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2); %[t/y]
r = finance.r;
finance.fillingStation.a_station = r./(1-(1+r).^-filling_station.lifetime);

%% CAPEX

C_temporaryStorage_CAPEX_CHF_per_y = temporaryStorage.capture(i).costs.AIC_CHF_per_y; %[CHF/y]
C_fillingStation_CAPEX_CHF_per_y = filling_station.C_filling_station_CHF.*finance.fillingStation.a_station; %[CHF/y]

CAPEX_tot_CHF = temporaryStorage.capture(i).costs.CAPEX_CHF + filling_station.C_filling_station_CHF; %[CHF]
AIC_CHF_per_y = C_temporaryStorage_CAPEX_CHF_per_y + C_fillingStation_CAPEX_CHF_per_y; %[CHF/y]

for st = 1:size(train.start,2)
    for go = 1:size(train.start(st).goal,2)

        n_shipment = train.capture(i).start(st).goal(go).n_shipment;
        %n_RTC = train.capture(i).start(st).goal(go).n_RTC;
        n_blocktrain = train.capture(i).start(st).goal(go).n_blocktrain;
        t_roundtrip_train_h = train.start(st).goal(go).t_roundtrip_h; %[h]
        d_km = train.start(st).goal(go).d_km; %[km]

        %% OPEX

        C_wagon_rent = n_shipment.*ceil(t_roundtrip_train_h./24).*train.C_wagon_4axles; %[CHF/y]
        C_customs = n_blocktrain.*train.C_custom; %[CHF/y]
        %C_weigh = n_transshipment.*train.C_weight_CHF_per_container; %[CHF/y]
        C_transport = n_shipment.*(d_km.*train.C_transport_CHF_per_RTC_per_km + train.C_transport_base_CHF_per_RTC); %[CHF/y]
        C_temporaryStorage_OPEX = temporaryStorage.capture(i).costs.AOC_CHF_per_y; %[CHF/y]
        C_fillingStation_OPEX = filling_station.n_personnel.*filling_station.C_salary_CHF_per_y; %[CHF/y] missing: energy
        
        %Total costs
        AOC_CHF_per_y = C_wagon_rent + C_customs + C_transport + C_temporaryStorage_OPEX + C_fillingStation_OPEX; %[CHF/y]

        %% Yearly costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]

        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t]

        %% Rename and resize matrices

        train.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot_CHF; %[CHF]
        train.capture(i).start(st).goal(go).costs.CAPEX_CHF_per_y = AIC_CHF_per_y; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_temporaryStorage_CHF_per_y = C_temporaryStorage_CAPEX_CHF_per_y + C_temporaryStorage_OPEX; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_fillingStation_CHF_per_y = C_fillingStation_CAPEX_CHF_per_y + C_fillingStation_OPEX; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_wagon_rent_CHF_per_y = C_wagon_rent; %[CHF/y]
        train.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y = C_customs; %[CHF/y]
        %train.capture(i).start(st).goal(go).costs.C_weigh_CHF_per_y = C_weigh; %[CHF/y]
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
            train.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [train.capture(i).start(st).goal(go).costs.C_temporaryStorage_CHF_per_y(l,:)', ...
                train.capture(i).start(st).goal(go).costs.C_fillingStation_CHF_per_y(l,:)', ...
                train.capture(i).start(st).goal(go).costs.C_wagon_rent_CHF_per_y(l,:)', ...
                train.capture(i).start(st).goal(go).costs.C_customs_CHF_per_y(l,:)', ...
                train.capture(i).start(st).goal(go).costs.C_transport_CHF_per_y(l,:)'];
        end
        train.capture(i).start(st).goal(go).costs.TextLegend = {'Temporary storage','Filling station',...
            'Wagon rent', 'Customs', 'Train transport'};
          

    end
end

end