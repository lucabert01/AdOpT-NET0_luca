function [ship, finance] = CharacteristicsAndCostsShipBulk(ship, conditioning, finance, i, n_scen, n_work, filenameRoussanalyTable)
%This function calculates the costs for the ship bulk transport based on
%the study from Roussanaly, Energies, 2021, 14, 5635
%We assume only discrete sizes for ships are possible
%INPUT: ship = struct containing the ship information
%       conditioning = struct containing the conditioning information
%       finance = struct containing the finance information
%       i = plant index
%       n_scen = number of scenarios
%       n_work = number of working days scenarios
%       filenameRoussanalyTable = filename for table
%OUTPUT: ship = struct containing the ship information
%           CAPEX_tot_CHF = total CAPEX [CHF]
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]
%           CostMatrix = cost matrix
%           TextLegend = string containing the categories of costs

%% Data

switch ship.p
    case 8
        m_liq_t_per_y = conditioning.capture(i).m_liq7barg_t_per_y; %[t/y]
    case 16
        m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y; %[t/y]
end
m_liq_t_per_y = m_liq_t_per_y(2);

[TableRoussanaly,DataRoussanaly] = OpenRoussanalyTable(filenameRoussanalyTable);

v_km_per_h = ship.v_km_per_h;

r = finance.r;
x_EURCHF = finance.x_EURCHF;

for st = 1:size(ship.start,2)
    for go = 1:size(ship.start(st).goal,2)

        d_km = ship.start(st).goal(go).d_transport_km; %[km]

        emissions_t_per_t = d_km.*ship.gamma_kg_per_t_per_km./1000; %[tCO2/tCO2transp]
        leakage_t = d_km.*ship.lambda_t_per_km; %[tCO2]
        leakage_t_per_t = leakage_t./m_liq_t_per_y; %[tCO2/tCO2transp]

        %         t_transport_h = d_km./DataRoussanaly.v_km_per_h; %[h]
        t_transport_h = d_km./v_km_per_h; %[h]
        t_roundtrip_h = DataRoussanaly.tDep_h + 2.*t_transport_h + DataRoussanaly.tArr_h; %[h]

        %% Rename and resize matrices for ship
        ship.start(st).goal(go).t_transport_h = t_transport_h.*ones(1,3);
        ship.start(st).goal(go).t_roundtrip_h = t_roundtrip_h.*ones(1,3); %[h]
        ship.capture(i).start(st).goal(go).t_transport_h = t_transport_h;
        ship.capture(i).start(st).goal(go).t_roundtrip_h = t_roundtrip_h;
        ship.capture(i).start(st).goal(go).emissions_t_per_t = emissions_t_per_t;
        ship.capture(i).start(st).goal(go).leakage_t = leakage_t;
        ship.capture(i).start(st).goal(go).leakage_t_per_t = leakage_t_per_t;

        [C_opt_tCO2, C_ship_real_tCO2, n_shipments, f_per_y, n_ship, C_stor_t, ...
            CAPEX_stor_EUR, CAPEX_stor_EUR_per_y, OPEX_stor_EUR_per_y, CAPEX_load_EUR, ...
            CAPEX_load_EUR_per_y, OPEX_load_EUR_per_y, CAPEX_1ship_EUR, CAPEX_ship_EUR, ...
            CAPEX_ship_EUR_per_y, cFuel_t_per_tCO2_per_km, OPEX_fix_ship_EUR_per_y, ...
            OPEX_fuel_1ship_EUR_per_y, OPEX_fuel_ship_EUR_per_y, OPEX_harbor_ship_EUR_per_y, ...
            TYC_EUR_per_y, LC_EUR_per_t] = ShipBulkCostsRoussanaly(ship, r, m_liq_t_per_y, d_km, t_roundtrip_h, DataRoussanaly, TableRoussanaly);

        %% CAPEX

        CAPEX_tot = (CAPEX_stor_EUR + CAPEX_load_EUR + CAPEX_ship_EUR).*x_EURCHF; %[CHF]
        AIC_CHF_per_y = (CAPEX_stor_EUR_per_y + CAPEX_load_EUR_per_y + CAPEX_ship_EUR_per_y).*x_EURCHF; %[CHF/y]

        %% OPEX

        AOC_CHF_per_y = (OPEX_stor_EUR_per_y + OPEX_load_EUR_per_y + OPEX_fix_ship_EUR_per_y + ...
            OPEX_fuel_ship_EUR_per_y + OPEX_harbor_ship_EUR_per_y).*x_EURCHF; %[CHF/y]

        %% Yearly costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]

        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t]

        %% Rename and resize matrices

        ship.capture(i).start(st).goal(go).C_opt_tCO2 = C_opt_tCO2; %[t]
        ship.capture(i).start(st).goal(go).C_ship_real_tCO2 = C_ship_real_tCO2; %[t]
        ship.capture(i).start(st).goal(go).mMax_t = C_ship_real_tCO2;
        ship.capture(i).start(st).goal(go).n_ship = n_ship;
        ship.capture(i).start(st).goal(go).n_shipment = n_shipments;
        ship.capture(i).start(st).goal(go).f_per_y = f_per_y;
        ship.capture(i).start(st).goal(go).n_journey = n_shipments;
        ship.capture(i).start(st).goal(go).n_isotainer = 0;
        ship.capture(i).start(st).goal(go).C_stor_t = C_stor_t;

        ship.capture(i).start(st).goal(go).costs.CAPEX_stor_CHF = CAPEX_stor_EUR.*x_EURCHF; %[CHF]
        ship.capture(i).start(st).goal(go).costs.CAPEX_stor_CHF_per_y = CAPEX_stor_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.OPEX_stor_CHF_per_y = OPEX_stor_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.CAPEX_load_CHF = CAPEX_load_EUR.*x_EURCHF; %[CHF]
        ship.capture(i).start(st).goal(go).costs.CAPEX_load_CHF_per_y = CAPEX_load_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.OPEX_load_CHF_per_y = OPEX_load_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.CAPEX_1ship_CHF = CAPEX_1ship_EUR.*x_EURCHF; %[CHF]
        ship.capture(i).start(st).goal(go).costs.CAPEX_ship_CHF = CAPEX_ship_EUR.*x_EURCHF; %[CHF]
        ship.capture(i).start(st).goal(go).costs.CAPEX_ship_CHF_per_y = CAPEX_ship_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.OPEX_fix_ship_CHF_per_y = OPEX_fix_ship_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.OPEX_fuel_1ship_CHF_per_y = OPEX_fuel_1ship_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.OPEX_fuel_ship_CHF_per_y = OPEX_fuel_ship_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.OPEX_harbor_ship_CHF_per_y = OPEX_harbor_ship_EUR_per_y.*x_EURCHF; %[CHF/y]

        ship.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
        ship.capture(i).start(st).goal(go).costs.AIC_CHF_per_y = AIC_CHF_per_y; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_temporaryStorage_CHF_per_y = CAPEX_stor_EUR_per_y.*x_EURCHF + OPEX_stor_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_vesselLoadingStation_CHF_per_y = CAPEX_load_EUR_per_y.*x_EURCHF + OPEX_load_EUR_per_y.*x_EURCHF; %[CHF/y]
        ship.capture(i).start(st).goal(go).costs.C_transport_ship_CHF_per_y = CAPEX_ship_EUR_per_y.*x_EURCHF + (OPEX_fix_ship_EUR_per_y + OPEX_fuel_ship_EUR_per_y).*x_EURCHF; %[CHF/y]
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
        ship.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,4,n_work);
        for l = 1:n_work
            %sprintf('%5d',[i j st go l])
            ship.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [ship.capture(i).start(st).goal(go).costs.C_transport_ship_CHF_per_y(l,:)', ...
                ship.capture(i).start(st).goal(go).costs.C_temporaryStorage_CHF_per_y(l,:)',...
                ship.capture(i).start(st).goal(go).costs.C_vesselLoadingStation_CHF_per_y(l,:)',...
                ship.capture(i).start(st).goal(go).costs.OPEX_harbor_ship_CHF_per_y(l,:)'];
        end
        ship.capture(i).start(st).goal(go).costs.TextLegend = {'Transport','Intermediary storage','Loading station','Harbor fee'};
        %ship.capture(i).start(st).goal(go).costs.TextXTickLabel = {'Optimistic','Average','Conservative'};

    end
end


end

