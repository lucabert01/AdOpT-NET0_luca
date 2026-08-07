function [truckBulk, finance] = CostsTruckBulk(truck, truckBulk, conditioning, temporaryStorage, filling_station,finance, i, n_scen, n_work)
%This function calculates the costs for the truck batch transport (isotainer)
%INPUT: truck = struct containing the truck information
%       conditioning = struct containing the conditioning information
%       isotainer = struct containing the isotainer information
%       finance = struct containing the finance information
%       i = plant index
%       j = personnel index
%       n_scen = number of scenarios
%       n_work = number of working days scenarios
%OUTPUT: truck = struct containing the truck information
%           C_truck_CHF = yearly costs for trucks (tractors + trailers + evt. isotainers (if bought)) [CHF]
%           C_tractor_CHF_per_y = yearly costs for tractors [CHF/y]
%           C_trailer_CHF_per_y = yearly costs for trailers [CHF/y]
%           C_isotainer_buy_CHF_per_y = yearly costs for isotainers (if bought) [CHF/y]
%           CAPEX_tot_CHF = total CAPEX [CHF]
%           C_fuel_CHF_per_y = yearly costs for fuel (for all trucks) [CHF/y]
%           C_maintenance_CHF_per_y = yearly maintenance costs [CHF/y]
%           C_HGVT_CHF_per_y = yearly costs for the heavy goods vehicle tax [CHF/y]
%           C_personnel_CHF_per_y = yearly costs for personnel [CHF/y]
%           C_insurance_CHF_per_y = yearly insurance costs [CHF/y]
%           C_vehicleTax_CHF_per_y = yearly costs for vehicle tax [CHF/y]
%           C_administration_CHF_per_y = yearly administration costs [CHF/y]
%           C_tires_CHF_per_y = yearly costs for tires [CHF/y]
%           C_infrastructure_CHF_per_y = yearly costs for infrastructure %[CHF/y]
%           C_isotainer_rent_CHF_per_y = yearly cost for renting isotainers (if rent) [CHF/y]
%           beta1_CHF_per_km = factor beta1, depending on the distance covered %[CHF/km]
%           beta2_CHF_per_y = factor beta2, independent of the distance covered %[CHF/y]
%           estimation_from_Johannes_CHF_per_t = cost estimation from the overall price from Neustark [CHF/t_CO2]
%           estimation2_from_Johannes_CHF_per_t = cost estimation from the overall price from Neustark [CHF/t_CO2]
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]
%           CostMatrix = cost matrix
%           TextLegend = string containing the categories of costs

%% Data

C_salary_CHF_per_h = truck.C_salary_CHF_per_h; %[CHF/h]
m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2); %[t/y]

r = finance.r;

for st = 1:size(truckBulk.start,2)
    for go = 1:size(truckBulk.start(st).goal,2)

        n_truck = truckBulk.capture(i).start(st).goal(go).n_truck;
        d_truck_tot_km = truckBulk.capture(i).start(st).goal(go).d_truck_tot_km; %[km]
%        t_roundtrip_truck_h = truckBulk.start(st).goal(go).t_roundtrip_h; %[h]
%         n_roundtrip_per_truck_per_y = truck.capture(i).start(st).goal(go).n_roundtrip_per_truck_per_y; 
        t_truck_tot_h = truckBulk.capture(i).start(st).goal(go).t_truck_tot_h; %[h]


        %% CAPEX
        
        finance.truckBulk.a_tractor = r./(1-(1+r).^-truckBulk.lifetime_tractor);
        finance.truckBulk.a_trailer = r./(1-(1+r).^-truckBulk.lifetime_trailer);
        finance.fillingStation.a_station = r./(1-(1+r).^-filling_station.lifetime);

        C_tractor = truckBulk.C_tractor_CHF.*n_truck.*finance.truckBulk.a_tractor; %[CHF/y]
        C_trailer = truckBulk.C_trailer_CHF.*n_truck.*finance.truckBulk.a_trailer; %[CHF/y]
        
        C_temporaryStorage_CAPEX = temporaryStorage.capture(i).costs.AIC_CHF_per_y;
        
        C_fillingStation_CAPEX = filling_station.C_filling_station_CHF.*finance.fillingStation.a_station; %[CHF/y]
        
        CAPEX_tot = n_truck.*(truckBulk.C_tractor_CHF + truckBulk.C_trailer_CHF) + ...
            temporaryStorage.capture(i).costs.CAPEX_CHF + filling_station.C_filling_station_CHF; %[CHF]
        AIC_CHF_per_y = C_tractor + C_trailer + C_temporaryStorage_CAPEX + C_fillingStation_CAPEX; %[CHF/y]

        %% OPEX

        C_fuel = truckBulk.c_fuel_L_per_km.*truck.C_fuel_CHF_per_L.*d_truck_tot_km; %[CHF/y]
        C_maintenance = truckBulk.C_maintenance_CHF_per_km.*d_truck_tot_km; %[CHF/y]
        C_HGVT = CostHeavyVehicleTax(truck, truckBulk, st, go, i); %[CHF/y]
        C_personnel = t_truck_tot_h.*C_salary_CHF_per_h; %[CHF/y]
        C_insurance = truckBulk.C_insurance_CHF_per_y.*n_truck; %[CHF/y]
        C_vehicleTax = truckBulk.C_vehicleTax_CHF_per_y.*n_truck; %[CHF/y]
        C_administration = truckBulk.C_administration_CHF_per_y.*n_truck; %[CHF/y]
        C_tires = truckBulk.C_tires_CHF_per_y.*n_truck; %[CHF/y]
        C_infrastructure = truckBulk.C_infrastructure_CHF_per_y.*n_truck; %[CHF/y]
        C_temporaryStorage_OPEX = temporaryStorage.capture(i).costs.AOC_CHF_per_y; %[CHF/y]
        C_fillingStation_OPEX = filling_station.n_personnel.*filling_station.C_salary_CHF_per_y; %[CHF/y] missing: energy

        beta1 = truckBulk.c_fuel_L_per_km.*truck.C_fuel_CHF_per_L + truckBulk.C_maintenance_CHF_per_km; %[CHF/km]
        beta2 = C_HGVT + C_personnel + C_insurance + C_vehicleTax + C_administration + ...
                    C_tires + C_infrastructure + C_temporaryStorage_OPEX + C_fillingStation_OPEX; %[CHF/y]

        AOC_CHF_per_y = beta1.*d_truck_tot_km + beta2; %[CHF/y]

        %% Yearly and levelized costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]
       
        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t_CO2]

        %% Rename and resize matrices

%         truck.capture(i).start(st).goal(go).personnel(j).costs.C_truck_CHF = C_truck; %[CHF]
        truckBulk.capture(i).start(st).goal(go).costs.C_tractor_CHF_per_y = C_tractor; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_trailer_CHF_per_y = C_trailer; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_temporaryStorage_CAPEX_CHF_per_y = C_temporaryStorage_CAPEX; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_fillingStation_CAPEX_CHF_per_y = C_fillingStation_CAPEX; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
        truckBulk.capture(i).start(st).goal(go).costs.C_fuel_CHF_per_y = C_fuel; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_maintenance_CHF_per_y = C_maintenance; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_HGVT_CHF_per_y = C_HGVT; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_personnel_CHF_per_y = C_personnel; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_insurance_CHF_per_y = C_insurance; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_vehicleTax_CHF_per_y = C_vehicleTax; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_administration_CHF_per_y = C_administration; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_tires_CHF_per_y = C_tires; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_infrastructure_CHF_per_y = C_infrastructure; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_temporaryStorage_OPEX_CHF_per_y = C_temporaryStorage_OPEX; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.C_fillingStation_OPEX_CHF_per_y = C_fillingStation_OPEX; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.beta1_CHF_per_km = beta1; %[CHF/km]
        truckBulk.capture(i).start(st).goal(go).costs.beta2_CHF_per_y = beta2; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        truckBulk.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t_CO2]

        fn = fieldnames(truckBulk.capture(i).start(st).goal(go).costs);

        for l = 1:numel(fn)
            if size(truckBulk.capture(i).start(st).goal(go).costs.(fn{l}),1) == 1 && size(truckBulk.capture(i).start(st).goal(go).costs.(fn{l}),2) == 3
                truckBulk.capture(i).start(st).goal(go).costs.(fn{l}) = truckBulk.capture(i).start(st).goal(go).costs.(fn{l}).*ones(3,1);
            elseif size(truckBulk.capture(i).start(st).goal(go).costs.(fn{l}),1) == 3 && size(truckBulk.capture(i).start(st).goal(go).costs.(fn{l}),2) == 1
                truckBulk.capture(i).start(st).goal(go).costs.(fn{l}) = truckBulk.capture(i).start(st).goal(go).costs.(fn{l}).*ones(1,3);
            elseif size(truckBulk.capture(i).start(st).goal(go).costs.(fn{l}),1) == 1 && size(truckBulk.capture(i).start(st).goal(go).costs.(fn{l}),2) == 1
                truckBulk.capture(i).start(st).goal(go).costs.(fn{l}) = truckBulk.capture(i).start(st).goal(go).costs.(fn{l}).*ones(3,3);
            end
        end
        
        
        % Prepare matrix for plots
        truckBulk.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,13,n_work);
        for l = 1:n_work
            truckBulk.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [truckBulk.capture(i).start(st).goal(go).costs.C_tractor_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_trailer_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_temporaryStorage_CAPEX_CHF_per_y(l,:)'+truckBulk.capture(i).start(st).goal(go).costs.C_temporaryStorage_OPEX_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_fillingStation_CAPEX_CHF_per_y(l,:)'+truckBulk.capture(i).start(st).goal(go).costs.C_fillingStation_CAPEX_CHF_per_y(l,:)',...
                        truckBulk.capture(i).start(st).goal(go).costs.C_tires_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_maintenance_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_infrastructure_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_vehicleTax_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_insurance_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_HGVT_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_administration_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_fuel_CHF_per_y(l,:)', ...
                        truckBulk.capture(i).start(st).goal(go).costs.C_personnel_CHF_per_y(l,:)'];
        end
        truckBulk.capture(i).start(st).goal(go).costs.TextLegend = ...
            {'Tractor','Trailer','Temporary storage','Filling station','Tires', ...
            'Maintenance','Infrastructure',...
            'Vehicle tax','Insurance','HGVT','Administration','Fuel','Personnel'};
        
    end
end

end

