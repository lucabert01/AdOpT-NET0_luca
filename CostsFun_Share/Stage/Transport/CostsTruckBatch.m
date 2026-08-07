function [truckBatch, finance] = CostsTruckBatch(truck, truckBatch, conditioning, isotainer, finance, i, n_scen, n_work)
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

for st = 1:size(truckBatch.start,2)
    for go = 1:size(truckBatch.start(st).goal,2)

        n_truck = truckBatch.capture(i).start(st).goal(go).n_truck;
        n_isotainer = truckBatch.capture(i).start(st).goal(go).n_isotainer;
        d_truck_tot_km = truckBatch.capture(i).start(st).goal(go).d_truck_tot_km; %[km]
        t_roundtrip_truck_h = truckBatch.start(st).goal(go).t_roundtrip_h; %[h]
%         n_roundtrip_per_truck_per_y = truck.capture(i).start(st).goal(go).n_roundtrip_per_truck_per_y; 
        t_truck_tot_h = truckBatch.capture(i).start(st).goal(go).t_truck_tot_h; %[h]


        %% CAPEX
        
        finance.truckBatch.a_tractor = r./(1-(1+r).^-truckBatch.lifetime_tractor);
        finance.truckBatch.a_trailer = r./(1-(1+r).^-truckBatch.lifetime_trailer);
        finance.isotainer.a_isotainer = r./(1-(1+r).^-isotainer.lifetime);

        C_tractor = truckBatch.C_tractor_CHF.*n_truck.*finance.truckBatch.a_tractor; %[CHF/y]
        C_trailer = truckBatch.C_trailer_CHF.*n_truck.*finance.truckBatch.a_trailer; %[CHF/y]
        C_isotainer_buy = isotainer.C_1isotainer_CHF.*n_isotainer.*finance.isotainer.a_isotainer; %[CHF/y]
        
        switch isotainer.choice
            case 'rent'
                CAPEX_tot = n_truck.*(truckBatch.C_tractor_CHF + truckBatch.C_trailer_CHF); %[CHF]
                AIC_CHF_per_y = C_tractor + C_trailer; %[CHF/y]
            case 'buy'
                CAPEX_tot = n_truck.*(truckBatch.C_tractor_CHF + truckBatch.C_trailer_CHF) + n_isotainer.*isotainer.C_1isotainer_CHF; %[CHF]
                AIC_CHF_per_y = C_tractor + C_trailer + C_isotainer_buy; %[CHF/y]
        end

        %% OPEX

        C_fuel = truckBatch.c_fuel_L_per_km.*truck.C_fuel_CHF_per_L.*d_truck_tot_km; %[CHF/y]
        C_maintenance = truckBatch.C_maintenance_CHF_per_km.*d_truck_tot_km; %[CHF/y]
        C_HGVT = CostHeavyVehicleTax(truck, truckBatch, st, go, i); %[CHF/y]
        C_personnel = t_truck_tot_h.*C_salary_CHF_per_h; %[CHF/y]
        C_insurance = truckBatch.C_insurance_CHF_per_y.*n_truck; %[CHF/y]
        C_vehicleTax = truckBatch.C_vehicleTax_CHF_per_y.*n_truck; %[CHF/y]
        C_administration = truckBatch.C_administration_CHF_per_y.*n_truck; %[CHF/y]
        C_tires = truckBatch.C_tires_CHF_per_y.*n_truck; %[CHF/y]
        C_infrastructure = truckBatch.C_infrastructure_CHF_per_y.*n_truck; %[CHF/y]
        %C_isotainer_rent = isotainer.C_isotainer_CHF_per_y.*n_truck; %[CHF/y]


        beta1 = truckBatch.c_fuel_L_per_km.*truck.C_fuel_CHF_per_L + truckBatch.C_maintenance_CHF_per_km; %[CHF/km]
        switch isotainer.choice
            case 'rent'
                beta2 = C_HGVT + C_personnel + C_insurance + C_vehicleTax + C_administration + ...
                    C_tires + C_infrastructure + C_isotainer_rent; %[CHF/y]
            case 'buy'
                beta2 = C_HGVT + C_personnel + C_insurance + C_vehicleTax + C_administration + ...
                    C_tires + C_infrastructure; %[CHF/y]
        end

        AOC_CHF_per_y = beta1.*d_truck_tot_km + beta2; %[CHF/y]

        %% Yearly and levelized costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]
       
        LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t_CO2]

        %% Cost estimation with Johannes' number

        estimation_from_Johannes_CHF_per_t = t_truck_tot_h.*truckBatch.C_Johannes_CHF_per_h/m_liq_t_per_y; %[CHF/t]
        estimation2_from_Johannes_CHF_per_t = t_roundtrip_truck_h.*truckBatch.C_Johannes_CHF_per_h./isotainer.m_CO2_t; %[CHF/t]

        %% Rename and resize matrices

%         truck.capture(i).start(st).goal(go).personnel(j).costs.C_truck_CHF = C_truck; %[CHF]
        truckBatch.capture(i).start(st).goal(go).costs.C_tractor_CHF_per_y = C_tractor; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_trailer_CHF_per_y = C_trailer; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y = C_isotainer_buy; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.CAPEX_tot_CHF = CAPEX_tot; %[CHF]
        truckBatch.capture(i).start(st).goal(go).costs.C_fuel_CHF_per_y = C_fuel; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_maintenance_CHF_per_y = C_maintenance; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_HGVT_CHF_per_y = C_HGVT; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_personnel_CHF_per_y = C_personnel; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_insurance_CHF_per_y = C_insurance; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_vehicleTax_CHF_per_y = C_vehicleTax; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_administration_CHF_per_y = C_administration; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_tires_CHF_per_y = C_tires; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.C_infrastructure_CHF_per_y = C_infrastructure; %[CHF/y]
        %truck.capture(i).start(st).goal(go).personnel(j).costs.C_isotainer_rent_CHF_per_y = C_isotainer_rent; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.beta1_CHF_per_km = beta1; %[CHF/km]
        truckBatch.capture(i).start(st).goal(go).costs.beta2_CHF_per_y = beta2; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        truckBatch.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t_CO2]
        truckBatch.capture(i).start(st).goal(go).costs.estimation_from_Johannes_CHF_per_t = estimation_from_Johannes_CHF_per_t; %[CHF/t_CO2]
        truckBatch.capture(i).start(st).goal(go).costs.estimation2_from_Johannes_CHF_per_t = estimation2_from_Johannes_CHF_per_t; %[CHF/t_CO2]


        fn = fieldnames(truckBatch.capture(i).start(st).goal(go).costs);

        for l = 1:numel(fn)
            if size(truckBatch.capture(i).start(st).goal(go).costs.(fn{l}),1) == 1 && size(truckBatch.capture(i).start(st).goal(go).costs.(fn{l}),2) == 3
                truckBatch.capture(i).start(st).goal(go).costs.(fn{l}) = truckBatch.capture(i).start(st).goal(go).costs.(fn{l}).*ones(3,1);
            elseif size(truckBatch.capture(i).start(st).goal(go).costs.(fn{l}),1) == 3 && size(truckBatch.capture(i).start(st).goal(go).costs.(fn{l}),2) == 1
                truckBatch.capture(i).start(st).goal(go).costs.(fn{l}) = truckBatch.capture(i).start(st).goal(go).costs.(fn{l}).*ones(1,3);
            elseif size(truckBatch.capture(i).start(st).goal(go).costs.(fn{l}),1) == 1 && size(truckBatch.capture(i).start(st).goal(go).costs.(fn{l}),2) == 1
                truckBatch.capture(i).start(st).goal(go).costs.(fn{l}) = truckBatch.capture(i).start(st).goal(go).costs.(fn{l}).*ones(3,3);
            end
        end
        
        
        % Prepare matrix for plots
        truckBatch.capture(i).start(st).goal(go).costs.CostMatrix = zeros(n_scen,12,n_work);
        for l = 1:n_work
            switch isotainer.choice
                case 'rent'
                    C_isotainer_CHF_per_y = truckBatch.capture(i).start(st).goal(go).costs.C_isotainer_rent_CHF_per_y(l,:)';
                case 'buy'
                    C_isotainer_CHF_per_y = truckBatch.capture(i).start(st).goal(go).costs.C_isotainer_buy_CHF_per_y(l,:)';
            end
            truckBatch.capture(i).start(st).goal(go).costs.CostMatrix(:,:,l) = ...
                [truckBatch.capture(i).start(st).goal(go).costs.C_tractor_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_trailer_CHF_per_y(l,:)', ...
                        C_isotainer_CHF_per_y, ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_tires_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_maintenance_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_infrastructure_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_vehicleTax_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_insurance_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_HGVT_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_administration_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_fuel_CHF_per_y(l,:)', ...
                        truckBatch.capture(i).start(st).goal(go).costs.C_personnel_CHF_per_y(l,:)'];
        end
        truckBatch.capture(i).start(st).goal(go).costs.TextLegend = ...
            {'Tractor','Trailer','Isotainer','Tires', 'Maintenance','Infrastructure',...
            'Vehicle tax','Insurance','HGVT','Administration','Fuel','Personnel'};
        
    end
end

end

