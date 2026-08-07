function [buildNetwork] = CostsBuildNetwork(buildNetwork, i)
%This function calculates the "costs" for the artificial network building

for st = 1:size(buildNetwork.start,2)
    for go = 1:size(buildNetwork.start(st).goal,2)
        
        C_CHF = buildNetwork.start(st).goal(go).C_CHF;

        %% Calculs

        LC_CHF_per_t = C_CHF; %[CHF/t]

        buildNetwork.capture(i).start(st).goal(go).costs.LC_CHF_per_t = LC_CHF_per_t.*ones(3,3);
        buildNetwork.capture(i).start(st).goal(go).costs.TextLegend = '';
        buildNetwork.capture(i).start(st).goal(go).emissions_t_per_t = buildNetwork.start(st).goal(go).gamma_tot_t_per_t;
        buildNetwork.capture(i).start(st).goal(go).leakage_t_per_t = buildNetwork.lambda_t_per_km_per_t;
        buildNetwork.capture(i).start(st).goal(go).f_per_y = Inf;
        buildNetwork.capture(i).start(st).goal(go).mMax_t = buildNetwork.mMax_t;
        buildNetwork.capture(i).start(st).goal(go).n_isotainer = 0;
           
    end
end

end

