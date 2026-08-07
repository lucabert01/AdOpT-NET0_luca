function [plant, finance] = CostsCapture(plant, finance, i, isforarticle)
%This function calculates the costs for the capture unit
%INPUT: plant = struct containing the plant information
%       finance = struct containing the finance information
%       i = plant index
%       isforarticle = boolean: 1 = for article (fixed cost of capture), 2
%       = for report (cost from Casale)
%OUTPUT: plant = struct containing the plant information
%        AIC_CHF_per_y = annualized investment costs [CHF/y]
%        AOC_CHF_per_y = annualized operating costs [CHF/y]
%        TAC_CHF_per_y = total annualized costs [CHF/y]
%        LC_CHF_per_t = levelized costs [CHF/t]

%% Data

m_capt_t_per_y = plant.capture(i).m_capt_t_per_y;
r = finance.r;
xEURCHF = finance.x_EURCHF;

finance.capture.a_CAPEX = r./(1-(1+r).^-plant.lifetime_capture);

% switch plant.capture(i).b_PCC
%     case 0
%     case 1
% end

switch isforarticle
    case 1
        switch i
            case 4
                LC_CHF_per_t = 0; %[CHF/t]
            otherwise
                LC_CHF_per_t = 100.*xEURCHF; %[CHF/t]
        end
        TAC_CHF_per_y = LC_CHF_per_t.*m_capt_t_per_y; %[CHF/t]
        
        plant.capture(i).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        plant.capture(i).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t]
    case 2
        %% CAPEX

        AIC_CHF_per_y = plant.capture(i).CAPEX_EUR_per_t.*m_capt_t_per_y.*xEURCHF; %[CHF/y]
        CAPEX_tot_CHF = AIC_CHF_per_y./finance.capture.a_CAPEX;

        %% OPEX

        OPEX_fix_CHF_per_y = plant.capture(i).OPEX_fixed_EUR_per_t.*m_capt_t_per_y.*xEURCHF; %[CHF/y]
        OPEX_var_CHF_per_y = (plant.capture(i).OPEX_el_EUR_per_t + plant.capture(i).OPEX_steam_EUR_per_t + ...
            plant.capture(i).OPEX_other_EUR_per_t).*m_capt_t_per_y.*xEURCHF; %[CHF/y]

        AOC_CHF_per_y = OPEX_fix_CHF_per_y + OPEX_var_CHF_per_y; %[CHF/y]

        %% Yearly costs

        TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]

        LC_CHF_per_t = plant.capture(i).C_capt_EUR_per_t.*xEURCHF; %[CHF/t]

        %% Rename and resize matrices

        plant.capture(i).costs.CAPEX_tot_CHF = CAPEX_tot_CHF; %[CHF]
        plant.capture(i).costs.AIC_CHF_per_y = AIC_CHF_per_y; %[CHF/y]
        plant.capture(i).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
        plant.capture(i).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
        plant.capture(i).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t]
end

end

