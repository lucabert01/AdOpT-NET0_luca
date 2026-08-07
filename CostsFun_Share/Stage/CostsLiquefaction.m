function [CAPEX_liquefaction_CHF, CAPEX_liquefaction_CHF_per_y, ...
    OPEX_fix_CHF_per_y, OPEX_var_CHF_per_y, OPEX_CHF_per_y, ...
    year_costs_liquefaction_CHF_per_y, levelized_costs_liquefaction_CHF_per_t] = CostsLiquefaction(p, m_liq_t_per_y, conditioning, finance, C_electricity_CHF_per_kWh)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

r = finance.r;
x_EURCHF = finance.x_EURCHF;
tau = conditioning.tau_cond;
CRF = r/(1-(1+r)^-tau);
% UCCI_2019_2021 = mean([178 185 190 195])/mean([183 184 183 181]);
% UCCI_2010_2021 = 0.9824;
Chemie_Technik_PCD_2019_2021 = 111.3./107.0;
CI_2019_2021 = Chemie_Technik_PCD_2019_2021;


switch p
    case 7
        %Investment costs in CHF
        C0_liquefaction_CHF = conditioning.C0_liquefaction_7barg_EUR2019.*CI_2019_2021.*x_EURCHF;
        %Liquefaction energy
        E_liquefaction_kWh_per_t = conditioning.E_liquefaction_7barg_kWh_per_t;
        C_impurity_removal = conditioning.C_impurity_removal_7barg;
    case 15
        %Investment costs in CHF
        C0_liquefaction_CHF = conditioning.C0_liquefaction_15barg_EUR2019.*CI_2019_2021.*x_EURCHF;
        %Liquefaction energy
        E_liquefaction_kWh_per_t = conditioning.E_liquefaction_15barg_kWh_per_t;
        C_impurity_removal = conditioning.C_impurity_removal_15barg;
end

%CAPEX
CAPEX_liquefaction_CHF = SevenTenthRule(C0_liquefaction_CHF, ...
    m_liq_t_per_y, conditioning.S0_liquefaction_t_per_y, conditioning.R_liquefaction); %[CHF]
CAPEX_liquefaction_CHF_per_y = CAPEX_liquefaction_CHF.*CRF; %[CHF/y]

%OPEX
OPEX_fix_CHF_per_y = CAPEX_liquefaction_CHF.*conditioning.muOM_liquefaction;

OPEX_var_CHF_per_y = (E_liquefaction_kWh_per_t.*C_electricity_CHF_per_kWh + ...
    (conditioning.C_water_EUR_per_t + C_impurity_removal).*x_EURCHF).*m_liq_t_per_y;

OPEX_CHF_per_y = OPEX_fix_CHF_per_y + OPEX_var_CHF_per_y;

%Yearly costs

year_costs_liquefaction_CHF_per_y = CAPEX_liquefaction_CHF_per_y + OPEX_CHF_per_y; %[CHF/y]
levelized_costs_liquefaction_CHF_per_t = year_costs_liquefaction_CHF_per_y./m_liq_t_per_y; %[CHF/t]

end