function [conditioning, finance] = CostsConditioning(plant, conditioning, electricity, finance, i, isforarticle)
%This function calculates the costs for the capture unit
%INPUT: conditioning = struct containing the conditioning information
%       finance = struct containing the finance information
%       i = plant index
%       isforarticle = boolean: 1 = for article (fixed cost of capture), 2
%       = for report (cost from Casale)
%OUTPUT: conditioning = struct containing the conditioning information
%           liquefaction = information on liquefaction
%           compression = information on compression
%           CAPEX_tot_CHF = total CAPEX [CHF]
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs [CHF/t]

%% Data

DACE_2010_2015 = 100./93;
Chemie_Technik_PCD_2015_2021 = 111.3./100.0;
UCCI_2010_2021 = DACE_2010_2015.*Chemie_Technik_PCD_2015_2021;
x_EURCHF = finance.x_EURCHF;

%Electricity cost and emissions
Country = plant.capture(i).CountryISO;
switch ismember(Country,electricity.T_electricity.Properties.VariableNames)
    case 0
        C_electricity_CHF_per_kWh = electricity.T_electricity.EU27(2);
        gamma_electricity_t_per_kWh = electricity.T_electricity.EU27(4);
    case 1
        C_electricity_CHF_per_kWh = electricity.T_electricity{2,Country};
        gamma_electricity_t_per_kWh = electricity.T_electricity{4,Country};
end


%CO2 compressed per year
conditioning.capture(i).m_comp_t_per_y = plant.capture(i).m_capt_t_per_y.*conditioning.eta_CO2_compression;
m_comp_t_per_y = conditioning.capture(i).m_comp_t_per_y;

%Adaptations of units to those needed by Knoope

I0_compression_CHF = conditioning.I0_EUR2010.*UCCI_2010_2021.*x_EURCHF;

m_comp_kg_per_s = m_comp_t_per_y./3.6./conditioning.H_comp_h;
E_compression_GasOn_kJ_per_kg = conditioning.E_compression_GasOn_kWh_per_t.*3.6;
E_compression_GasOff_kJ_per_kg = conditioning.E_compression_GasOff_kWh_per_t.*3.6;
E_compression_DenseOn_kJ_per_kg = conditioning.E_compression_DenseOn_kWh_per_t.*3.6;
E_compression_DenseOff_kJ_per_kg = conditioning.E_compression_DenseOff_kWh_per_t.*3.6;
y = conditioning.R_compression;
me = conditioning.me;
Wcomp0 = conditioning.Wcomp0;
W1compMax = conditioning.W1compMax;

%% Discount

r = finance.r;
CRF = r./(1-(1+r).^-conditioning.tau_cond);
finance.conditioning.a_cond = CRF;

%% Liquefaction

%CO2 liquefied per year
conditioning.capture(i).m_liq7barg_t_per_y = plant.capture(i).m_capt_t_per_y.*conditioning.eta_CO2_liquefaction_7barg;
m_liq7barg_t_per_y = conditioning.capture(i).m_liq7barg_t_per_y;

[CAPEX_liquefaction7barg_CHF, CAPEX_liquefaction7barg_CHF_per_y, ...
    OPEX_fix_7barg_CHF_per_y, OPEX_var_7barg_CHF_per_y, OPEX_7barg_CHF_per_y, ...
    year_costs_liquefaction_7barg_CHF_per_y, levelized_costs_liquefaction_7barg_CHF_per_t] = CostsLiquefaction(7, m_liq7barg_t_per_y, conditioning, finance, C_electricity_CHF_per_kWh);

switch isforarticle
    case 1
        %CO2 liquefied per year
        conditioning.capture(i).m_liq15barg_t_per_y = plant.capture(i).m_capt_t_per_y.*conditioning.eta_CO2_liquefaction_15barg;
        conditioning.capture(i).m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2);
        m_liq15barg_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y;

        %Costs
        [CAPEX_liquefaction15barg_CHF, CAPEX_liquefaction15barg_CHF_per_y, ...
            OPEX_fix_15barg_CHF_per_y, OPEX_var_15barg_CHF_per_y, OPEX_15barg_CHF_per_y, ...
            year_costs_liquefaction_15barg_CHF_per_y, levelized_costs_liquefaction_15barg_CHF_per_t] = CostsLiquefaction(15, m_liq15barg_t_per_y, conditioning, finance, C_electricity_CHF_per_kWh);
    case 2
        %CO2 liquefied per year
        conditioning.capture(i).m_liq15barg_t_per_y = plant.capture(i).m_capt_t_per_y.*conditioning.capture(i).eta_cond.*ones(1,3);
        conditioning.capture(i).m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2);

        m_liq15barg_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y;
        m_comp_t_per_y = conditioning.capture(i).m_comp_t_per_y;

        % CAPEX
        CAPEX_liquefaction15barg_CHF_per_y = conditioning.capture(i).CAPEX_EUR_per_t.*m_liq15barg_t_per_y.*x_EURCHF;
        CAPEX_liquefaction15barg_CHF = CAPEX_liquefaction15barg_CHF_per_y./CRF;

        % OPEX
        OPEX_fix_15barg_CHF_per_y = conditioning.capture(i).OPEX_fixed_EUR_per_t.*m_liq15barg_t_per_y.*x_EURCHF;
        OPEX_var_15barg_CHF_per_y = (conditioning.capture(i).OPEX_el_EUR_per_t + conditioning.capture(i).OPEX_other_EUR_per_t).*m_liq15barg_t_per_y.*x_EURCHF;
        OPEX_15barg_CHF_per_y = OPEX_fix_15barg_CHF_per_y + OPEX_var_15barg_CHF_per_y;

        %Total
        year_costs_liquefaction_15barg_CHF_per_y = CAPEX_liquefaction15barg_CHF_per_y + OPEX_15barg_CHF_per_y;
        levelized_costs_liquefaction_15barg_CHF_per_t = year_costs_liquefaction_15barg_CHF_per_y./m_liq15barg_t_per_y;
end


%% Compression

%CAPEX
CAPEX_compression_GasOn_CHF = I0_compression_CHF.*...
    (((E_compression_GasOn_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./Wcomp0).^y).*...
    ((ceil((E_compression_GasOn_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./W1compMax)).^me);
CAPEX_compression_GasOn_CHF_per_y = CAPEX_compression_GasOn_CHF.*CRF; %[CHF/y]

CAPEX_compression_GasOff_CHF = I0_compression_CHF.*...
    (((E_compression_GasOff_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./Wcomp0).^y).*...
    ((ceil((E_compression_GasOff_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./W1compMax)).^me);
CAPEX_compression_GasOff_CHF_per_y = CAPEX_compression_GasOff_CHF.*CRF; %[CHF/y]

CAPEX_compression_DenseOn_CHF = I0_compression_CHF.*...
    (((E_compression_DenseOn_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./Wcomp0).^y).*...
    ((ceil((E_compression_DenseOn_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./W1compMax)).^me);
CAPEX_compression_DenseOn_CHF_per_y = CAPEX_compression_DenseOn_CHF.*CRF; %[CHF/y]

CAPEX_compression_DenseOff_CHF = I0_compression_CHF.*...
    (((E_compression_DenseOff_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./Wcomp0).^y).*...
    ((ceil((E_compression_DenseOff_kJ_per_kg.*m_comp_kg_per_s.*1e-3)./W1compMax)).^me);
CAPEX_compression_DenseOff_CHF_per_y = CAPEX_compression_DenseOff_CHF.*CRF; %[CHF/y]

% OPEX

OPEX_fix_compression_GasOn_CHF_per_y = CAPEX_compression_GasOn_CHF.*conditioning.muOM_compression;
OPEX_var_compression_GasOn_CHF_per_y = E_compression_GasOn_kJ_per_kg.*m_comp_kg_per_s.*conditioning.H_comp_h.*C_electricity_CHF_per_kWh;
OPEX_compression_GasOn_CHF_per_y = OPEX_fix_compression_GasOn_CHF_per_y + OPEX_var_compression_GasOn_CHF_per_y;

OPEX_fix_compression_GasOff_CHF_per_y = CAPEX_compression_GasOff_CHF.*conditioning.muOM_compression;
OPEX_var_compression_GasOff_CHF_per_y = E_compression_GasOff_kJ_per_kg.*m_comp_kg_per_s.*conditioning.H_comp_h.*C_electricity_CHF_per_kWh;
OPEX_compression_GasOff_CHF_per_y = OPEX_fix_compression_GasOff_CHF_per_y + OPEX_var_compression_GasOff_CHF_per_y;

OPEX_fix_compression_DenseOn_CHF_per_y = CAPEX_compression_DenseOn_CHF.*conditioning.muOM_compression;
OPEX_var_compression_DenseOn_CHF_per_y = E_compression_DenseOn_kJ_per_kg.*m_comp_kg_per_s.*conditioning.H_comp_h.*C_electricity_CHF_per_kWh;
OPEX_compression_DenseOn_CHF_per_y = OPEX_fix_compression_DenseOn_CHF_per_y + OPEX_var_compression_DenseOn_CHF_per_y;

OPEX_fix_compression_DenseOff_CHF_per_y = CAPEX_compression_DenseOff_CHF.*conditioning.muOM_compression;
OPEX_var_compression_DenseOff_CHF_per_y = E_compression_DenseOff_kJ_per_kg.*m_comp_kg_per_s.*conditioning.H_comp_h.*C_electricity_CHF_per_kWh;
OPEX_compression_DenseOff_CHF_per_y = OPEX_fix_compression_DenseOff_CHF_per_y + OPEX_var_compression_DenseOff_CHF_per_y;

%Yearly costs

year_costs_compression_GasOn_CHF_per_y = CAPEX_compression_GasOn_CHF_per_y + OPEX_compression_GasOn_CHF_per_y; %[CHF/y]
levelized_costs_compression_GasOn_CHF_per_t = year_costs_compression_GasOn_CHF_per_y./m_comp_t_per_y; %[CHF/t]

year_costs_compression_GasOff_CHF_per_y = CAPEX_compression_GasOff_CHF_per_y + OPEX_compression_GasOff_CHF_per_y; %[CHF/y]
levelized_costs_compression_GasOff_CHF_per_t = year_costs_compression_GasOff_CHF_per_y./m_comp_t_per_y; %[CHF/t]

year_costs_compression_DenseOn_CHF_per_y = CAPEX_compression_DenseOn_CHF_per_y + OPEX_compression_DenseOn_CHF_per_y; %[CHF/y]
levelized_costs_compression_DenseOn_CHF_per_t = year_costs_compression_DenseOn_CHF_per_y./m_comp_t_per_y; %[CHF/t]

year_costs_compression_DenseOff_CHF_per_y = CAPEX_compression_DenseOff_CHF_per_y + OPEX_compression_DenseOff_CHF_per_y; %[CHF/y]
levelized_costs_compression_DenseOff_CHF_per_t = year_costs_compression_DenseOff_CHF_per_y./m_comp_t_per_y; %[CHF/t]

%% Rename and resize matrices

% conditioning.C_electricity_liquefaction_CHF_per_t = C_electricity_liquefaction_CHF_per_t; %[CHF/t]
% conditioning.C_electricity_compression_CHF_per_t = C_electricity_compression_CHF_per_t; %[CHF/t]

conditioning.capture(i).C_electricity_CHF_per_kWh = C_electricity_CHF_per_kWh;
conditioning.capture(i).gamma_electricity_t_per_kWh = gamma_electricity_t_per_kWh;

conditioning.capture(i).liquefaction7barg.costs.CAPEX_tot_CHF = CAPEX_liquefaction7barg_CHF; %[CHF]
conditioning.capture(i).liquefaction7barg.costs.AIC_CHF_per_y = CAPEX_liquefaction7barg_CHF_per_y; %[CHF]
conditioning.capture(i).liquefaction7barg.costs.OPEX_fix_CHF_per_y = OPEX_fix_7barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction7barg.costs.OPEX_var_CHF_per_y = OPEX_var_7barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction7barg.costs.AOC_CHF_per_y = OPEX_7barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction7barg.costs.TAC_CHF_per_y = year_costs_liquefaction_7barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction7barg.costs.levelized_costs_0_CHF_per_t = levelized_costs_liquefaction_7barg_CHF_per_t; %[CHF/t]
conditioning.capture(i).liquefaction7barg.gamma_liq_t_per_t = gamma_electricity_t_per_kWh.*conditioning.E_liquefaction_7barg_kWh_per_t; %[t/t]

conditioning.capture(i).liquefaction15barg.costs.CAPEX_tot_CHF = CAPEX_liquefaction15barg_CHF; %[CHF]
conditioning.capture(i).liquefaction15barg.costs.AIC_CHF_per_y = CAPEX_liquefaction15barg_CHF_per_y; %[CHF]
conditioning.capture(i).liquefaction15barg.costs.OPEX_fix_CHF_per_y = OPEX_fix_15barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction15barg.costs.OPEX_var_CHF_per_y = OPEX_var_15barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction15barg.costs.AOC_CHF_per_y = OPEX_15barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction15barg.costs.TAC_CHF_per_y = year_costs_liquefaction_15barg_CHF_per_y; %[CHF/y]
conditioning.capture(i).liquefaction15barg.costs.LC_CHF_per_t = levelized_costs_liquefaction_15barg_CHF_per_t; %[CHF/t]
conditioning.capture(i).liquefaction15barg.gamma_liq_t_per_t = gamma_electricity_t_per_kWh.*conditioning.E_liquefaction_15barg_kWh_per_t; %[t/t]

conditioning.capture(i).compression_GasOn.costs.CAPEX_tot_CHF = CAPEX_compression_GasOn_CHF; %[CHF]
conditioning.capture(i).compression_GasOn.costs.AIC_CHF_per_y = CAPEX_compression_GasOn_CHF_per_y; %[CHF]
conditioning.capture(i).compression_GasOn.costs.OPEX_fix_CHF_per_y = OPEX_fix_compression_GasOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOn.costs.OPEX_var_CHF_per_y = OPEX_var_compression_GasOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOn.costs.AOC_CHF_per_y = OPEX_compression_GasOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOn.costs.TAC_CHF_per_y = year_costs_compression_GasOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOn.costs.LC_CHF_per_t = levelized_costs_compression_GasOn_CHF_per_t; %[CHF/t]
conditioning.capture(i).compression_GasOn.gamma_comp_t_per_t = gamma_electricity_t_per_kWh.*conditioning.E_compression_GasOn_kWh_per_t; %[t/t]

conditioning.capture(i).compression_GasOff.costs.CAPEX_tot_CHF = CAPEX_compression_GasOff_CHF; %[CHF]
conditioning.capture(i).compression_GasOff.costs.AIC_CHF_per_y = CAPEX_compression_GasOff_CHF_per_y; %[CHF]
conditioning.capture(i).compression_GasOff.costs.OPEX_fix_CHF_per_y = OPEX_fix_compression_GasOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOff.costs.OPEX_var_CHF_per_y = OPEX_var_compression_GasOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOff.costs.AOC_CHF_per_y = OPEX_compression_GasOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOff.costs.TAC_CHF_per_y = year_costs_compression_GasOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_GasOff.costs.LC_CHF_per_t = levelized_costs_compression_GasOff_CHF_per_t; %[CHF/t]
conditioning.capture(i).compression_GasOff.gamma_comp_t_per_t = gamma_electricity_t_per_kWh.*conditioning.E_compression_GasOff_kWh_per_t; %[t/t]

conditioning.capture(i).compression_DenseOn.costs.CAPEX_tot_CHF = CAPEX_compression_DenseOn_CHF; %[CHF]
conditioning.capture(i).compression_DenseOn.costs.AIC_CHF_per_y = CAPEX_compression_DenseOn_CHF_per_y; %[CHF]
conditioning.capture(i).compression_DenseOn.costs.OPEX_fix_CHF_per_y = OPEX_fix_compression_DenseOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOn.costs.OPEX_var_CHF_per_y = OPEX_var_compression_DenseOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOn.costs.AOC_CHF_per_y = OPEX_compression_DenseOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOn.costs.TAC_CHF_per_y = year_costs_compression_DenseOn_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOn.costs.LC_CHF_per_t = levelized_costs_compression_DenseOn_CHF_per_t; %[CHF/t]
conditioning.capture(i).compression_DenseOn.gamma_comp_t_per_t = gamma_electricity_t_per_kWh.*conditioning.E_compression_DenseOn_kWh_per_t; %[t/t]

conditioning.capture(i).compression_DenseOff.costs.CAPEX_tot_CHF = CAPEX_compression_DenseOff_CHF; %[CHF]
conditioning.capture(i).compression_DenseOff.costs.AIC_CHF_per_y = CAPEX_compression_DenseOff_CHF_per_y; %[CHF]
conditioning.capture(i).compression_DenseOff.costs.OPEX_fix_CHF_per_y = OPEX_fix_compression_DenseOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOff.costs.OPEX_var_CHF_per_y = OPEX_var_compression_DenseOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOff.costs.AOC_CHF_per_y = OPEX_compression_DenseOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOff.costs.TAC_CHF_per_y = year_costs_compression_DenseOff_CHF_per_y; %[CHF/y]
conditioning.capture(i).compression_DenseOff.costs.LC_CHF_per_t = levelized_costs_compression_DenseOff_CHF_per_t; %[CHF/t]
conditioning.capture(i).compression_DenseOff.gamma_comp_t_per_t = gamma_electricity_t_per_kWh.*conditioning.E_compression_DenseOff_kWh_per_t; %[t/t]
end

