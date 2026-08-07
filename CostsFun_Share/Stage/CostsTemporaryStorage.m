function [temporaryStorage, finance] = CostsTemporaryStorage(temporaryStorage, conditioning, finance, i)
%This function calculates the costs for temporary storage
%INPUT: temporaryStorage = struct containing the temporary storage information
%       conditioning = struct containing the conditioning information
%       finance = struct containing the finance information
%       i = plant index
%OUTPUT: temporaryStorage = struct containing the temporary storage information
%           capacity_storageTank_t = needed capacity for the storage tank [t]
%           CAPEX_tot_CHF = total CAPEX [CHF] (initial investment)
%           AIC_CHF_per_y = annualized investment costs [CHF/y]
%           AOC_CHF_per_y = annualized operating costs [CHF/y]
%           TAC_CHF_per_y = total annualized costs [CHF/y]
%           LC_CHF_per_t = levelized costs for intermediate storage [CHF/t]

%% Data

m_liq_t_per_y = conditioning.capture(i).m_liq15barg_t_per_y(2); %[t/y]
S_t = max(m_liq_t_per_y.*temporaryStorage.cap_time_d./365,temporaryStorage.Smin_t); %[t]

r = finance.r;

finance.temporaryStorage.a_CAPEX = r./(1-(1+r).^-temporaryStorage.lifetime);

%% CAPEX

%Capacity of intermediary storage

C_intermediaryStorage_CHF = interp1(temporaryStorage.S0_t,temporaryStorage.CapEx0_CHF,S_t,"linear","extrap");
C_intermediaryStorage_CHF_per_y = C_intermediaryStorage_CHF.*finance.temporaryStorage.a_CAPEX; %[CHF/y]

CAPEX_tot = C_intermediaryStorage_CHF; %[CHF]
AIC_CHF_per_y = C_intermediaryStorage_CHF_per_y; %[CHF/y]

%% OPEX

AOC_CHF_per_y = CAPEX_tot.*temporaryStorage.OPEX_perc./100;

%% Yearly costs

TAC_CHF_per_y = AIC_CHF_per_y + AOC_CHF_per_y; %[CHF/y]

LC_CHF_per_t = TAC_CHF_per_y./m_liq_t_per_y; %[CHF/t]

%% Rename and resize matrices

temporaryStorage.capture(i).S_t = S_t; %[t]
temporaryStorage.capture(i).costs.CAPEX_CHF = CAPEX_tot; %[CHF]
temporaryStorage.capture(i).costs.AIC_CHF_per_y = AIC_CHF_per_y; %[CHF]
temporaryStorage.capture(i).costs.AOC_CHF_per_y = AOC_CHF_per_y; %[CHF/y]
temporaryStorage.capture(i).costs.TAC_CHF_per_y = TAC_CHF_per_y; %[CHF/y]
temporaryStorage.capture(i).costs.LC_CHF_per_t = LC_CHF_per_t; %[CHF/t]

switch size(LC_CHF_per_t,2)
    case 1
       temporaryStorage.capture(i).costs.TAC_CHF_per_y = temporaryStorage.capture(i).costs.TAC_CHF_per_y.*ones(1,3);
       temporaryStorage.capture(i).costs.LC_CHF_per_t = temporaryStorage.capture(i).costs.LC_CHF_per_t.*ones(1,3);
end

end

