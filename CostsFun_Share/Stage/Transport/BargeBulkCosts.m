function [Vbarge_m3, mLoadAvg_t, mBarge_t, mTot_t, n_shipments, f_per_y, ...
    nBarge, CAPEX_barge_EUR_per_y, OPEX_harbour_EUR_per_y, OPEX_gasoil_EUR_per_y, ...
    TAC_EUR_per_y, LC_EUR_per_t] = BargeBulkCosts(barge, m_t_per_y, d_km, Vmax_m3, t_roundtrip_h)
%This function calculates the optimized barge size among the discrete range 
% of possible sizes and the corresponding costs for the barge bulk transport
%INPUT: barge = struct containing the barge information
%       m_t_per_y = amount of CO2 to be transported yearly [t.y-1]
%       d_km = distance of the connection [km]
%       Vmax_m3 = maximal volume of the barge on the connection [m3]
%       t_roundtrip_h = roundtrip duration [h]
%OUTPUT: Vbarge_m3 = selected barge volume [m3]
%        mLoadAvg_t = average load of the barge over the year [t]
%        mBarge_t = weight of the empty barge [t]
%        mTot_t = total weight of the barge incl. CO2 [t]
%        n_shipment = number of shipments per year
%        f_per_y = frequency of transportation per year
%        nBarge = number of barges
%        CAPEX_barge_EUR_per_y = yearly CAPEX costs for the barge [EUR.y-1]
%        OPEX_harbour_EUR_per_y = harbour costs per year [EUR.y-1]
%        OPEX_gasoil_EUR_per_y = oil costs per year [EUR.y-1]
%        TAC_EUR_per_y = total annualized costs [EUR.y-1]
%        LC_EUR_per_t = levelized costs [EUR.t-1]

%% Calculate the different barge sizes and their costs

Vbarge_m3 = (3040:380:Vmax_m3)';
mLoadAvg_t = Vbarge_m3.*(0.87444 - 2.5270*1e-4.*d_km);                      %function obtained in Excel
mBarge_t = 0.66447*Vbarge_m3 - 360;                                         %idem
mTot_t = mLoadAvg_t + mBarge_t;

mGasoil_barrel = d_km.*mTot_t.*barge.c_barrel_per_km_per_ttot;

f_1barge_per_y = floor(barge.n_work./(t_roundtrip_h./24));
n_shipments = ceil(m_t_per_y./mLoadAvg_t);

nBarge = ceil(m_t_per_y./(mLoadAvg_t.*f_1barge_per_y));

C_1barge_EUR_per_y = Vbarge_m3.*(1198.9 - 0.15868.*mBarge_t);               %idem
CAPEX_barge_EUR_per_y = C_1barge_EUR_per_y.*nBarge;

OPEX_harbour_EUR_per_y = barge.C_harbour_EUR.*nBarge;
OPEX_gasoil_EUR_per_y = mGasoil_barrel.*n_shipments.*barge.C_fuel_EUR_per_unit;

TAC_EUR_per_y = CAPEX_barge_EUR_per_y + OPEX_harbour_EUR_per_y + OPEX_gasoil_EUR_per_y;

%% Take the minimal total costs
[~,idx] = min(TAC_EUR_per_y(:,2));

Vbarge_m3 = Vbarge_m3(idx);
mLoadAvg_t = mLoadAvg_t(idx);
mBarge_t = mBarge_t(idx);
mTot_t = mTot_t(idx);
n_shipments = n_shipments(idx);
f_per_y = n_shipments;
nBarge = nBarge(idx,:);

C_1barge_EUR_per_y = Vbarge_m3.*(1198.9 - 0.15868.*mBarge_t);
CAPEX_barge_EUR_per_y = C_1barge_EUR_per_y.*nBarge;

OPEX_harbour_EUR_per_y = barge.C_harbour_EUR.*nBarge;
mGasoil_barrel = mGasoil_barrel(idx);
OPEX_gasoil_EUR_per_y = mGasoil_barrel.*n_shipments.*barge.C_fuel_EUR_per_unit;

TAC_EUR_per_y = CAPEX_barge_EUR_per_y + OPEX_harbour_EUR_per_y + OPEX_gasoil_EUR_per_y;

LC_EUR_per_t = TAC_EUR_per_y./m_t_per_y;

end