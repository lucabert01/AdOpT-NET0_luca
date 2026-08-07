barge1 = bargeBulk;

Vbarge_m3 = (3040:380:Vmax_m3)';
mLoadAvg_t = Vbarge_m3.*(0.87444 - 2.5270*1e-4.*d_km);
mBarge_t = 0.66447*Vbarge_m3 - 360;
mTot_t = mLoadAvg_t + mBarge_t;

mGasoil_barrel = d_km.*mTot_t.*barge1.c_barrel_per_km_per_ttot;

f_1barge_per_y = floor(barge1.n_work./(t_roundtrip_h./24));
n_shipments = ceil(m_t_per_y./mLoadAvg_t);

nBarge = ceil(m_t_per_y./(mLoadAvg_t.*f_1barge_per_y));

C_1barge_EUR_per_y = Vbarge_m3.*(1198.9 - 0.15868.*mBarge_t);
CAPEX_barge_EUR_per_y = C_1barge_EUR_per_y.*nBarge;

OPEX_harbour_EUR_per_y = barge1.C_harbour_EUR.*nBarge;
OPEX_gasoil_EUR_per_y = mGasoil_barrel.*n_shipments.*barge1.C_fuel_EUR_per_unit;

TYC_EUR_per_y = CAPEX_barge_EUR_per_y + OPEX_harbour_EUR_per_y + OPEX_gasoil_EUR_per_y;

[~,idx] = min(TYC_EUR_per_y(:,2));

Vbarge_m3 = Vbarge_m3(idx);
mLoadAvg_t = mLoadAvg_t(idx);
mBarge_t = mBarge_t(idx);
mTot_t = mTot_t(idx);
n_shipments = n_shipments(idx);
f_per_y = n_shipments;
nBarge = nBarge(idx,:);

C_1barge_EUR_per_y = Vbarge_m3.*(1198.9 - 0.15868.*mBarge_t);
CAPEX_barge_EUR_per_y = C_1barge_EUR_per_y.*nBarge;

OPEX_harbour_EUR_per_y = barge1.C_harbour_EUR.*nBarge;
mGasoil_barrel = mGasoil_barrel(idx);
OPEX_gasoil_EUR_per_y = mGasoil_barrel.*n_shipments.*barge1.C_fuel_EUR_per_unit;

TYC_EUR_per_y = CAPEX_barge_EUR_per_y + OPEX_harbour_EUR_per_y + OPEX_gasoil_EUR_per_y;

LC_EUR_per_t = TYC_EUR_per_y./m_t_per_y;
