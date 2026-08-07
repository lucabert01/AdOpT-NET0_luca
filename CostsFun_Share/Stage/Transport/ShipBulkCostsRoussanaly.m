function [C_opt_tCO2, C_ship_real_tCO2, n_shipments, f_per_y, n_ship, C_stor_t, ...
    CAPEX_stor_EUR, CAPEX_stor_EUR_per_y, OPEX_stor_EUR_per_y, CAPEX_load_EUR, ...
    CAPEX_load_EUR_per_y, OPEX_load_EUR_per_y, CAPEX_1ship_EUR, CAPEX_ship_EUR, ...
    CAPEX_ship_EUR_per_y, cFuel_t_per_tCO2_per_km, OPEX_fix_ship_EUR_per_y, ...
    OPEX_fuel_1shipment_EUR_per_y, OPEX_fuel_ship_EUR_per_y, OPEX_harbor_ship_EUR_per_y, ...
    TYC_EUR_per_y, LC_EUR_per_t] = ShipBulkCostsRoussanaly(shipXbarg, r, m_t_per_y, d_km, t_roundtrip_h, DataRoussanaly, TableRoussanaly)
%This function calculates the optimized ship size among the discrete range 
% of possible sizes and the corresponding costs for the ship bulk transport
%INPUT: shipXbarg = struct containing the ship information
%       m_t_per_y = amount of CO2 to be transported yearly [t.y-1]
%       d_km = distance of the connection [km]
%       t_roundtrip_h = roundtrip duration [h]
%OUTPUT: C_opt_tCO2 = optimal ship size [tCO2]
%        C_ship_real_tCO2 = real capacity of the optimal ship size [tCO2]
%        n_shipment = number of shipments per year
%        f_per_y = frequency of transportation per year
%        n_ship = number of ships
%        C_stor_t = storage capacity [t]
%        CAPEX_stor_EUR = CAPEX for storage [EUR]
%        CAPEX_stor_EUR_per_y = annualized CAPEX for storage [EUR.y-1]
%        OPEX_stor_EUR = OPEX for storage [EUR.y-1]
%        CAPEX_load_EUR = CAPEX for loading [EUR]
%        CAPEX_load_EUR_per_y = annualized CAPEX for loading [EUR.y-1]
%        OPEX_load_EUR = OPEX for loading [EUR.y-1]
%        CAPEX_1ship_EUR = CAPEX for 1 ship [EUR]
%        CAPEX_ship_EUR = CAPEX for all ships [EUR]
%        CAPEX_ship_EUR_per_y = annualized CAPEX for all ships [EUR.y-1]
%        cFuel_t_per_tCO2_per_km = specific fuel consumption [t.tCO2-1.km-1]
%        OPEX_fix_ship_EUR_per_y = fixed OPEX for all ships [EUR.y-1]
%        OPEX_fuel_1shipment_EUR_per_y = fuel costs for 1 shipment [EUR.y-1]
%        OPEX_fuel_ship_EUR_per_y = fuel OPEX [EUR.y-1]
%        OPEX_harbor_ship_EUR_per_y = harbor OPEX [EUR.y-1]
%        TYC_EUR_per_y = total yearly costs [EUR.y-1]
%        LC_EUR_per_t = levelized costs [EUR.t-1]

%% Data

CRF = r./(1-(1+r).^-shipXbarg.tau_y);
C_fuel_EUR_per_t = shipXbarg.C_fuel_EUR_per_t;
eta_ship = shipXbarg.eta_ship;
delta_Stor = shipXbarg.delta_Stor;
DACE_2017_2019 = 103/105;
ChemieTechnik_2019_2021 = 111.3/107.0;
CostIndex_2017_2021 = DACE_2017_2019.*ChemieTechnik_2019_2021;

p = shipXbarg.p;

switch p
    case 8
        cStor_EUR_per_tCO2 = DataRoussanaly.cStor_EUR_per_tCO2(1);
        Table = TableRoussanaly(TableRoussanaly.ShipCapacity_tCO2<=5e4,[1:2 4]);
    case 16
        cStor_EUR_per_tCO2 = DataRoussanaly.cStor_EUR_per_tCO2(2);
        Table = TableRoussanaly(TableRoussanaly.ShipCapacity_tCO2<=1e4,[1 3:4]);
end
Table.Properties.VariableNames(2) = {'CAPEX_EUR'};

%% Initialization

C_ship_tCO2 = zeros(1,height(Table));
C_ship_real_tCO2 = zeros(1,height(Table));
n_shipments = zeros(1,height(Table));
n_ship = zeros(1,height(Table));
C_stor_t = zeros(1,height(Table));
CAPEX_stor_EUR = zeros(1,height(Table));
OPEX_stor_EUR_per_y = zeros(1,height(Table));
CAPEX_load_EUR_per_y = m_t_per_y.*DataRoussanaly.cLoad_EUR_per_tCO2.*CostIndex_2017_2021; %[EUR_2021/y]
% CAPEX_load_EUR = CAPEX_load_EUR_per_y./DataRoussanaly.CRF; %[EUR_2021]
CAPEX_load_EUR = CAPEX_load_EUR_per_y./CRF; %[EUR_2021]
OPEX_load_EUR_per_y = CAPEX_load_EUR_per_y.*DataRoussanaly.mu_Load; %[EUR_2021/y]
CAPEX_1ship_EUR = zeros(1,height(Table));
CAPEX_ship_EUR = zeros(1,height(Table));
cFuel_t_per_tCO2_per_km = zeros(1,height(Table));
OPEX_fix_ship_EUR_per_y = zeros(1,height(Table));
OPEX_fuel_1shipment_EUR_per_y = zeros(1,height(Table));
OPEX_fuel_ship_EUR_per_y = zeros(1,height(Table));
OPEX_harbor_ship_EUR_per_y = zeros(1,height(Table));
TYC_EUR_per_y = zeros(1,height(Table));

%% Calculation of the costs for different sizes

for k = 1:height(Table)
    C_ship_tCO2(k) = Table.ShipCapacity_tCO2(k);
    %     C_ship_real_tCO2(k) = C_ship_tCO2(k).*DataRoussanaly.eta_ship; %[t]
    C_ship_real_tCO2(k) = C_ship_tCO2(k).*eta_ship; %[t]
    n_shipments(k) = ceil(m_t_per_y/C_ship_real_tCO2(k));
    n_ship(k) = ceil(n_shipments(k).*t_roundtrip_h./DataRoussanaly.tOp_h_per_y);
    %     C_stor_t(k) = DataRoussanaly.delta_Stor.*C_ship_tCO2(k); %[t]
    C_stor_t(k) = delta_Stor.*C_ship_tCO2(k); %[t]

    CAPEX_stor_EUR(k) = C_stor_t(k).*cStor_EUR_per_tCO2.*CostIndex_2017_2021; %[EUR_2021]
    OPEX_stor_EUR_per_y(k) = CAPEX_stor_EUR(k).*DataRoussanaly.mu_Stor; %[EUR_2021/y]
    CAPEX_1ship_EUR(k) = Table.CAPEX_EUR(k).*CostIndex_2017_2021; %[EUR_2021]
    CAPEX_ship_EUR(k) = n_ship(k).*CAPEX_1ship_EUR(k); %[EUR_2021]
    cFuel_t_per_tCO2_per_km(k) = Table.cFuel_t_per_tCO2_per_km(k); %[t/t/km]
    OPEX_fix_ship_EUR_per_y(k) = CAPEX_ship_EUR(k).*DataRoussanaly.mu_ship; %[EUR_2021/y]
    %     OPEX_fuel_1shipment_EUR_per_y(k) = cFuel_t_per_tCO2_per_km(k).*DataRoussanaly.C_fuel_EUR_per_t.*...
    %         d_km.*C_ship_tCO2(k); %[EUR/ship]
    OPEX_fuel_1shipment_EUR_per_y(k) = cFuel_t_per_tCO2_per_km(k).*C_fuel_EUR_per_t.*...
        d_km.*C_ship_tCO2(k).*CostIndex_2017_2021; %[EUR_2021/ship]
    OPEX_fuel_ship_EUR_per_y(k) = n_shipments(k).*OPEX_fuel_1shipment_EUR_per_y(k); %[EUR_2021/y]
    OPEX_harbor_ship_EUR_per_y(k) = n_shipments(k).*2.*DataRoussanaly.C_harbor_EUR_per_tCO2.*C_ship_tCO2(k).*CostIndex_2017_2021; %[EUR_2021/y]

    %     TYC_EUR_per_y(k) = DataRoussanaly.CRF.*(CAPEX_stor_EUR(k) + CAPEX_load_EUR + CAPEX_ship_EUR(k)) +...
    %         OPEX_stor_EUR_per_y(k) + OPEX_load_EUR_per_y + OPEX_fix_ship_EUR_per_y(k) + ...
    %         OPEX_fuel_ship_EUR_per_y(k) + OPEX_harbor_ship_EUR_per_y(k);
    TYC_EUR_per_y(k) = CRF.*(CAPEX_stor_EUR(k) + CAPEX_load_EUR + CAPEX_ship_EUR(k)) +...
        OPEX_stor_EUR_per_y(k) + OPEX_load_EUR_per_y + OPEX_fix_ship_EUR_per_y(k) + ...
        OPEX_fuel_ship_EUR_per_y(k) + OPEX_harbor_ship_EUR_per_y(k);
end

%% Choice of the minimal costs among all sizes

[~,idx] = min(TYC_EUR_per_y);

C_opt_tCO2 = C_ship_tCO2(idx);
C_ship_real_tCO2 = C_ship_real_tCO2(idx);
n_shipments = n_shipments(idx);
f_per_y = n_shipments;
n_ship = n_ship(idx);
C_stor_t = C_stor_t(idx);
CAPEX_stor_EUR = CAPEX_stor_EUR(idx);
% CAPEX_stor_EUR_per_y = DataRoussanaly.CRF.*CAPEX_stor_EUR;
CAPEX_stor_EUR_per_y = CRF.*CAPEX_stor_EUR;
% CAPEX_stor_EUR_per_y = CAPEX_stor_EUR_per_y(idx);
OPEX_stor_EUR_per_y = OPEX_stor_EUR_per_y(idx);
% CAPEX_load_EUR_per_y = DataRoussanaly.CRF.*CAPEX_load_EUR_per_y;
%CAPEX_load_EUR = CAPEX_load_EUR
%OPEX_load_EUR_per_y = OPEX_load_EUR_per_y
CAPEX_1ship_EUR = CAPEX_1ship_EUR(idx);
CAPEX_ship_EUR = CAPEX_ship_EUR(idx);
% CAPEX_ship_EUR_per_y = DataRoussanaly.CRF.*CAPEX_ship_EUR;
CAPEX_ship_EUR_per_y = CRF.*CAPEX_ship_EUR;
cFuel_t_per_tCO2_per_km = cFuel_t_per_tCO2_per_km(idx);
OPEX_fix_ship_EUR_per_y = OPEX_fix_ship_EUR_per_y(idx);
OPEX_fuel_1shipment_EUR_per_y = OPEX_fuel_1shipment_EUR_per_y(idx);
OPEX_fuel_ship_EUR_per_y = OPEX_fuel_ship_EUR_per_y(idx);
OPEX_harbor_ship_EUR_per_y = OPEX_harbor_ship_EUR_per_y(idx);
TYC_EUR_per_y = TYC_EUR_per_y(idx);
LC_EUR_per_t = TYC_EUR_per_y./m_t_per_y;

end