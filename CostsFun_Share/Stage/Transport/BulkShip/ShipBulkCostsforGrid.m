function [ShipBulkOpt] = ShipBulkCostsforGrid(m_t_per_y_lin, d_km_lin, shipBulk, ...
    finance, filenameRoussanalyTable)

%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[TableRoussanaly,DataRoussanaly] = OpenRoussanalyTable(filenameRoussanalyTable);
r = finance.r;
v_km_per_h = shipBulk.v_km_per_h;
ship.tau_y = shipBulk.tau_y;
ship.C_fuel_EUR_per_t = shipBulk.C_fuel_EUR_per_t;
ship.eta_ship = shipBulk.eta_ship;
ship.delta_Stor = shipBulk.delta_Stor;

[M, D] = meshgrid(m_t_per_y_lin, d_km_lin);

M_lin = reshape(M,[numel(M),1,1]);
D_lin = reshape(D,[numel(D),1,1]);

t_transport_h = D_lin./v_km_per_h; %[h]
t_roundtrip_h = DataRoussanaly.tDep_h + 2.*t_transport_h + DataRoussanaly.tArr_h; %[h]

Data = table(M_lin, D_lin, t_roundtrip_h, 'VariableNames',{'m_t_per_y','d_km', 't_roundtrip_h'});

pressure = [8 16];
ShipBulkOpt = struct;

%% Alg

for pr = 1:2
    for i = 1:height(Data)

        ship.p = pressure(pr);
        m_t_per_y = Data.m_t_per_y(i);
        d_km = Data.d_km(i);
        t_rdtrip_h = Data.t_roundtrip_h(i);

[C_opt_tCO2, C_ship_real_tCO2, n_shipments, f_per_y, n_ship, C_stor_t, ...
    CAPEX_stor_EUR, CAPEX_stor_EUR_per_y, OPEX_stor_EUR_per_y, CAPEX_load_EUR, ...
    CAPEX_load_EUR_per_y, OPEX_load_EUR_per_y, CAPEX_1ship_EUR, CAPEX_ship_EUR, ...
    CAPEX_ship_EUR_per_y, cFuel_t_per_tCO2_per_km, OPEX_fix_ship_EUR_per_y, ...
    OPEX_fuel_1shipment_EUR_per_y, OPEX_fuel_ship_EUR_per_y, OPEX_harbor_ship_EUR_per_y, ...
    TYC_EUR_per_y, LC_EUR_per_t] = ShipBulkCostsRoussanaly(ship, r, m_t_per_y, d_km, t_rdtrip_h, DataRoussanaly, TableRoussanaly);

        Data.C_opt_tCO2(i) = C_opt_tCO2;
        Data.C_ship_real_tCO2(i) = C_ship_real_tCO2;
        Data.n_shipments(i) = n_shipments;
        Data.f_per_y(i) = f_per_y;
        Data.n_ship(i) = n_ship;
        Data.C_stor_t(i) = C_stor_t;
        Data.CAPEX_EUR(i) = CAPEX_stor_EUR + CAPEX_load_EUR + CAPEX_ship_EUR;
        Data.OPEX_EUR_per_y(i) = OPEX_stor_EUR_per_y + OPEX_load_EUR_per_y + ...
            OPEX_fix_ship_EUR_per_y + OPEX_fuel_ship_EUR_per_y + OPEX_harbor_ship_EUR_per_y;
        Data.TYC_EUR_per_y(i) = TYC_EUR_per_y;
        Data.LC_EUR_per_t(i) = LC_EUR_per_t;

    end
    ShipBulkOpt(pr).Data = Data;

end

end