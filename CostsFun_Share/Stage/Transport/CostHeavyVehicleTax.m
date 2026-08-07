function [C_HGVT] = CostHeavyVehicleTax(truck, truckBatch, st, go, i)
%This function calculates the heavy good vehicle tax depending on the
%country crossed and the distance covered
%INPUT: truck = struct containing truck information
%       st = start index
%       go = goal index
%       i = plant index
%OUTPUT: C_HGVT = total cost heavy good vehicle tax [CHF/y]

n_shipment = truckBatch.capture(i).start(st).goal(go).n_shipment;
n_truck = truckBatch.capture(i).start(st).goal(go).n_truck;
d_CH_tot_km_1way = truck.start(st).goal(go).d_CH_km.*n_shipment;
d_BE_tot_km_1way = truck.start(st).goal(go).d_BE_km.*n_shipment;
d_D_tot_km = truck.start(st).goal(go).d_D_km.*2.*n_shipment;
d_AT_tot_km = truck.start(st).goal(go).d_AT_km.*2.*n_shipment;
d_IS_tot_km = truck.start(st).goal(go).d_IS_km.*2.*n_shipment;
b_EU = truck.start(st).goal(go).b_EU;
b_NO = truck.start(st).goal(go).b_NO;

C_HGVT = n_truck.*(b_EU.*truck.C_LSVA_EU_CHF_per_y + b_NO.*truck.C_LSVA_NO_CHF_per_y) + ...
    d_CH_tot_km_1way.*(truck.C_LSVA_20t_CH_CHF_per_km + truck.C_LSVA_40t_CH_CHF_per_km) + ...
    d_BE_tot_km_1way.*(truck.C_LSVA_20t_BE_CHF_per_km + truck.C_LSVA_40t_BE_CHF_per_km) + ...
    d_D_tot_km.*truck.C_LSVA_D_CHF_per_km + d_AT_tot_km.*truck.C_LSVA_AT_CHF_per_km + ...
    d_IS_tot_km.*truck.C_LSVA_IS_CHF_per_km;

end



