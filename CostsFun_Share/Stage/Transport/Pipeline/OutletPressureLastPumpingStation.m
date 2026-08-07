function [Poutlet_last_pump_Pa] = OutletPressureLastPumpingStation(Poutlet_Pa, L_m, Lpump_m, Npump, DeltaPact_Pa_per_m)
%Equation (B.23) in Supplementary Material
%OUTPUT: Poutlet_last_pump_Pa = outlet pressure of the last pump [Pa]
%INPUT: Poutlet_Pa = outlet pressure [Pa]
%       L_m = length of the pipe [m]
%       L_pump_m = maximum distance between pumping stations [m]
%       Npump = number of pumping stations [-]
%       DeltaPact_Pa_per_m = actual pressure drop [Pa.m-1]


Poutlet_last_pump_Pa = Poutlet_Pa + (L_m - Lpump_m.*Npump).*DeltaPact_Pa_per_m;

end