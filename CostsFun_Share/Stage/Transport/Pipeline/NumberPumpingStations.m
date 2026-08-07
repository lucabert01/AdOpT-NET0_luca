function [Npump] = NumberPumpingStations(L_m, Lpump_m)
%Equation (B.22) in Supplementary Material
%OUTPUT: Npump = number of pumping stations [-]
%INPUT: L_m = length of the pipe [m]
%       L_pump_m = maximum distance between pumping stations [m]

Npump = floor(L_m./Lpump_m);

end