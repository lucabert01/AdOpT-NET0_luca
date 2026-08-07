function [Lpump_m] = MaximumDistanceBetweenPumpingStations(Pinlet_Pa, Poutlet_Pa, DeltaPact_Pa_per_m)
%Equation (B.21) in Supplementary Material
%OUTPUT: L_pump_m = maximum distance between pumping stations [m]
%INPUT: Pinlet_Pa = inlet pressure [Pa]
%       Poutlet_Pa = outlet pressure [Pa]
%       DeltaPact_Pa_per_m = actual pressure drop [Pa.m-1]

Lpump_m = (Pinlet_Pa - Poutlet_Pa)./DeltaPact_Pa_per_m;

end