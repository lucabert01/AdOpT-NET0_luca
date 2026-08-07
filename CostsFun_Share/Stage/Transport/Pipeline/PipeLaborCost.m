function [Ilab_EUR] = PipeLaborCost(ODNPS_m, L_m, Clab_EUR_per_m2)
%Equation (B.27) in Supplementary Material
%OUTPUT: Ilab_EUR = labour costs for the pipe [EUR]
%INPUT: ODNPS_m = outer diameter of the nominal pipe size [m]
%       L_m = length of the pipe [m]
%       Clab_EUR_per_m2 = labour costs [EUR.m-2]

Ilab_EUR = ODNPS_m.*L_m.*Clab_EUR_per_m2;

end