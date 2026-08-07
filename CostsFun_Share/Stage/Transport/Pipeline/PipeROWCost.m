function [IROW_EUR] = PipeROWCost(L_m, CROW_EUR_per_m)
%Equation (B.28) in Supplementary Material
%OUTPUT: IROW_EUR = right-of-way costs for the pipe [EUR]
%INPUT: L_m = length of the pipe [m]
%       CROW_EUR_per_m = right-of-way costs [EUR.m-1]

IROW_EUR = L_m.*CROW_EUR_per_m;

end