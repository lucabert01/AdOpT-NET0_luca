function [OC_EUR_per_y] = OpAndM(I_EUR, muOM)
%Equations (B.33)-(B.35) in Supplementary Material
%OUTPUT: OC_EUR_per_y = costs for operation and maintenance [EUR.y-1]
%INPUT: I_EUR = investment costs [EUR]
%       muOM = operation and maintenance cost ratio [-]

OC_EUR_per_y = I_EUR.*muOM;

end