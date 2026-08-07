function [MAOP_MPa] = MaximumAllowableOperationPressure(Pinlet_MPa)
%Equation (B.7) in Supplementary Material
%OUTPUT: MAOP_MPa = maximum allowable operating pressure [MPa]
%INPUT: Pinlet_MPa = inlet pressure [MPa]

MAOP_MPa = ceil(Pinlet_MPa.*1.1.*10)./10;

end