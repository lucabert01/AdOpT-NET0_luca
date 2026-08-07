function [C] = SevenTenthRule(C0, S, S0, R)
%This function calculates the estimation for a process equipment cost based
%on known equipment.
%INPUT: C0 = cost for equipment of capacity S0
%       S = capacity of the equipment for which one wants to calculate the cost
%       S0 = capacity of known equipment
%       R = cost exponent according to Remer and Chai
%OUTPUT: C = cost of equipment of capacity S

C = C0.*(S./S0).^R;

end

