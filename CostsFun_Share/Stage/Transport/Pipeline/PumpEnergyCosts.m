function [ECpump_EUR_per_y] = PumpEnergyCosts(Wpump_MW, Npump, H_h, C_el_EUR_per_MWh, varargin)
%Equation (B.37) in Supplementary Material
%OUTPUT: ECpump_EUR_per_y = energy costs of pumping stations [EUR.y-1]
%INPUT: Wpump_MW = pumping station capacity [MW]
%       Npump = number of pumping stations [-]
%       H_h = operating hours within a year [h]
%       C_el_EUR_per_MWh = electricity costs [EUR.MWh-1]
% opt   W_lastpump_MW = capacity of the last pumping station [MW]

switch nargin
    case 4
        ECpump_EUR_per_y = Npump.*Wpump_MW.*H_h.*C_el_EUR_per_MWh;
    case 5
        W_lastpump_MW = varargin{1};
        ECpump_EUR_per_y = ((Npump-1).*Wpump_MW + W_lastpump_MW).*H_h.*C_el_EUR_per_MWh;
end

switch ECpump_EUR_per_y < 0
    case 1
        ECpump_EUR_per_y = 0;
end

end