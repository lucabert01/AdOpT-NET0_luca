function [Ipump_EUR] = PumpInvestment(Wpump_MW, Npump, phase, varargin)
%Equation (B.31) & (B.32) in Supplementary Material
%OUTPUT: Ipump_EUR = investment costs of pumping stations [EUR]
%INPUT: Wpump_MW = pumping station capacity [MW]
%       Npump = number of pumping stations [-]
%       phase = (1) gas (2) liquid
% opt   W_lastpump_MW = capacity of the last pumping station [MW]

Ipump0_EUR = 74.3e3;
WpumpMAX_MW = 2.0;

%For all pumps before the last

switch phase
    case 1 %gas
        I1pump_EUR = CompressorInvestment(Wpump_MW);
    case 2 %liquid
        n = ceil(Wpump_MW./WpumpMAX_MW);
        I1pump_EUR = Ipump0_EUR.*((Wpump_MW.*1e3).^0.58).*n.^0.32;
end

switch nargin
    case 3
        Ipump_EUR = Npump.*I1pump_EUR;
    case 4
        %For the last pump
        W_lastpump_MW = varargin{1};

        switch phase
            case 1 %gas
                I_lastpump_EUR = CompressorInvestment(W_lastpump_MW);
            case 2 %liquid
                n_last_pump = ceil(W_lastpump_MW./WpumpMAX_MW);
                I_lastpump_EUR = Ipump0_EUR.*((W_lastpump_MW.*1e3).^0.58).*n_last_pump.^0.32;
        end
        %Overall

        Ipump_EUR = (Npump-1).*I1pump_EUR + I_lastpump_EUR;
end

switch Ipump_EUR < 0
    case 1
        Ipump_EUR = 0;
end

end