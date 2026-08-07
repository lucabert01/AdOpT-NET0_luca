function [Icompressor_EUR] = CompressorInvestment(Wcomp_MW)
%Equation (B.30) in Supplementary Material
%OUTPUT: Icompressor_EUR = investment costs of compressor [EUR]
%INPUT: Wcomp_MW = compressor capacity [MW]

Icomp0_EUR = 21.9e6; %[EUR]
Wcomp0_MW = 13.0; %[MW]
WcompMAX_MW = 35.0; %[MW]
y = 0.67;
me = 0.9;

n = ceil(Wcomp_MW./WcompMAX_MW);

switch n
    case 0
        W1comp_MW = 0;
    otherwise
        W1comp_MW = Wcomp_MW./n;
end

Icompressor_EUR = Icomp0_EUR.*((W1comp_MW./Wcomp0_MW).^y).*n.^me;

end