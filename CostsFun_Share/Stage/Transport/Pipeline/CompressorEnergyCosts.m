function [ECcomp_EUR_per_y] = CompressorEnergyCosts(Wcomp_MW, H_h, C_el_EUR_per_MWh)
%Equation (B.36) in Supplementary Material
%OUTPUT: ECcomp_EUR_per_y = energy costs of compressor [EUR.y-1]
%INPUT: Wcomp_MW = compressor capacity [MW]
%       H_h = operating hours within a year [h]
%       C_el_EUR_per_MWh = electricity costs [EUR.MWh-1]

ECcomp_EUR_per_y = Wcomp_MW.*H_h.*C_el_EUR_per_MWh;

end