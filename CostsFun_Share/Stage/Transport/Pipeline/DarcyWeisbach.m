function [f] = DarcyWeisbach(epsilon_m, IDNPS_m, Re)
%Equation (B.10) in Supplementary Material
%OUTPUT: f = Darcy-Weisbach friction factor [-]
%INPUT: epsilon_m = roughness height [m]
%       IDNPS_m = inner diameter of the nominal pipe size [m]
%       Re = Reynolds number [-]

f = (1./(-1.8.*log10((epsilon_m./IDNPS_m./3.7).^1.11 + 6.9./Re))).^2;

end