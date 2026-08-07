function [IDNPS_m] = InnerDiameter(ODNPS_m, t_m)
%Equation (B.11) in Supplementary Material
%OUTPUT: IDNPS_m = inner diameter of the nominal pipe size [m]
%INPUT: ODNPS_m = outer diameter of the nominal pipe size [m]
%       t_m = pipe thickness [m]

IDNPS_m = ODNPS_m - 2.*t_m;

end