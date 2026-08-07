function [t_m] = PipeThickness(ODNPS_m, MAOP_MPa, S_MPa, F, E, CA_m, dtRatio)
%Equation (B.6) in Supplementary Material
%OUTPUT: t_m = pipe thickness [m]
%INPUT: ODNPS_m = outer diameter of the nominal pipe size [m]
%       MAOP_MPa = maximum allowable operating pressure [MPa]
%       S_MPa = minimum yield stress [MPa]
%       F = design factor of the pipeline [-]
%       E = longitudinal joint factor [-]
%       CA_m = corrosion allowance [m]
%       dtRatio = ODNPS/t ratio [-]

t_m = ODNPS_m.*MAOP_MPa./(2.*S_MPa.*F.*E) + CA_m;

switch t_m/ODNPS_m < dtRatio
    case 1
        t_m = ODNPS_m.*dtRatio;
end

%Round to the next 0.5 mm
t_m = ceil(t_m.*2000)./2000; 

end