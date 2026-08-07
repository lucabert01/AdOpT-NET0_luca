function [Imaterial_EUR] = PipeMaterialCost(t_m, ODNPS_m, L_m, rhoSteel_kg_per_m3, Csteel_EUR_per_kg, SteelFactor)
%Equation (B.26) in Supplementary Material
%OUTPUT: Imaterial_EUR = material costs for the pipe [EUR]
%INPUT: t_m = pipe thickness [m]
%       ODNPS_m = outer diameter of the nominal pipe size [m]
%       L_m = length of the pipe [m]
%       rhoSteel_kg_per_m3 = steel density [kg.m-3]
%       Csteel_EUR_per_kg = steel cost [EUR.kg-1]
%       SteelFactor = steel factor [-]

Imaterial_EUR = t_m.*pi.*(ODNPS_m - t_m).*L_m.*rhoSteel_kg_per_m3.*Csteel_EUR_per_kg.*SteelFactor;

end