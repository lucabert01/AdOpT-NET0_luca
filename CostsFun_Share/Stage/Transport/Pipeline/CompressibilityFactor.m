function [Zave] = CompressibilityFactor(Pave_Pa, Tave_K, DataFluidProperties277K, DataFluidProperties288K)
%Equation (B.5) in Supplementary Material
%OUTPUT: Zave = average compressibility factor [-]
%INPUT: Pave_Pa = average pressure [Pa]
%       T_ave_K = average temperature [K]
%       DataFluidProperties277K = Table with fluid properties at 277 K
%       DataFluidProperties288K = Table with fluid properties at 288 K

switch Tave_K - 273.15
    case 4
       Zref = DataFluidProperties277K.CompressibilityFactorZ;
       Pref_MPa = DataFluidProperties277K.Pressure_MPa_;
    case 15
       Zref = DataFluidProperties288K.CompressibilityFactorZ;
       Pref_MPa = DataFluidProperties288K.Pressure_MPa_;
end

Zave = interp1(Pref_MPa,Zref,Pave_Pa./1e6,'linear','extrap');

end