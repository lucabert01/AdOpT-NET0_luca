function [Imisc_EUR] = PipeMiscellaneous(Imaterial_EUR, Ilab_EUR, mu_misc)
%Equation (B.29) in Supplementary Material
%OUTPUT: Imisc_EUR = miscellaneous costs for the pipe [EUR]
%INPUT: Imaterial_EUR = material costs for the pipe [EUR]
%       Ilab_EUR = labour costs for the pipe [EUR]
%       mu_misc = miscellaneous cost ratio [-]

Imisc_EUR = mu_misc.*(Imaterial_EUR + Ilab_EUR);

end