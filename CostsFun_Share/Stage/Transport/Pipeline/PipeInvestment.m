function [Ipipe_EUR] = PipeInvestment(Imaterial_EUR, Ilab_EUR, IROW_EUR, Imisc_EUR)
%Equation (B.25) in Supplementary Material
%OUTPUT: Ipipe_EUR = investment costs for the pipe [EUR]
%INPUT: Imaterial_EUR = material costs for the pipe [EUR]
%       Ilab_EUR = labour costs for the pipe [EUR]
%       IROW_EUR = right-of-way costs for the pipe [EUR]
%       Imisc_EUR = miscellaneous costs for the pipe [EUR]

Ipipe_EUR = Imaterial_EUR + Ilab_EUR + IROW_EUR + Imisc_EUR;

end