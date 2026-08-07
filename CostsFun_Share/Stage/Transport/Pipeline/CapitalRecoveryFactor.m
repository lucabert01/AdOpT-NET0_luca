function [CRF] = CapitalRecoveryFactor(r,z)
%Capital recovery factor calculated by eq. (1) in the main text
%OUTPUT: CRF = capital recovery factor [-]
%INPUT: r = discount rate [-]
%       z = lifetime of equipment [y]

CRF = r./(1 - ((1+r).^(-z)));

end