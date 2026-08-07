function [Table,Data] = OpenRoussanalyTable(filenameRoussanalyTable)
%This function opens the Table based on Roussanaly study Energies 2021

Table = readtable(filenameRoussanalyTable,'Sheet','Table');
%Table(1:2,:) = [];
Table.Properties.VariableNames = {'ShipCapacity_tCO2','CAPEX_7barg_EUR','CAPEX_15barg_EUR','cFuel_t_per_tCO2_per_km'};
RawData = readtable(filenameRoussanalyTable,'Sheet','Data');
Data_text = readcell(filenameRoussanalyTable,'Sheet','Data');
RawData(1:3,:) = [];
Data_text(2:4,:) = [];

Data = struct;
for i = 1:height(RawData)
    Data.(Data_text{i+1,2}) = table2array(RawData(i,[false false false ~isnan(table2array(RawData(i,4:6)))]));
end

end

