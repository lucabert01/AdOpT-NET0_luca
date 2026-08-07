filenameRoussanalyTable = 'Roussanaly_Table4.xlsx';
[TableRoussanaly,DataRoussanaly] = OpenRoussanalyTable(filenameRoussanalyTable);

Table = TableRoussanaly(TableRoussanaly.ShipCapacity_tCO2<=1e4,[1 3:4]);
Table.Properties.VariableNames(2) = {'CAPEX_EUR'};

