% Create enumeration structure
function enum = createEnum(fields)
    %params = [fields;num2cell(1:length(fields))];
    %enum = struct(params{:});
    enum = cell2struct(num2cell(1:length(fields)),fields,2);
end