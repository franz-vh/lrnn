% Abstract class from which all dataset inputs inherit

classdef DataSet < InputData
    
    % Public constants (abstract)
    properties (Abstract,Constant)
        Path {mustBeA(Path,["char","string"])}
        Classifs {mustBeA(Classifs,"struct")}
        AllLbls {mustBeA(AllLbls,"cell")}
    end
    
    % Main public methods
    methods
        % Constructor
        function obj = DataSet(params)
            obj.Data.splitprops = params.splitprops;
            obj.initData(params);
            obj.restart(params);
        end
        % Restart state
        function restart(obj,params)
            obj.Stt.splitset = obj.Data.splitsets{params.splitset};
            obj.Stt.nxtsmpls = [];
            obj.Stt.nxtlbls = [];
        end
    end
    
    % Data initialization methods (abstract)
    methods (Abstract,Access = protected)
        % Input data initialization
        initData(obj)
        % Read dataset
        readDataset(obj)
        % Normalize input between 0.1 and 0.9
        normData(obj)
        % Split data into sets
        splitInSets(obj)
    end

end