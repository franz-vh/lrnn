% Abstract class from which all compound inputs (formed of subinputs)
% inherit

classdef CompoundInput < InputData
    
    % Public constants
    properties (Abstract,Constant)
        SubInputs {mustBeA(SubInputs,"struct")}
    end
    
    % Public constants
    properties (SetAccess = protected)
        SubDims
        SubSizes
    end
    
    % Main public methods
    methods
        % Constructor
        function obj = CompoundInput(~)
            obj.SubDims = cell(1,length(fieldnames(obj.SubInputs)));
            obj.SubSizes = zeros(1,length(fieldnames(obj.SubInputs)));
        end
        % Get subinput batch
        function batch = getSubInput(obj,batch,subinput)
            sample0 = sum(obj.SubSizes(1:subinput-1));
            batch = batch(sample0+1:sample0+obj.SubSizes(subinput),:);
        end
    end
    
    % Data initialization methods
    methods (Access = protected)
        % Input data initialization
        function initData(obj)
            for i = 1:length(obj.SubSizes)
                obj.SubSizes(i) = prod(obj.SubDims{i});
            end
            obj.Size = sum(obj.SubSizes);
            obj.Dims = obj.Size;
        end
    end
    
    % Visualization methods
    methods
        % Input plot generation
        function input_plot = getInputPlot(obj,input)
            n_subinputs = length(obj.SubSizes);
            input_plots = cell(1,n_subinputs);
            heights = zeros(1,n_subinputs);
            widths = zeros(1,n_subinputs);
            depths = zeros(1,n_subinputs);
            for i = 1:n_subinputs
                input_plots{i} = obj.getSubInputPlot(input,i);
                heights(i) = size(input_plots{i},1);
                widths(i) = size(input_plots{i},2);
                depths(i) = size(input_plots{i},3);
            end
            cumwidths = [-1,cumsum(widths)+(0:n_subinputs-1)];
            input_plot = zeros(max(heights),cumwidths(end)-1,max(depths));
            for i = 1:n_subinputs
                input_plot(1:heights(i),cumwidths(i)+2:cumwidths(i+1),:) = input_plots{i}.*ones(heights(i),widths(i),size(input_plot,3));
            end
        end
        % Subinput plot generation
        function input_plot = getSubInputPlot(obj,input,subinput)
            input_plot = InputData.getInputPlotStatic(obj.getSubInput(input,subinput),obj.SubDims{subinput});
        end
    end
    
end