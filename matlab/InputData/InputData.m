% Abstract class from which all datasets and other inputs inherit

classdef InputData < handle
    
    % Public constants (abstract)
    properties (Abstract,Constant)
        %Classifs {mustBeA(Classifs,"struct")}
        %AllLbls {mustBeA(AllLbls,"cell")}
        %Range {mustBeVector}
    end
    
     % Public constants (abstract)
    properties (Abstract,SetAccess = protected)
        AllNLbls
    end
    
    % Public constants
    properties (SetAccess = protected)
        Dims
        Size {mustBeInteger,mustBePositive}
    end
    
    % Input data and state variables
    properties (SetAccess = protected)
        Data
        Stt
    end
    
    % Main public methods (abstract)
    methods (Abstract)
        % Restart
        restart(obj,params)
        % Hard restart
        hardRestart(obj,params)
    end
    
    % Main public methods
    methods
        % Get next sample
        function [sample,lbl] = getSample(obj)
            [sample,lbl] = obj.getBatch(1);
        end
        % Get next batch of samples
        function [batch,lbls] = getBatch(obj,batchsize)
            batch = zeros(obj.Size,batchsize);
            lbls = zeros(length(obj.AllLbls),batchsize);
            nxtsmpls = obj.Stt.nxtsmpls;
            nxtlbls = obj.Stt.nxtlbls;
            i1 = 1;
            i2 = i1 + size(nxtsmpls,2);
            while i2 <= batchsize
                batch(:,i1:i2-1) = nxtsmpls;
                lbls(:,i1:i2-1) = nxtlbls;
                [nxtsmpls,nxtlbls] = obj.getSequence();
                i1 = i2;
                i2 = i1 + size(nxtsmpls,2);
            end
            batch(:,i1:end) = nxtsmpls(:,1:batchsize-i1+1);
            lbls(:,i1:end) = nxtlbls(:,1:batchsize-i1+1);
            obj.Stt.nxtsmpls = nxtsmpls(:,batchsize-i1+2:end);
            obj.Stt.nxtlbls = nxtlbls(:,batchsize-i1+2:end);
        end
        % Get next batch of samples with categorical label
        function [x,y] = getCategoricalBatch(obj,batchsize,classif)
            [x,lbls] = obj.getBatch(batchsize);
            ys = eye(obj.AllNLbls(classif));
            y = ys(:,lbls(classif,:));
        end
        % Get next batch of samples and internal states
        function [batch,lbls,stts] = getStateBatch(obj,batchsize)
            [batch,lbls] = obj.getBatch(batchsize);
            stts = cell(1,batchsize);
        end
    end
    
    % Data obtention methods (abstract)
    methods (Abstract,Access = protected)
        % Get sample sequence
        [smpls,lbls] = getSequence(obj)
    end
    
    % Internal general purpose static methods
    methods (Static,Access = protected)
        % Set property AllNLbls
        function allnlbls = setAllNLbls(alllbls)
            allnlbls = zeros(size(alllbls));
            for i = 1:length(alllbls)
                allnlbls(i) = length(fieldnames(alllbls{i}));
            end
        end
    end
    
    % Visualization methods
    methods
        % Input plot generation
        function input_plot = getInputPlot(obj,input)
            input_plot = InputData.getInputPlotStatic(input,obj.Dims);
        end
    end
    
    % Static visualization methods
    methods (Static)
        % Input plot generation
        function input_plot = getInputPlotStatic(input,dims)
            if length(dims) == 2
                input_plot = reshape(input,dims);
            else
                input_plot = input.*ones(length(input),length(input));
            end
        end
    end
    
end