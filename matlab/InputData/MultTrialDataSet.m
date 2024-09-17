% Abstract class from which all multiple-trial dataset inputs inherit

classdef MultTrialDataSet < DataSet
    
    % Public constants
    properties (Constant)
        Range = [0.1,0.9]
    end
    
    % Main public methods
    methods
        % Hard restart (resplit in sets)
        function hardRestart(obj,params)
            if isfield(params,'splitprops')
                obj.Data.splitprops = params.splitprops;
            end
            nsets = nnz(params.resplit);
            nlbls = zeros(1,length(obj.AllLbls));
            for i = 1:length(obj.AllLbls)
                nlbls(i) = length(fieldnames(obj.AllLbls{i}));
            end
            smpls_setlbl = zeros([nsets,nlbls]);
            resplitsets = [obj.Data.splitsets{params.resplit}];
            set_idxs = zeros(size(resplitsets));
            for i = randperm(length(resplitsets))
                lblcell = num2cell(obj.Data.trials{resplitsets(i)}.lbl);
                [~,set_idxs(i)] = min(smpls_setlbl(:,lblcell{:})./obj.Data.splitprops(params.resplit)');
                smpls_setlbl(set_idxs(i),lblcell{:}) = smpls_setlbl(set_idxs(i),lblcell{:}) + 1;
            end
            newsplitsets = cell(1,nsets);
            for i = 1:nsets
                newsplitsets{i} = resplitsets(set_idxs == i);
            end
            resplitposs = find(params.resplit);
            for i = 1:nsets
                obj.Data.splitsets{resplitposs(i)} = newsplitsets{i};
            end
            obj.restart(params);
        end
    end
    
    % Data initialization methods
    methods (Access = protected)
        % Input data initialization
        function initData(obj,~)
            obj.readDataset();
            obj.Size = size(obj.Data.trials{1}.x,1);
            obj.Dims = obj.Size;
            obj.removeNans();
            obj.normData();
            obj.splitInSets();
        end
        % Remove trials containing NaN or Inf values
        function removeNans(obj)
            for i = length(obj.Data.trials):-1:1
                if any(isnan(obj.Data.trials{i}.x(:))) || any(isinf(obj.Data.trials{i}.x(:)))
                    obj.Data.trials(i) = [];
                end
            end
        end
        % Normalize input between 0.1 and 0.9
        function normData(obj)
            n = 0;
            s = zeros(obj.Size,1);
            s2 = zeros(obj.Size,1);
            for i = 1:length(obj.Data.trials)
                n = n + size(obj.Data.trials{i}.x,2);
                s = s + sum(obj.Data.trials{i}.x,2);
                s2 = s2 + sum(obj.Data.trials{i}.x.^2,2);
            end
            mns = s/n;
            pstds = 3 * sqrt((n*s2-s.^2)/(n*(n-1))); % var = sum((x-s/n).^2)/(n-1) = (s2+n*(s/n).^2-2*s*s/n)/(n-1) = (n*s2-s.^2)/(n*(n-1))
            for i = 1:length(obj.Data.trials)
                obj.Data.trials{i}.x = obj.Data.trials{i}.x - mns;
                obj.Data.trials{i}.x = max(min(obj.Data.trials{i}.x,pstds),-pstds)./pstds;
                obj.Data.trials{i}.x = (obj.Data.trials{i}.x + 1) * 0.4 + 0.1;
            end
        end
        % Split data into sets (using stratification)
        function splitInSets(obj)
            nsets = length(obj.Data.splitprops);
            nlbls = zeros(1,length(obj.AllLbls));
            for i = 1:length(obj.AllLbls)
                nlbls(i) = length(fieldnames(obj.AllLbls{i}));
            end
            smpls_setlbl = zeros([nsets,nlbls]);
            set_idxs = zeros(size(obj.Data.trials));
            for i = 1:length(obj.Data.trials)
                lblcell = num2cell(obj.Data.trials{i}.lbl);
                [~,set_idxs(i)] = min(smpls_setlbl(:,lblcell{:})./obj.Data.splitprops(:));
                smpls_setlbl(set_idxs(i),lblcell{:}) = smpls_setlbl(set_idxs(i),lblcell{:}) + 1;
            end
            obj.Data.splitsets = cell(1,nsets);
            for i = 1:nsets
                obj.Data.splitsets{i} = find(set_idxs == i);
            end
        end
    end
    
    % Data obtention methods
    methods (Access = protected)
        % Get sample sequence
        function [smpls,lbls] = getSequence(obj)
            trial = obj.Data.trials{obj.Stt.splitset(randi(length(obj.Stt.splitset)))};
            smpls = trial.x;
            lbls = trial.lbl.*ones(1,size(smpls,2));
        end
    end
    
end