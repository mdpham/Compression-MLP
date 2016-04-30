classdef mlp
  properties
    BIAS_IH;
    WEIGHTS_IH;
    HIDDEN;
    BIAS_HO;
    WEIGHTS_HO;
    OUTPUT;
    PATTERN;

    DELTA_O;
    DELTA_H;
    DELTA_BIAS_IH;
    DELTA_BIAS_HO;
    DELTA_WEIGHTS_IH;
    DELTA_WEIGHTS_HO;
  end
  methods
    function obj=set.BIAS_IH(obj,value)
      obj.BIAS_IH=value;
    end
    function obj=set.WEIGHTS_IH(obj,value)
      obj.WEIGHTS_IH=value;
    end
    function obj=set.HIDDEN(obj,value)
      obj.HIDDEN=value;
    end
    function obj=set.BIAS_HO(obj,value)
      obj.BIAS_HO=value;
    end
    function obj=set.WEIGHTS_HO(obj,value)
      obj.WEIGHTS_HO=value;
    end
    function obj=set.OUTPUT(obj,value)
      obj.OUTPUT=value;
    end
    function obj=set.DELTA_O(obj,value)
      obj.DELTA_O=value;
    end
    function obj=set.DELTA_H(obj,value)
      obj.DELTA_H=value;
    end
    function obj=set.DELTA_WEIGHTS_IH(obj,value)
        obj.DELTA_WEIGHTS_IH=value;
    end
    % SSE: calculate current sum squared error of network
    function sse=SSE(obj)
      target=obj.PATTERN(:,2);
      err=target-obj.OUTPUT;
      sse=0.5*err'*err;
    end
    % COST: return number of values we need to store from network
    function info=COST(obj)
      info=numel(obj.BIAS_IH)+numel(obj.WEIGHTS_IH)+numel(obj.WEIGHTS_HO)+numel(obj.BIAS_HO)+numel(obj.HIDDEN);
    end
  end
end
