function [ new_struct_inst ] = class_to_struct ( new_class_inst )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    new_struct_inst = new_class_inst;
    
    
    if (numel(new_struct_inst) > 1)
        if iscell(new_struct_inst)
            for i = 1:numel(new_struct_inst)
                new_struct_inst{i} = class_to_struct(new_struct_inst{i});
            end
        else
            for i = 1:numel(new_struct_inst)
                new_struct_inst(i) = class_to_struct(new_struct_inst(i));
            end
        end
    elseif isobject(new_struct_inst) | isstruct(new_struct_inst)
        if isobject(new_struct_inst)
            new_struct_inst = struct(new_struct_inst);
        end
        new_struct_inst_fields = fieldnames(new_struct_inst);
        for i = 1:numel(new_struct_inst_fields)
            disp(inputname(1))
            disp(new_struct_inst_fields{i})
            new_struct_inst.(new_struct_inst_fields{i}) = class_to_struct(new_struct_inst.(new_struct_inst_fields{i}));
        end
    end
    
end

