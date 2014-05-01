function [ new_struct_inst ] = class_to_struct ( new_class_inst )
%class_to_struct  Takes anything and turns any class objects it contains to
%struct instances.

    % If nothing else, we return the same thing we started with.
    new_struct_inst = new_class_inst;
    
    % We break groups of objects to deal with them separately.
    if (numel(new_struct_inst) > 1)
        % Cells require {} for indexing. Otherwise, we don't extract their
        % contents. Everything else is the same.
        if iscell(new_struct_inst)
            for i = 1:numel(new_struct_inst)
                new_struct_inst{i} = class_to_struct(new_struct_inst{i});
            end
        else
            for i = 1:numel(new_struct_inst)
                new_struct_inst(i) = class_to_struct(new_struct_inst(i));
            end
        end
    % We take objects and structs and deal with them the same way.
    elseif isobject(new_struct_inst) | isstruct(new_struct_inst)
        if isobject(new_struct_inst)
            % We save the classname for later so we can read it back if we
            % want to.
            old_classname = class(new_struct_inst);
            new_struct_inst = struct(new_struct_inst);
            new_struct_inst.classname = old_classname;
        end
        
        % Take all fields and apply class_to_struct on them recursively.
        new_struct_inst_fields = fieldnames(new_struct_inst);
        for i = 1:numel(new_struct_inst_fields)
            new_struct_inst.(new_struct_inst_fields{i}) = class_to_struct(new_struct_inst.(new_struct_inst_fields{i}));
        end
    end
    
end

